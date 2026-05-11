"""
fraud_rings.py
==============
Stripe-style fraud ring detection using pairwise similarity learning and graph
connected components.

Pipeline
--------
1. **Similarity learning** — train an XGBoost binary classifier on *pairs* of
   payment cards.  Each pair is described by pairwise features that capture
   whether the two cards share devices, IP prefixes, geography, merchants, and
   behavioural patterns.  The model outputs P(same fraud ring | pair features).

2. **Graph construction** — treat every card as a node.  For every pair whose
   similarity score exceeds a configurable threshold, add a weighted edge.

3. **Connected components** — run NetworkX ``connected_components`` over the
   graph to find clusters of mutually-linked cards.  Each cluster receives a
   unique ``ring_id``.

4. **Ring statistics** — aggregate per-ring metrics for analyst review.

Typical usage
-------------
>>> import pandas as pd
>>> from fraud_rings import generate_training_pairs, train_similarity_model, detect_fraud_rings, get_ring_stats

>>> df = pd.read_parquet("transactions.parquet")
>>> pairs = generate_training_pairs(df)
>>> model = train_similarity_model(pairs)
>>> labeled = detect_fraud_rings(df, model, threshold=0.7)
>>> stats = get_ring_stats(labeled)
>>> print(stats.head())
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAIR_FEATURE_COLS: List[str] = [
    "same_device",
    "same_ip_prefix",
    "same_state",
    "merchant_jaccard",
    "amt_similarity",
    "time_overlap",
    "both_fraud",
]

MODEL_VERSION = "1.0.0"
_DEFAULT_MODEL_PATH = Path("models") / "fraud_ring_similarity.joblib"

# ---------------------------------------------------------------------------
# Internal helpers — card-level aggregation
# ---------------------------------------------------------------------------


def _build_card_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw transaction rows into one row per ``card_id``.

    Expected columns in *df*
    ------------------------
    card_id        : str / int  — unique card identifier
    device_id      : str        — device used for the transaction
    ip_address     : str        — IP address (IPv4, dotted-quad)
    state          : str        — billing state
    merchant       : str        — merchant name / ID
    amt            : float      — transaction amount in USD
    trans_date_trans_time : str / datetime — transaction timestamp
    is_fraud       : int        — 1 = confirmed fraud, 0 = legitimate

    Returns a DataFrame indexed by ``card_id`` with one row per card.
    """
    required = {
        "card_id",
        "device_id",
        "ip_address",
        "state",
        "merchant",
        "amt",
        "trans_date_trans_time",
        "is_fraud",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], errors="coerce"
    )

    profiles = (
        df.groupby("card_id")
        .agg(
            devices=("device_id", lambda x: set(x.dropna())),
            ip_prefixes=(
                "ip_address",
                lambda x: {
                    ".".join(str(ip).split(".")[:3])
                    for ip in x.dropna()
                    if "." in str(ip)
                },
            ),
            state=("state", lambda x: x.mode().iloc[0] if len(x) > 0 else "UNK"),
            merchants=("merchant", lambda x: set(x.dropna())),
            avg_amt=("amt", "mean"),
            min_ts=("trans_date_trans_time", "min"),
            max_ts=("trans_date_trans_time", "max"),
            fraud_flag=("is_fraud", "max"),  # 1 if ANY transaction is fraud
            n_txns=("amt", "count"),
        )
        .reset_index()
    )
    return profiles


def _jaccard(set_a: set, set_b: set) -> float:
    """Return Jaccard similarity between two sets; returns 0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _amt_similarity(avg_a: float, avg_b: float) -> float:
    """
    Return 1 - normalised absolute difference in average transaction amounts.
    Ranges from 0.0 (completely different) to 1.0 (identical).
    """
    denom = max(avg_a, avg_b)
    if denom == 0:
        return 1.0
    return float(1.0 - abs(avg_a - avg_b) / denom)


def _time_ranges_overlap(
    min_a: pd.Timestamp,
    max_a: pd.Timestamp,
    min_b: pd.Timestamp,
    max_b: pd.Timestamp,
) -> int:
    """Return 1 if the two [min, max] date ranges overlap, else 0."""
    if pd.isna(min_a) or pd.isna(max_a) or pd.isna(min_b) or pd.isna(max_b):
        return 0
    return int(min_a <= max_b and min_b <= max_a)


# ---------------------------------------------------------------------------
# Pair feature extraction
# ---------------------------------------------------------------------------


def _compute_pair_features(row_a: pd.Series, row_b: pd.Series) -> Dict[str, float]:
    """
    Compute the seven pairwise features between two card profile rows.

    Parameters
    ----------
    row_a, row_b : pd.Series
        Rows from the card-profile DataFrame produced by :func:`_build_card_profiles`.

    Returns
    -------
    dict
        Keys match ``PAIR_FEATURE_COLS``.
    """
    same_device = int(bool(row_a["devices"] & row_b["devices"]))
    same_ip = int(bool(row_a["ip_prefixes"] & row_b["ip_prefixes"]))
    same_state = int(row_a["state"] == row_b["state"])
    merch_j = _jaccard(row_a["merchants"], row_b["merchants"])
    amt_sim = _amt_similarity(row_a["avg_amt"], row_b["avg_amt"])
    t_overlap = _time_ranges_overlap(
        row_a["min_ts"],
        row_a["max_ts"],
        row_b["min_ts"],
        row_b["max_ts"],
    )
    both_fraud = int(row_a["fraud_flag"] == 1 and row_b["fraud_flag"] == 1)

    return {
        "same_device": same_device,
        "same_ip_prefix": same_ip,
        "same_state": same_state,
        "merchant_jaccard": merch_j,
        "amt_similarity": amt_sim,
        "time_overlap": t_overlap,
        "both_fraud": both_fraud,
    }


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------


def generate_training_pairs(
    df: pd.DataFrame,
    max_positive_pairs: int = 20_000,
    max_negative_pairs: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a labelled dataset of card pairs for training the similarity model.

    Labelling heuristic
    -------------------
    **Positive (label = 1)** — both cards are fraudulent AND share at least one
    device or /24 IP prefix.  These are the strongest signal that two cards
    belong to the same fraud ring.

    **Negative (label = 0)** — one of the following:
      * One fraud card + one legitimate card that happen to share a device
        (e.g. a stolen device used once then reported).
      * Two legitimate cards that share a device (household members, shared
        work device).

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions DataFrame.
    max_positive_pairs : int
        Cap on the number of positive training pairs (sampled if exceeded).
    max_negative_pairs : int
        Cap on the number of negative training pairs.
    random_state : int
        NumPy / pandas random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``card_id_a``, ``card_id_b``, all PAIR_FEATURE_COLS, ``label``.
    """
    rng = np.random.default_rng(random_state)
    profiles = _build_card_profiles(df)
    log.info("Built %d card profiles.", len(profiles))

    fraud_cards = profiles[profiles["fraud_flag"] == 1].set_index("card_id")
    legit_cards = profiles[profiles["fraud_flag"] == 0].set_index("card_id")

    records: List[Dict] = []

    # ------------------------------------------------------------------
    # Positive pairs: both fraud, share device or IP prefix
    # ------------------------------------------------------------------
    fraud_ids = fraud_cards.index.tolist()
    log.info("Generating positive pairs from %d fraud cards …", len(fraud_ids))

    candidate_positives: List[Tuple[str, str]] = []
    for a, b in itertools.combinations(fraud_ids, 2):
        ra, rb = fraud_cards.loc[a], fraud_cards.loc[b]
        if (ra["devices"] & rb["devices"]) or (ra["ip_prefixes"] & rb["ip_prefixes"]):
            candidate_positives.append((a, b))

    if len(candidate_positives) > max_positive_pairs:
        idx = rng.choice(
            len(candidate_positives), size=max_positive_pairs, replace=False
        )
        candidate_positives = [candidate_positives[i] for i in idx]

    for a, b in candidate_positives:
        feats = _compute_pair_features(fraud_cards.loc[a], fraud_cards.loc[b])
        feats["card_id_a"] = a
        feats["card_id_b"] = b
        feats["label"] = 1
        records.append(feats)

    log.info("Collected %d positive pairs.", len(candidate_positives))

    # ------------------------------------------------------------------
    # Negative pairs
    # ------------------------------------------------------------------
    log.info("Generating negative pairs …")

    # Build a device → card_ids index for fast lookup
    device_to_cards: Dict[str, List[str]] = {}
    for _, row in profiles.iterrows():
        for dev in row["devices"]:
            device_to_cards.setdefault(dev, []).append(row["card_id"])

    candidate_negatives: List[Tuple[str, str, int, int]] = []  # (a, b, flag_a, flag_b)
    for dev, card_list in device_to_cards.items():
        if len(card_list) < 2:
            continue
        for a, b in itertools.combinations(card_list, 2):
            fa = int(profiles.set_index("card_id").loc[a, "fraud_flag"])
            fb = int(profiles.set_index("card_id").loc[b, "fraud_flag"])
            if fa == fb == 1:
                continue  # already handled as positive (or ambiguous)
            candidate_negatives.append((a, b, fa, fb))

    if len(candidate_negatives) > max_negative_pairs:
        idx = rng.choice(
            len(candidate_negatives), size=max_negative_pairs, replace=False
        )
        candidate_negatives = [candidate_negatives[i] for i in idx]

    profiles_idx = profiles.set_index("card_id")
    for a, b, _fa, _fb in candidate_negatives:
        if a not in profiles_idx.index or b not in profiles_idx.index:
            continue
        feats = _compute_pair_features(profiles_idx.loc[a], profiles_idx.loc[b])
        feats["card_id_a"] = a
        feats["card_id_b"] = b
        feats["label"] = 0
        records.append(feats)

    log.info("Collected %d negative pairs.", len(candidate_negatives))

    pairs_df = pd.DataFrame(records)
    pairs_df = pairs_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    log.info(
        "Total training pairs: %d  (pos=%d, neg=%d)",
        len(pairs_df),
        pairs_df["label"].sum(),
        (pairs_df["label"] == 0).sum(),
    )
    return pairs_df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_similarity_model(
    pairs_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    xgb_params: Optional[Dict] = None,
) -> XGBClassifier:
    """
    Train an XGBoost binary classifier to predict whether two cards share a
    fraud ring.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Output of :func:`generate_training_pairs`.
    test_size : float
        Fraction of pairs held out for evaluation.
    random_state : int
        Random seed.
    xgb_params : dict, optional
        Override default XGBoost hyper-parameters.

    Returns
    -------
    XGBClassifier
        Fitted model.  Call ``model.predict_proba(X)[:, 1]`` to get similarity
        scores.
    """
    default_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "scale_pos_weight": 1.0,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if xgb_params:
        default_params.update(xgb_params)

    X = pairs_df[PAIR_FEATURE_COLS].astype(float)
    y = pairs_df["label"].astype(int)

    # Rebalance scale_pos_weight
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos > 0:
        default_params["scale_pos_weight"] = n_neg / n_pos
        log.info(
            "scale_pos_weight set to %.2f (neg=%d / pos=%d)",
            default_params["scale_pos_weight"],
            n_neg,
            n_pos,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = XGBClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluation
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(
        y_test, y_pred, target_names=["different_ring", "same_ring"]
    )
    log.info("Test ROC-AUC: %.4f", auc)
    log.info("Classification report:\n%s", report)

    # Feature importances
    importances = dict(zip(PAIR_FEATURE_COLS, model.feature_importances_))
    log.info("Feature importances: %s", {k: f"{v:.4f}" for k, v in importances.items()})

    return model


# ---------------------------------------------------------------------------
# Fraud ring detection
# ---------------------------------------------------------------------------


def detect_fraud_rings(
    df: pd.DataFrame,
    model: XGBClassifier,
    threshold: float = 0.7,
    min_ring_size: int = 2,
) -> pd.DataFrame:
    """
    Run end-to-end fraud ring detection on a transactions DataFrame.

    Steps
    -----
    1. Build card-level profiles.
    2. For every fraud–fraud card pair that shares a device or IP, compute
       pairwise features and score with *model*.
    3. Build a NetworkX graph; add an edge for every pair with score ≥ *threshold*.
    4. Find connected components; assign a ``ring_id`` to each card in a
       component of size ≥ *min_ring_size*.
    5. Merge ``ring_id`` back onto the original transactions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions.
    model : XGBClassifier
        Fitted similarity model from :func:`train_similarity_model`.
    threshold : float
        Minimum similarity score to draw an edge (default 0.7).
    min_ring_size : int
        Minimum number of cards to constitute a "ring" (default 2).

    Returns
    -------
    pd.DataFrame
        Original *df* with an additional ``ring_id`` column.
        Cards not assigned to any ring have ``ring_id = "NO_RING"``.
    """
    profiles = _build_card_profiles(df)
    fraud_profiles = profiles[profiles["fraud_flag"] == 1].set_index("card_id")

    log.info(
        "Scoring pairs for %d fraud cards (threshold=%.2f) …",
        len(fraud_profiles),
        threshold,
    )

    G = nx.Graph()
    G.add_nodes_from(fraud_profiles.index.tolist())

    pair_records: List[Dict] = []
    card_ids_a: List[str] = []
    card_ids_b: List[str] = []

    fraud_ids = fraud_profiles.index.tolist()
    for a, b in itertools.combinations(fraud_ids, 2):
        ra, rb = fraud_profiles.loc[a], fraud_profiles.loc[b]
        if not (
            (ra["devices"] & rb["devices"]) or (ra["ip_prefixes"] & rb["ip_prefixes"])
        ):
            continue
        feats = _compute_pair_features(ra, rb)
        pair_records.append(feats)
        card_ids_a.append(a)
        card_ids_b.append(b)

    if pair_records:
        X_pairs = pd.DataFrame(pair_records)[PAIR_FEATURE_COLS].astype(float)
        scores = model.predict_proba(X_pairs)[:, 1]

        edges_added = 0
        for a, b, score in zip(card_ids_a, card_ids_b, scores):
            if score >= threshold:
                G.add_edge(a, b, weight=float(score))
                edges_added += 1

        log.info(
            "Evaluated %d candidate pairs → %d edges added (threshold=%.2f).",
            len(pair_records),
            edges_added,
            threshold,
        )
    else:
        log.warning(
            "No candidate pairs found (no shared devices/IPs among fraud cards)."
        )

    # Connected components → ring IDs
    card_to_ring: Dict[str, str] = {}
    ring_counter = 1
    for component in nx.connected_components(G):
        if len(component) >= min_ring_size:
            ring_label = f"RING_{ring_counter:04d}"
            for card in component:
                card_to_ring[card] = ring_label
            ring_counter += 1

    log.info(
        "Identified %d fraud rings (min_size=%d).", ring_counter - 1, min_ring_size
    )

    # Merge back onto card profiles
    profiles["ring_id"] = profiles["card_id"].map(card_to_ring).fillna("NO_RING")

    # Merge ring_id onto the transaction-level DataFrame
    result_df = df.copy()
    result_df = result_df.merge(
        profiles[["card_id", "ring_id"]],
        on="card_id",
        how="left",
    )
    result_df["ring_id"] = result_df["ring_id"].fillna("NO_RING")
    return result_df


# ---------------------------------------------------------------------------
# Ring statistics
# ---------------------------------------------------------------------------


def get_ring_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ring summary statistics from a DataFrame that has been
    enriched by :func:`detect_fraud_rings`.

    Parameters
    ----------
    df : pd.DataFrame
        Transactions DataFrame with a ``ring_id`` column.

    Returns
    -------
    pd.DataFrame
        One row per ring (excluding "NO_RING"), sorted by ``total_amt`` desc.
        Columns:
        - ``ring_id``
        - ``n_cards``       : distinct cards in ring
        - ``n_txns``        : total transactions
        - ``fraud_rate``    : fraction of transactions flagged as fraud
        - ``total_amt``     : total USD transacted
        - ``avg_amt``       : average transaction amount
        - ``n_states``      : number of distinct billing states
        - ``n_merchants``   : number of distinct merchants
        - ``first_txn``     : earliest transaction timestamp
        - ``last_txn``      : latest transaction timestamp
        - ``span_days``     : days between first and last transaction
    """
    if "ring_id" not in df.columns:
        raise ValueError(
            "DataFrame must have a 'ring_id' column. Run detect_fraud_rings first."
        )

    ring_df = df[df["ring_id"] != "NO_RING"].copy()
    if ring_df.empty:
        log.warning("No fraud rings found in the DataFrame.")
        return pd.DataFrame()

    ring_df["trans_date_trans_time"] = pd.to_datetime(
        ring_df["trans_date_trans_time"], errors="coerce"
    )

    stats = (
        ring_df.groupby("ring_id")
        .agg(
            n_cards=("card_id", "nunique"),
            n_txns=("amt", "count"),
            fraud_rate=("is_fraud", "mean"),
            total_amt=("amt", "sum"),
            avg_amt=("amt", "mean"),
            n_states=("state", "nunique"),
            n_merchants=("merchant", "nunique"),
            first_txn=("trans_date_trans_time", "min"),
            last_txn=("trans_date_trans_time", "max"),
        )
        .reset_index()
    )

    stats["span_days"] = (
        (stats["last_txn"] - stats["first_txn"]).dt.days.fillna(0).astype(int)
    )
    stats = stats.sort_values("total_amt", ascending=False).reset_index(drop=True)
    return stats


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_similarity_model(
    model: XGBClassifier,
    path: str | Path = _DEFAULT_MODEL_PATH,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Persist the fitted similarity model to disk using joblib.

    Parameters
    ----------
    model : XGBClassifier
        Fitted model to save.
    path : str | Path
        Output file path (will create parent directories).
    metadata : dict, optional
        Arbitrary metadata to bundle alongside the model (e.g. threshold,
        training date, dataset version).

    Returns
    -------
    str
        Absolute path where the model was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_cols": PAIR_FEATURE_COLS,
        "model_version": MODEL_VERSION,
        "metadata": metadata or {},
    }
    joblib.dump(payload, path)
    log.info("Similarity model saved to: %s", path.resolve())
    return str(path.resolve())


def load_similarity_model(
    path: str | Path = _DEFAULT_MODEL_PATH,
) -> Tuple[XGBClassifier, Dict]:
    """
    Load a previously saved similarity model from disk.

    Parameters
    ----------
    path : str | Path
        Path to the joblib file produced by :func:`save_similarity_model`.

    Returns
    -------
    (XGBClassifier, dict)
        The fitted model and its metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = joblib.load(path)
    model: XGBClassifier = payload["model"]
    metadata: Dict = payload.get("metadata", {})
    log.info("Loaded similarity model v%s from: %s", payload.get("model_version"), path)
    return model, metadata


# ---------------------------------------------------------------------------
# __main__ — smoke-test with synthetic data
# ---------------------------------------------------------------------------


def _make_mock_transactions(n: int = 5_000, random_state: int = 0) -> pd.DataFrame:
    """Generate a synthetic transactions DataFrame for testing."""
    rng = np.random.default_rng(random_state)

    n_cards = max(50, n // 20)
    n_devices = max(20, n // 50)
    n_merchants = 30
    states = ["CA", "TX", "NY", "FL", "WA", "IL", "OH", "GA", "NC", "MI"]
    categories = [
        "grocery_pos",
        "misc_net",
        "gas_transport",
        "entertainment",
        "misc_pos",
    ]

    card_ids = [f"CARD_{i:04d}" for i in range(n_cards)]
    device_ids = [f"DEV_{i:04d}" for i in range(n_devices)]
    merchant_ids = [f"MERCH_{i:03d}" for i in range(n_merchants)]

    # Designate ~10 % of cards as part of two fraud rings
    ring1_cards = card_ids[:5]
    ring2_cards = card_ids[5:10]
    fraud_cards = set(ring1_cards + ring2_cards)

    rows = []
    for _ in range(n):
        cid = rng.choice(card_ids)
        is_fraud_card = cid in fraud_cards

        # Fraud ring cards share devices
        if cid in ring1_cards:
            dev = rng.choice(device_ids[:3])
            ip = f"192.168.{rng.integers(0, 3)}.{rng.integers(1, 255)}"
        elif cid in ring2_cards:
            dev = rng.choice(device_ids[3:6])
            ip = f"10.0.{rng.integers(0, 3)}.{rng.integers(1, 255)}"
        else:
            dev = rng.choice(device_ids)
            ip = f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"

        ts = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
        amt = float(rng.exponential(150))
        is_fraud = int(is_fraud_card and rng.random() < 0.8)

        rows.append(
            {
                "card_id": cid,
                "device_id": dev,
                "ip_address": ip,
                "state": rng.choice(states),
                "merchant": rng.choice(merchant_ids),
                "category": rng.choice(categories),
                "amt": round(amt, 2),
                "trans_date_trans_time": ts,
                "is_fraud": is_fraud,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    log.info("=== Fraud Ring Detection — smoke test ===")

    df = _make_mock_transactions(n=4_000)
    log.info(
        "Mock dataset: %d transactions, %d fraud (%.1f%%)",
        len(df),
        df["is_fraud"].sum(),
        100 * df["is_fraud"].mean(),
    )

    # --- Step 1: generate training pairs ---
    log.info("--- Step 1: Generating training pairs ---")
    pairs = generate_training_pairs(
        df, max_positive_pairs=2_000, max_negative_pairs=2_000
    )
    log.info(
        "Pairs shape: %s  label distribution:\n%s",
        pairs.shape,
        pairs["label"].value_counts(),
    )

    if pairs["label"].sum() == 0:
        log.error(
            "No positive pairs generated — check fraud card device-sharing logic."
        )
        sys.exit(1)

    # --- Step 2: train model ---
    log.info("--- Step 2: Training similarity model ---")
    model = train_similarity_model(pairs)

    # --- Step 3: detect rings ---
    log.info("--- Step 3: Detecting fraud rings ---")
    labeled_df = detect_fraud_rings(df, model, threshold=0.5)
    ring_counts = labeled_df["ring_id"].value_counts()
    log.info("Ring assignments:\n%s", ring_counts.head(10))

    # --- Step 4: ring stats ---
    log.info("--- Step 4: Ring statistics ---")
    stats = get_ring_stats(labeled_df)
    if not stats.empty:
        log.info("Ring stats:\n%s", stats.to_string(index=False))
    else:
        log.warning("No rings detected at this threshold.")

    # --- Step 5: save / load model ---
    log.info("--- Step 5: Save / load model ---")
    save_path = save_similarity_model(
        model,
        path="models/fraud_ring_test.joblib",
        metadata={"threshold": 0.5, "trained_on": "mock_data"},
    )
    loaded_model, meta = load_similarity_model(save_path)
    log.info("Loaded model metadata: %s", meta)

    log.info("=== Smoke test complete ===")
