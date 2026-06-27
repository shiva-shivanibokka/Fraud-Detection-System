"""
fp_growth_rules.py
==================
Uber RADAR Protect-style automatic fraud rule generation using FP-Growth
frequent itemset mining.

Pipeline
--------
1. **Discretisation** — convert raw numeric/categorical transaction fields
   into meaningful string "items" (e.g. ``amt_high``, ``hour_night``).

2. **FP-Growth mining** — run ``mlxtend.frequent_patterns.fpgrowth`` over the
   one-hot encoded item matrix built from *confirmed fraud* transactions to
   discover item combinations that co-occur frequently in fraud.

3. **Association rules** — derive rules via
   ``mlxtend.frequent_patterns.association_rules``.  Filter by minimum
   confidence and lift to isolate strong fraud signals.

4. **Human-readable formatting** — convert each rule row into a natural-
   language blocking condition suitable for analyst review dashboards.

5. **Transaction scoring** — for any new transaction, determine which active
   rules it triggers and surface them to a real-time decisioning layer.

6. **Persistence** — save / load the active rule set as JSON for easy sharing
   with rule-engine microservices.

Typical usage
-------------
>>> import pandas as pd
>>> from fp_growth_rules import (
...     discretize_for_fpgrowth,
...     mine_fraud_rules,
...     format_rule_for_display,
...     score_transaction_against_rules,
...     save_rules_to_json,
...     load_rules_from_json,
... )

>>> df = pd.read_parquet("transactions.parquet")
>>> fraud_df = df[df["is_fraud"] == 1]
>>> disc = discretize_for_fpgrowth(df, fraud_df)
>>> rules = mine_fraud_rules(disc, min_support=0.02, min_confidence=0.6, min_lift=2.0)
>>> for _, rule in rules.iterrows():
...     print(format_rule_for_display(rule))
>>> save_rules_to_json(rules, "active_rules.json")
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

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
_DEFAULT_RULES_PATH = Path("models") / "fraud_rules.json"

# Dollar thresholds for amount buckets
AMT_LOW_THRESHOLD: float = 50.0
AMT_HIGH_THRESHOLD: float = 500.0

# Hour boundaries for time-of-day buckets (24h clock)
NIGHT_START: int = 22  # 10 PM
NIGHT_END: int = 6  # 6  AM

# Minimum Haversine distance (km) for "geo_far"
GEO_FAR_KM: float = 50.0

# Number of top fraud states to include as individual items
TOP_N_STATES: int = 20

# Earth's radius for Haversine calculation
_EARTH_RADIUS_KM: float = 6371.0


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in kilometres between two (lat, lon) points.

    Uses the Haversine formula.  Returns NaN if any coordinate is NaN / None.
    """
    try:
        if any(math.isnan(v) for v in [lat1, lon1, lat2, lon2]):
            return float("nan")
    except (TypeError, ValueError):
        return float("nan")

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------


def discretize_for_fpgrowth(
    full_df: pd.DataFrame,
    fraud_df: Optional[pd.DataFrame] = None,
    top_n_states: int = TOP_N_STATES,
) -> pd.DataFrame:
    """
    Convert raw transaction fields into a DataFrame of string "items" ready
    for one-hot encoding and FP-Growth mining.

    One row per transaction; each column contains a string item token that
    represents a discretised value of a feature.  Columns that cannot be
    computed (e.g. missing lat/lon) are set to ``None`` and are excluded
    during one-hot encoding.

    Expected input columns (all optional except those starred)
    -----------------------------------------------------------
    * ``amt``                   : float — transaction amount USD
    * ``trans_date_trans_time`` : str / datetime — transaction timestamp
      ``category``              : str — merchant category
      ``state``                 : str — billing state
      ``lat``                   : float — cardholder latitude
      ``long``                  : float — cardholder longitude
      ``merch_lat``             : float — merchant latitude
      ``merch_long``            : float — merchant longitude

    Parameters
    ----------
    full_df : pd.DataFrame
        Full transactions dataset (used to determine top fraud states when
        ``fraud_df`` is provided).
    fraud_df : pd.DataFrame, optional
        Subset of *full_df* containing only confirmed fraud transactions.
        If ``None``, *full_df* itself is discretised (useful for scoring).
    top_n_states : int
        Include only the ``top_n_states`` most frequent states in fraud as
        explicit items; rarer states become ``"state_other"``.

    Returns
    -------
    pd.DataFrame
        Index-aligned with *fraud_df* (or *full_df* if ``fraud_df`` is None).
        String-valued columns: ``amt_bucket``, ``hour_bucket``,
        ``category``, ``is_weekend``, ``state``, ``geo_bucket``.
        Cells that cannot be computed are ``None``.
    """
    target_df = fraud_df if fraud_df is not None else full_df
    target_df = target_df.copy()

    result = pd.DataFrame(index=target_df.index)

    # ------------------------------------------------------------------
    # 1. Amount bucket
    # ------------------------------------------------------------------
    if "amt" in target_df.columns:
        amt = target_df["amt"].astype(float)
        result["amt_bucket"] = np.select(
            [amt < AMT_LOW_THRESHOLD, amt <= AMT_HIGH_THRESHOLD],
            ["amt_low", "amt_medium"],
            default="amt_high",
        )
    else:
        log.warning("Column 'amt' not found; skipping amt_bucket.")
        result["amt_bucket"] = None

    # ------------------------------------------------------------------
    # 2. Hour bucket
    # ------------------------------------------------------------------
    if "trans_date_trans_time" in target_df.columns:
        ts = pd.to_datetime(
            target_df["trans_date_trans_time"],
            errors="coerce",
        )
        hour = ts.dt.hour
        # Night: [22, 24) ∪ [0, 6)
        is_night = (hour >= NIGHT_START) | (hour < NIGHT_END)
        result["hour_bucket"] = np.where(is_night, "hour_night", "hour_day")
        result.loc[ts.isna(), "hour_bucket"] = None
    else:
        log.warning("Column 'trans_date_trans_time' not found; skipping hour_bucket.")
        result["hour_bucket"] = None

    # ------------------------------------------------------------------
    # 3. Raw category
    # ------------------------------------------------------------------
    if "category" in target_df.columns:
        result["category"] = (
            target_df["category"]
            .astype(str)
            .str.strip()
            .str.lower()
            .where(target_df["category"].notna(), other=None)
        )
        # Prefix to keep item namespace clean
        result["category"] = result["category"].apply(
            lambda v: f"cat_{v}" if v is not None and v != "nan" else None
        )
    else:
        log.warning("Column 'category' not found; skipping category.")
        result["category"] = None

    # ------------------------------------------------------------------
    # 4. Weekend flag
    # ------------------------------------------------------------------
    if "trans_date_trans_time" in target_df.columns:
        ts = pd.to_datetime(
            target_df["trans_date_trans_time"],
            errors="coerce",
        )
        day_of_week = ts.dt.dayofweek  # Monday=0, Sunday=6
        result["is_weekend"] = np.where(day_of_week >= 5, "weekend", "weekday")
        result.loc[ts.isna(), "is_weekend"] = None
    else:
        result["is_weekend"] = None

    # ------------------------------------------------------------------
    # 5. State (top-N fraud states; rest → "state_other")
    # ------------------------------------------------------------------
    if "state" in target_df.columns:
        # Determine top states from fraud data if available
        fraud_states: pd.Series = fraud_df["state"] if fraud_df is not None else full_df["state"]
        top_states: Set[str] = set(fraud_states.value_counts().head(top_n_states).index.tolist())

        def _state_item(s: Any) -> Optional[str]:
            if pd.isna(s):
                return None
            s_str = str(s).strip().upper()
            return f"state_{s_str}" if s_str in top_states else "state_other"

        result["state"] = target_df["state"].apply(_state_item)
    else:
        log.warning("Column 'state' not found; skipping state.")
        result["state"] = None

    # ------------------------------------------------------------------
    # 6. Geo bucket (Haversine distance between cardholder and merchant)
    # ------------------------------------------------------------------
    geo_cols = {"lat", "long", "merch_lat", "merch_long"}
    if geo_cols.issubset(target_df.columns):

        def _geo_bucket(row: pd.Series) -> Optional[str]:
            d = _haversine_km(row["lat"], row["long"], row["merch_lat"], row["merch_long"])
            if math.isnan(d):
                return None
            return "geo_far" if d >= GEO_FAR_KM else "geo_near"

        result["geo_bucket"] = target_df.apply(_geo_bucket, axis=1)
    else:
        missing_geo = geo_cols - set(target_df.columns)
        log.warning("Geo columns missing (%s); skipping geo_bucket.", missing_geo)
        result["geo_bucket"] = None

    return result


# ---------------------------------------------------------------------------
# FP-Growth mining
# ---------------------------------------------------------------------------


def _build_onehot(disc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the discretised string DataFrame into a boolean one-hot DataFrame
    suitable for ``mlxtend.frequent_patterns``.

    Each unique non-null string value across all columns becomes a binary
    indicator column.

    Parameters
    ----------
    disc_df : pd.DataFrame
        Output of :func:`discretize_for_fpgrowth`.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame; columns are individual item strings.
    """
    # Build a list of item-sets (one per transaction)
    item_sets: List[List[str]] = []
    for _, row in disc_df.iterrows():
        items = [str(v) for v in row.values if v is not None and str(v) != "nan"]
        item_sets.append(items)

    te = TransactionEncoder()
    te_array = te.fit_transform(item_sets)
    onehot_df = pd.DataFrame(te_array, columns=te.columns_, index=disc_df.index)
    return onehot_df


def mine_fraud_rules(
    disc_df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.6,
    min_lift: float = 2.0,
    max_antecedent_len: int = 5,
) -> pd.DataFrame:
    """
    Run FP-Growth on a discretised fraud transaction DataFrame and return
    high-signal association rules.

    Parameters
    ----------
    disc_df : pd.DataFrame
        Output of :func:`discretize_for_fpgrowth` (fraud transactions only).
    min_support : float
        Minimum fraction of fraud transactions that must contain an itemset.
        Lower values find rarer (but potentially important) patterns.
    min_confidence : float
        Minimum rule confidence: P(consequent | antecedent).
    min_lift : float
        Minimum rule lift: confidence / P(consequent).  Values > 1.0 indicate
        positive correlation; > 2.0 is a strong fraud signal.
    max_antecedent_len : int
        Discard rules whose antecedent exceeds this many items (avoids overly
        specific, low-recall rules).

    Returns
    -------
    pd.DataFrame
        Rules DataFrame with columns:
        ``antecedents``, ``consequents``, ``support``, ``confidence``,
        ``lift``, ``leverage``, ``conviction``, ``support_count``,
        sorted by ``lift`` descending.

    Notes
    -----
    The "consequent" for all rules is a dummy ``FRAUD`` item injected into
    every transaction row so that FP-Growth rules point toward fraud.  Only
    rules whose consequents contain ``FRAUD`` are returned.
    """
    if disc_df.empty:
        log.warning("Empty discretised DataFrame; returning empty rules.")
        return pd.DataFrame()

    # Inject a "FRAUD" item into every row so we can mine rules → FRAUD
    fraud_disc = disc_df.copy()
    fraud_disc["__target__"] = "FRAUD"

    log.info("Building one-hot encoding for %d transactions …", len(fraud_disc))
    onehot = _build_onehot(fraud_disc)
    log.info("One-hot shape: %s (%d items)", onehot.shape, onehot.shape[1])

    log.info("Running FP-Growth (min_support=%.4f) …", min_support)
    itemsets = fpgrowth(onehot, min_support=min_support, use_colnames=True, max_len=None)
    log.info("Found %d frequent itemsets.", len(itemsets))

    if itemsets.empty:
        log.warning("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame()

    log.info("Generating association rules (min_confidence=%.2f) …", min_confidence)
    # num_itemsets required in mlxtend >= 0.23
    try:
        rules = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(onehot),
        )
    except TypeError:
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    log.info("Generated %d raw rules.", len(rules))

    if rules.empty:
        log.warning("No rules met the confidence threshold.")
        return pd.DataFrame()

    log.info("Rules columns available: %s", rules.columns.tolist())

    # Filter: consequent must contain FRAUD
    fraud_rules = rules[rules["consequents"].apply(lambda x: "FRAUD" in x)].copy()
    if fraud_rules.empty:
        log.warning("No rules with FRAUD as consequent. Returning top rules by lift instead.")
        fraud_rules = rules.copy()

    fraud_rules = fraud_rules[fraud_rules["lift"] >= min_lift].copy()
    if "antecedents" in fraud_rules.columns:
        fraud_rules = fraud_rules[
            fraud_rules["antecedents"].apply(len) <= max_antecedent_len
        ].copy()
        fraud_rules = fraud_rules[
            fraud_rules["antecedents"].apply(lambda x: "FRAUD" not in x)
        ].copy()
    rules = fraud_rules

    if rules.empty:
        log.warning("No rules after filtering.")
        return pd.DataFrame()

    # Add absolute support count — robustly find the support column
    for sup_col in ["support", "antecedent support", "consequent support"]:
        if sup_col in rules.columns:
            rules = rules.copy()
            rules["support_count"] = (rules[sup_col] * len(onehot)).round().astype(int)
            break
    else:
        rules["support_count"] = 0

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    log.info(
        "After filtering (lift>=%.1f, confidence>=%.2f): %d rules retained.",
        min_lift,
        min_confidence,
        len(rules),
    )
    return rules


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------


def format_rule_for_display(rule_row: pd.Series) -> str:
    """
    Convert a single rule row from :func:`mine_fraud_rules` output into a
    human-readable blocking condition string.

    Parameters
    ----------
    rule_row : pd.Series
        A single row from the rules DataFrame returned by
        :func:`mine_fraud_rules`.

    Returns
    -------
    str
        Example:
        ``"IF category=misc_net AND hour=night AND amt=high THEN FRAUD (confidence=0.82, lift=3.40, support=147 txns)"``
    """

    def _item_to_condition(item: str) -> str:
        """Map an internal item token to a readable condition phrase."""
        if item.startswith("cat_"):
            return f"category={item[4:]}"
        if item.startswith("amt_"):
            level = item[4:]  # low / medium / high
            return f"amt={level}"
        if item == "hour_night":
            return "hour=night"
        if item == "hour_day":
            return "hour=day"
        if item == "weekend":
            return "day=weekend"
        if item == "weekday":
            return "day=weekday"
        if item.startswith("state_"):
            return f"state={item[6:]}"
        if item == "geo_near":
            return "geo=near"
        if item == "geo_far":
            return "geo=far"
        # Fallback: return raw item
        return item

    antecedents: frozenset = rule_row["antecedents"]
    conditions = " AND ".join(sorted(_item_to_condition(i) for i in antecedents))

    confidence = rule_row["confidence"]
    lift = rule_row["lift"]
    support_count = int(rule_row.get("support_count", 0))

    return (
        f"IF {conditions} THEN FRAUD "
        f"(confidence={confidence:.2f}, lift={lift:.2f}, support={support_count} txns)"
    )


# ---------------------------------------------------------------------------
# Real-time transaction scoring
# ---------------------------------------------------------------------------


def score_transaction_against_rules(
    txn_dict: Dict[str, Any],
    rules_df: pd.DataFrame,
) -> List[str]:
    """
    Check which active fraud rules a new transaction triggers.

    Parameters
    ----------
    txn_dict : dict
        A single transaction represented as a flat dictionary.  Expected keys
        mirror the columns used in :func:`discretize_for_fpgrowth`:
        ``amt``, ``trans_date_trans_time``, ``category``, ``state``,
        ``lat``, ``long``, ``merch_lat``, ``merch_long``.
        Missing keys are treated as ``None``.
    rules_df : pd.DataFrame
        Active rules DataFrame from :func:`mine_fraud_rules` or
        :func:`load_rules_from_json`.

    Returns
    -------
    list of str
        Human-readable strings for each rule the transaction triggers,
        sorted by lift descending.  Empty list if no rules fire.

    Examples
    --------
    >>> txn = {"amt": 620.0, "category": "misc_net", "trans_date_trans_time": "2023-07-14 23:00:00"}
    >>> triggered = score_transaction_against_rules(txn, rules_df)
    >>> for r in triggered:
    ...     print(r)
    """
    if rules_df.empty:
        return []

    # Discretise the single transaction
    txn_series = pd.Series(txn_dict)
    single_df = txn_series.to_frame().T.reset_index(drop=True)
    disc = discretize_for_fpgrowth(single_df, fraud_df=None)

    # Build the item set for this transaction
    txn_items: Set[str] = {str(v) for v in disc.iloc[0].values if v is not None and str(v) != "nan"}

    triggered: List[pd.Series] = []
    for _, rule in rules_df.iterrows():
        antecedents: frozenset = rule["antecedents"]
        # A rule fires when ALL antecedent items are present in the transaction
        if antecedents.issubset(txn_items):
            triggered.append(rule)

    if not triggered:
        return []

    triggered_df = pd.DataFrame(triggered).sort_values("lift", ascending=False)
    return [format_rule_for_display(row) for _, row in triggered_df.iterrows()]


# ---------------------------------------------------------------------------
# Persistence — save / load rules as JSON
# ---------------------------------------------------------------------------


def save_rules_to_json(
    rules_df: pd.DataFrame,
    path: str | Path = _DEFAULT_RULES_PATH,
) -> str:
    """
    Persist the active rules DataFrame to a JSON file.

    ``frozenset`` antecedent/consequent columns are serialised as sorted
    lists so they are human-readable and portable.

    Parameters
    ----------
    rules_df : pd.DataFrame
        Rules DataFrame from :func:`mine_fraud_rules`.
    path : str | Path
        Output file path.

    Returns
    -------
    str
        Absolute path where the rules were saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = rules_df.copy()
    for col in ["antecedents", "consequents"]:
        if col in serialisable.columns:
            serialisable[col] = serialisable[col].apply(
                lambda fs: sorted(fs) if isinstance(fs, (frozenset, set)) else list(fs)
            )

    records = serialisable.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, default=str)

    log.info("Saved %d rules to: %s", len(rules_df), path.resolve())
    return str(path.resolve())


def load_rules_from_json(path: str | Path = _DEFAULT_RULES_PATH) -> pd.DataFrame:
    """
    Load a previously saved rules JSON file back into a DataFrame.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file written by :func:`save_rules_to_json`.

    Returns
    -------
    pd.DataFrame
        Rules DataFrame with ``antecedents`` and ``consequents`` restored as
        ``frozenset`` objects.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    rules_df = pd.DataFrame(records)
    for col in ["antecedents", "consequents"]:
        if col in rules_df.columns:
            rules_df[col] = rules_df[col].apply(
                lambda v: frozenset(v) if isinstance(v, list) else v
            )

    log.info("Loaded %d rules from: %s", len(rules_df), path)
    return rules_df


# ---------------------------------------------------------------------------
# __main__ — smoke-test with synthetic data
# ---------------------------------------------------------------------------


def _make_mock_fraud_transactions(
    n_fraud: int = 1_000,
    n_legit: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic transactions DataFrame for smoke-testing.

    Fraud transactions are intentionally biased toward high-amount, night-time,
    ``misc_net`` category, far-geography combinations to ensure the miner
    finds at least some rules.
    """
    rng = np.random.default_rng(random_state)
    states = ["NY", "CA", "TX", "FL", "NV", "IL", "WA", "OH", "GA", "AZ"]
    categories = [
        "misc_net",
        "grocery_pos",
        "gas_transport",
        "entertainment",
        "misc_pos",
    ]

    def _make_rows(n: int, is_fraud: int) -> List[Dict]:
        rows = []
        for _ in range(n):
            if is_fraud:
                amt = float(
                    rng.choice(
                        [
                            rng.uniform(500, 2000),  # mostly high
                            rng.uniform(50, 499),
                        ],
                        p=[0.7, 0.3],
                    )
                )
                hour = int(
                    rng.choice(
                        list(range(22, 24)) + list(range(0, 6)) + list(range(6, 22)),
                        p=[1 / 24] * 8 + [1 / 24 * (16 / 16)] * 16,
                    )
                )
                cat = rng.choice(categories, p=[0.5, 0.15, 0.1, 0.15, 0.1])
                state = rng.choice(states[:5])  # concentrated in 5 states
                card_lat = float(rng.uniform(30, 45))
                card_lon = float(rng.uniform(-120, -75))
                merch_lat = card_lat + float(rng.uniform(1.0, 5.0))  # always far
                merch_lon = card_lon + float(rng.uniform(1.0, 5.0))
            else:
                amt = float(rng.exponential(80))
                hour = int(rng.integers(0, 24))
                cat = rng.choice(categories)
                state = rng.choice(states)
                card_lat = float(rng.uniform(30, 45))
                card_lon = float(rng.uniform(-120, -75))
                merch_lat = card_lat + float(rng.uniform(0, 0.2))  # nearby
                merch_lon = card_lon + float(rng.uniform(0, 0.2))

            ts = pd.Timestamp("2023-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 365)),
                hours=hour,
            )
            rows.append(
                {
                    "card_id": f"CARD_{rng.integers(0, 500):04d}",
                    "device_id": f"DEV_{rng.integers(0, 200):04d}",
                    "ip_address": f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.1",
                    "amt": round(amt, 2),
                    "category": cat,
                    "state": state,
                    "trans_date_trans_time": ts,
                    "lat": card_lat,
                    "long": card_lon,
                    "merch_lat": merch_lat,
                    "merch_long": merch_lon,
                    "is_fraud": is_fraud,
                }
            )
        return rows

    fraud_rows = _make_rows(n_fraud, is_fraud=1)
    legit_rows = _make_rows(n_legit, is_fraud=0)
    df = pd.DataFrame(fraud_rows + legit_rows)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys

    log.info("=== FP-Growth Fraud Rule Mining — smoke test ===")

    # 1. Generate mock data
    df = _make_mock_fraud_transactions(n_fraud=1_200, n_legit=600)
    log.info(
        "Mock dataset: %d total  (%d fraud, %d legit)",
        len(df),
        df["is_fraud"].sum(),
        (df["is_fraud"] == 0).sum(),
    )

    fraud_df = df[df["is_fraud"] == 1].copy()

    # 2. Discretise
    log.info("--- Step 1: Discretising fraud transactions ---")
    disc = discretize_for_fpgrowth(df, fraud_df=fraud_df)
    log.info("Discretised shape: %s", disc.shape)
    log.info("Sample rows:\n%s", disc.head(3).to_string())

    # 3. Mine rules
    log.info("--- Step 2: Mining fraud rules (FP-Growth) ---")
    rules = mine_fraud_rules(disc, min_support=0.02, min_confidence=0.5, min_lift=1.5)

    if rules.empty:
        log.warning("No rules mined. Check min_support / min_confidence thresholds.")
        sys.exit(0)

    log.info("Top 10 rules by lift:")
    for _, row in rules.head(10).iterrows():
        log.info("  %s", format_rule_for_display(row))

    # 4. Save / load
    log.info("--- Step 3: Save / load rules ---")
    rules_path = save_rules_to_json(rules, "models/fraud_rules_test.json")
    loaded_rules = load_rules_from_json(rules_path)
    log.info("Loaded %d rules from JSON.", len(loaded_rules))

    # 5. Score a sample suspicious transaction
    log.info("--- Step 4: Scoring a high-risk transaction ---")
    suspicious_txn = {
        "amt": 850.0,
        "category": "misc_net",
        "trans_date_trans_time": "2023-06-15 23:30:00",
        "state": "NY",
        "lat": 40.7128,
        "long": -74.0060,
        "merch_lat": 42.3601,  # ~250 km away → geo_far
        "merch_long": -71.0589,
    }
    triggered = score_transaction_against_rules(suspicious_txn, loaded_rules)
    log.info("Triggered rules for suspicious transaction (%d):", len(triggered))
    for rule_str in triggered:
        log.info("  >> %s", rule_str)

    # 6. Score a benign transaction
    log.info("--- Step 5: Scoring a low-risk transaction ---")
    benign_txn = {
        "amt": 12.50,
        "category": "grocery_pos",
        "trans_date_trans_time": "2023-06-15 14:00:00",
        "state": "WI",
        "lat": 43.0731,
        "long": -89.4012,
        "merch_lat": 43.0741,  # very close
        "merch_long": -89.4022,
    }
    benign_triggered = score_transaction_against_rules(benign_txn, loaded_rules)
    log.info(
        "Triggered rules for benign transaction: %d",
        len(benign_triggered),
    )
    if benign_triggered:
        for rule_str in benign_triggered:
            log.info("  >> %s", rule_str)
    else:
        log.info("  (none — transaction appears low risk)")

    log.info("=== Smoke test complete ===")
