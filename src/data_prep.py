"""
Data Preparation Pipeline
==========================
Loads the Credit Card Fraud Transaction dataset, synthesizes entity identifiers
(device IDs, IP addresses) needed for graph construction, and produces a clean
enriched DataFrame saved as Parquet.

Provenance — this data is simulated, not real
  The source files (fraudTrain.csv / fraudTest.csv) are the Kaggle "Sparkov"
  credit-card fraud set, which is produced by a transaction generator rather
  than collected from real payments. Metrics computed here should be read as
  evidence that the pipeline is correct, not as production fraud performance.
  (The separate Elliptic module under src/graph_fraud/ *does* use real data.)

Why synthesize device/IP fields?
  The raw dataset contains card numbers and merchant IDs but no device or IP
  fields — these would come from the payment page in production. We synthesize
  them so the entity-resolution and graph stages have something to resolve:
    - Shared device IDs across cards (fraud rings share devices)
    - Shared IP prefixes across related accounts (household / VPN / proxy)
    - A minority of fraud cards collude in rings and share devices/IPs
  Because ring membership is drawn using the label, these fields correlate with
  fraud by construction. Graph features built on them are therefore optimistic
  — see synthesize_entity_fields() for the full leakage note.

Temporal split:
  We use a strict time-based split (Jan 2019 – Dec 2019 = train,
  Jan 2020 – Jun 2020 = test). Random splits cause label leakage
  in fraud because the same card appears in both sets.
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw(subsample: float = 1.0) -> pd.DataFrame:
    """Load and combine train/test splits into one chronological DataFrame."""
    train_path = os.path.join(RAW_DIR, "fraudTrain.csv")
    test_path = os.path.join(RAW_DIR, "fraudTest.csv")

    print("[data] Loading raw transaction files...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    df = pd.concat([train, test], ignore_index=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Parse timestamps
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values("trans_dt").reset_index(drop=True)

    if subsample < 1.0:
        df = (
            df.sample(frac=subsample, random_state=42)
            .sort_values("trans_dt")
            .reset_index(drop=True)
        )
        print(f"[data] Subsampled to {len(df):,} rows ({subsample:.0%})")
    else:
        print(
            f"[data] Loaded {len(df):,} total transactions | Fraud rate: {df.is_fraud.mean():.4f}"
        )

    return df


def synthesize_entity_fields(
    df: pd.DataFrame, seed: int = 42, ring_fraction: float = 0.25
) -> pd.DataFrame:
    """
    Synthesize device IDs and IP addresses with realistic fraud-ring properties.

    Statistical properties modelled on production fraud data:
      - Each card uses 1-2 devices (most use 1 primary device)
      - Fraud rings: a *minority* of fraud cards collude and share a device/IP
      - Legitimate cards occasionally share devices (household)
      - IP prefixes are shared within geographic areas (lat/long bucketed)

    Leakage note (read before trusting any graph-derived metric)
    -----------------------------------------------------------
    These fields do not exist in the raw dataset — they are simulated, and ring
    membership is drawn using ``is_fraud``. That means device/IP correlate with
    the label *by construction*, so any graph feature built on them is optimistic
    relative to production, where entities are observed rather than generated.

    An earlier version of this function made that correlation *deterministic*:
    fraud and legit cards drew devices from disjoint slices of the pool, and
    fraud cards were handed a dedicated ``192.168.*`` range while everyone else
    got ``10.*``. Either one alone lets a model recover the label exactly, which
    is what produced the implausible AUC-ROC of ~0.997. Three properties now
    keep the signal earned rather than free:

      1. One shared device pool — ring devices are drawn from the same pool every
         card draws from, so a device ID never identifies a class on its own.
      2. Only ``ring_fraction`` of fraud cards join rings. Most card fraud is
         one-off compromise, not collusion, so the rest look like ordinary cards.
      3. No fraud-specific address space — every card is addressed out of the
         same geographic ``10.*`` scheme; ring members merely share one of those
         prefixes, as a shared VPN or proxy would look in reality.

    Rings remain genuinely detectable via co-occurrence structure, which is the
    point of the graph stage — but a model now has to do the work to find them.
    See ``tests/test_data_prep_leakage.py``, which fails if separability returns.
    """
    rng = np.random.default_rng(seed)
    cards = df["cc_num"].unique()
    merchants = df["merchant"].unique()

    print("[data] Synthesizing device IDs and IP addresses...")

    # --- Device ID assignment ---
    # Pool of devices: ~65% of unique cards (many cards per device for fraud rings)
    n_devices = max(100, int(len(cards) * 0.65))
    device_pool = np.array([f"dev_{i:06d}" for i in range(n_devices)])

    fraud_cards = df.loc[df["is_fraud"] == 1, "cc_num"].unique()

    # Only a minority of fraud cards belong to organised rings.
    n_ring_cards = int(len(fraud_cards) * ring_fraction)
    ring_cards = (
        rng.choice(fraud_cards, size=n_ring_cards, replace=False)
        if n_ring_cards > 0
        else np.array([], dtype=fraud_cards.dtype)
    )
    n_fraud_rings = max(1, len(ring_cards) // 5)
    # Drawn from the whole pool: a ring device is not marked by its identity.
    ring_devices = rng.choice(device_pool, size=n_fraud_rings, replace=False)

    card_to_devices = {}
    for i, card in enumerate(ring_cards):
        # Primary device is the ring device; 20% chance of using a second device
        devices = [ring_devices[i % n_fraud_rings]]
        if rng.random() < 0.20:
            devices.append(rng.choice(device_pool))
        card_to_devices[card] = devices

    for card in cards:
        if card in card_to_devices:
            continue
        # Everyone else — legit cards and non-ring fraud alike: 1 primary device;
        # 8% chance of a second (shared household).
        devices = [rng.choice(device_pool)]
        if rng.random() < 0.08:
            devices.append(rng.choice(device_pool))
        card_to_devices[card] = devices

    # Assign a device per transaction (weighted toward primary device)
    def pick_device(cc_num):
        devices = card_to_devices.get(cc_num, [rng.choice(device_pool)])
        if len(devices) == 1:
            return devices[0]
        weights = np.array([0.85] + [0.15 / (len(devices) - 1)] * (len(devices) - 1))
        weights = weights / weights.sum()  # normalize to avoid float precision issues
        return rng.choice(devices, p=weights)

    df["device_id"] = df["cc_num"].map(lambda c: pick_device(c))

    # --- IP Address assignment ---
    # IP prefix (first 3 octets) is geographic (city-level); last octet varies
    # Use lat/long bucketing to assign /24 subnet per geographic area
    lat_bucket = (df["lat"] // 2).astype(int)
    lon_bucket = (df["long"] // 2).astype(int)
    geo_key = lat_bucket.astype(str) + "_" + lon_bucket.astype(str)
    unique_geos = geo_key.unique()
    geo_to_prefix = {g: f"10.{i // 256}.{i % 256}" for i, g in enumerate(unique_geos)}

    # Ring members share a /24 prefix (same VPN/proxy). It is drawn from the same
    # geographic address space everyone else uses — a dedicated range would make
    # the prefix a pure label indicator.
    geo_prefixes = list(geo_to_prefix.values())
    ring_prefixes = rng.choice(geo_prefixes, size=n_fraud_rings, replace=True)
    card_to_prefix = {card: ring_prefixes[i % n_fraud_rings] for i, card in enumerate(ring_cards)}

    def get_ip(row):
        if row["cc_num"] in card_to_prefix:
            prefix = card_to_prefix[row["cc_num"]]
        else:
            prefix = geo_to_prefix.get(f"{int(row['lat'] // 2)}_{int(row['long'] // 2)}", "10.0.0")
        return f"{prefix}.{rng.integers(1, 254)}"

    df["ip_address"] = df.apply(get_ip, axis=1)
    df["ip_prefix"] = df["ip_address"].str.rsplit(".", n=1).str[0]

    # --- Merchant device assignment (merchants use POS terminals = devices) ---
    merchant_devices = {m: f"pos_{i:05d}" for i, m in enumerate(merchants)}
    df["merchant_device"] = df["merchant"].map(merchant_devices)

    print(
        f"[data] Entity fields: {df['device_id'].nunique()} devices, "
        f"{df['ip_prefix'].nunique()} IP prefixes"
    )
    return df


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features. Temporal signals are core fraud indicators."""
    df = df.copy()
    df["hour"] = df["trans_dt"].dt.hour
    df["day_of_week"] = df["trans_dt"].dt.dayofweek
    df["month"] = df["trans_dt"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Age at time of transaction
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_dt"] - df["dob"]).dt.days / 365.25
    df["age"] = df["age"].clip(18, 100)

    # Geographic distance between cardholder and merchant
    df["geo_distance_km"] = (
        np.sqrt((df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2) * 111.0
    )  # approx km per degree

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categoricals needed for XGBoost."""
    from sklearn.preprocessing import LabelEncoder

    cat_cols = ["category", "gender", "state", "merchant"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    return df


def temporal_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict temporal split: train on 2019, test on 2020.

    This is the ONLY correct split for fraud data. Random splits cause:
    1. Label leakage — the same card appears in train and test
    2. Distribution mismatch — model sees future fraud patterns during training
    3. Overoptimistic AUC — real-world performance is always worse

    Production fraud teams always use time-based splits with a gap
    (e.g., train Jan-Oct, gap Nov, test Dec) to prevent temporal leakage.
    """
    split_date = pd.Timestamp("2020-01-01")
    train = df[df["trans_dt"] < split_date].copy()
    test = df[df["trans_dt"] >= split_date].copy()
    print(f"[data] Train: {len(train):,} | Test: {len(test):,} | Split date: {split_date.date()}")
    print(
        f"[data] Train fraud rate: {train.is_fraud.mean():.4f} | "
        f"Test fraud rate: {test.is_fraud.mean():.4f}"
    )
    return train, test


def save_processed(df: pd.DataFrame, name: str) -> str:
    """Save processed DataFrame to Parquet."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    print(f"[data] Saved {name}.parquet | {df.shape}")
    return path


def run_data_pipeline(subsample: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full data preparation pipeline."""
    df = load_raw(subsample=subsample)
    df = synthesize_entity_fields(df)
    df = engineer_temporal_features(df)
    df = encode_categoricals(df)

    train, test = temporal_train_test_split(df)

    save_processed(df, "transactions_full")
    save_processed(train, "train")
    save_processed(test, "test")

    return train, test


if __name__ == "__main__":
    train, test = run_data_pipeline(subsample=0.3)  # use 30% for quick dev
    print("\nSample:")
    print(train[["cc_num", "device_id", "ip_prefix", "amt", "is_fraud"]].head())
