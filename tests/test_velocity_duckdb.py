"""DuckDB velocity features must exactly match the pandas reference."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")

from src.velocity.feature_store import (
    compute_velocity_features_offline,
    get_velocity_feature_names,
)
from src.velocity.velocity_duckdb import compute_velocity_features_duckdb


def _synthetic(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01")
    # Mix of small and large time gaps (seconds..days) to exercise all windows,
    # plus deliberate duplicate timestamps (ties).
    secs = np.cumsum(rng.choice([0, 5, 45, 300, 4000, 90000], size=n))
    return pd.DataFrame({
        "trans_num": [f"t{i}" for i in range(n)],
        "trans_dt": [base + pd.Timedelta(seconds=int(s)) for s in secs],
        "cc_num": rng.choice(["C1", "C2", "C3"], size=n),
        "device_id": rng.choice(["D1", "D2"], size=n),
        "ip_prefix": rng.choice(["1.1", "2.2"], size=n),
        "merchant": rng.choice(["M1", "M2", "M3"], size=n),
        "amt": np.round(rng.uniform(5, 900, size=n), 2),
    })


def test_duckdb_matches_pandas_velocity():
    df = _synthetic()
    vel_cols = get_velocity_feature_names()

    a = compute_velocity_features_offline(df).set_index("trans_num")[vel_cols].sort_index()
    b = compute_velocity_features_duckdb(df).set_index("trans_num")[vel_cols].sort_index()
    b = b.loc[a.index]

    max_abs = (a - b).abs().to_numpy().max()
    assert max_abs < 1e-6, f"DuckDB diverges from pandas (max abs diff {max_abs})"
