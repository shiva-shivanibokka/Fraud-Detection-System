"""
DuckDB velocity feature computation (Sprint 3)
==============================================
A drop-in, much faster equivalent of
velocity.feature_store.compute_velocity_features_offline.

The pandas version uses groupby + time-based rolling windows
(rolling(window, closed="left")) — past transactions within each window,
excluding the current row. DuckDB expresses the exact same thing as a single
range-window query:

    <agg>(amt) OVER (
        PARTITION BY <entity> ORDER BY ts
        RANGE BETWEEN <wsec> PRECEDING AND 1 PRECEDING
    )

which is SQL on columnar data — 10-50x faster than per-group Python rolling on
large transaction tables, with no server (DuckDB is in-process).

This is offline/training-only. It is verified to produce *identical* features
to the pandas version (see tests/test_velocity_duckdb.py), so the trained model
and serving are unaffected.
"""

import numpy as np
import pandas as pd

try:
    from src.velocity.feature_store import WINDOWS
except ImportError:  # imported as `velocity.velocity_duckdb` (pipeline sys.path)
    from velocity.feature_store import WINDOWS

DEFAULT_ENTITY_COLS = {
    "card": "cc_num",
    "device": "device_id",
    "ip_prefix": "ip_prefix",
    "merchant": "merchant",
}


def _build_sql(entity_cols: dict) -> str:
    selects = ["_rid"]
    windows = []
    for etype, col in entity_cols.items():
        for wname, wsec in WINDOWS.items():
            w = f"w_{etype}_{wname}"
            windows.append(
                f'{w} AS (PARTITION BY "{col}" ORDER BY _ts '
                f"RANGE BETWEEN {wsec} PRECEDING AND 1 PRECEDING)"
            )
            selects.append(f"COUNT(amt) OVER {w} AS vel_{etype}_{wname}_count")
            selects.append(f"COALESCE(SUM(amt) OVER {w}, 0) AS vel_{etype}_{wname}_amt_sum")
            selects.append(f"COALESCE(MAX(amt) OVER {w}, 0) AS vel_{etype}_{wname}_amt_max")
    return (
        f"SELECT {', '.join(selects)} FROM t "
        f"WINDOW {', '.join(windows)} ORDER BY _rid"
    )


def compute_velocity_features_duckdb(
    df: pd.DataFrame,
    entity_cols: dict | None = None,
) -> pd.DataFrame:
    """Compute the 60 velocity features with DuckDB; identical to the pandas
    offline version. Returns df (sorted by trans_dt) with vel_* columns added."""
    import duckdb

    entity_cols = entity_cols or DEFAULT_ENTITY_COLS

    work = df.copy().sort_values("trans_dt").reset_index(drop=True)
    work["_rid"] = np.arange(len(work), dtype=np.int64)
    # Epoch seconds, resolution-independent: parquet may store trans_dt as
    # datetime64[us] (not [ns]), so force a known resolution before casting.
    work["_ts"] = work["trans_dt"].astype("datetime64[s]").astype(np.int64)

    cols = ["_rid", "_ts", "amt", *dict.fromkeys(entity_cols.values())]
    t = work[cols]  # noqa: F841 - referenced by name in the DuckDB SQL

    con = duckdb.connect()
    try:
        con.register("t", t)
        vel = con.execute(_build_sql(entity_cols)).df()
    finally:
        con.close()

    vel = vel.sort_values("_rid").drop(columns=["_rid"]).reset_index(drop=True)
    out = pd.concat([work.drop(columns=["_ts"]), vel], axis=1)
    vel_cols = [c for c in out.columns if c.startswith("vel_")]
    out[vel_cols] = out[vel_cols].fillna(0)
    out = out.drop(columns=["_rid"], errors="ignore")
    return out
