"""
Export feature ranges for DICE counterfactuals
==============================================
Sprint 3 — DICE generates counterfactual explanations ("what minimal change
would flip this decision?"). DICE only needs feature *ranges* (min/max), not
the training data, so we export a small models/cf_ranges.json. Serving builds
a DICE explainer from this — no training data shipped to production.

Only a few features are *actionable* / interpretable to vary (amount, geo
distance, hour); the rest (velocity, graph, GNN embeddings) are held fixed.

Run:  python -m src.model.export_cf_ranges
"""

import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.train import get_feature_cols  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Features a fraud analyst can meaningfully ask "what if" about.
VARY_FEATURES = ["amt", "geo_distance_km", "hour"]


def main() -> dict:
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_features.parquet"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    feature_cols = [c for c in feature_cols if c in train.columns] or get_feature_cols(train)

    X = train[feature_cols].fillna(0)
    ranges = {c: [float(X[c].min()), float(X[c].max())] for c in feature_cols}

    out = {
        "outcome_name": "is_fraud",
        "vary_features": [f for f in VARY_FEATURES if f in feature_cols],
        "ranges": ranges,
    }
    path = os.path.join(MODELS_DIR, "cf_ranges.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[cf_ranges] Saved {path} with {len(ranges)} feature ranges; "
          f"vary={out['vary_features']}")
    return out


if __name__ == "__main__":
    main()
