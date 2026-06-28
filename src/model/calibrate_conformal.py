"""
Conformal calibration (MAPIE LAC) for the fraud model
=====================================================
Sprint 3 — adds an uncertainty signal to each fraud score with a *coverage
guarantee*, instead of a bare point probability.

Method (Least Ambiguous set-valued Classifier / "lac" split conformal):
  - On a held-out calibration set, the conformity score of each point is
    s_i = 1 - p_model(true_class_i).
  - q_hat = the (1-alpha) conformal quantile of those scores.
  - A class k is in the prediction set iff p_k >= 1 - q_hat =: threshold t.

For binary fraud with t > 0.5 this yields an intuitive **confidence band**:
    p_fraud >= t          -> {fraud}  (confident fraud)
    p_fraud <= 1 - t      -> {legit}  (confident legit)
    1 - t < p_fraud < t   -> {}       (UNCERTAIN -> route to human review)

MAPIE is used here (offline) to fit and to validate empirical coverage. We
export only the scalar threshold to models/conformal.json, so the serving
API applies it with plain arithmetic — no MAPIE dependency in production.

Run:  python -m src.model.calibrate_conformal
"""

import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.train import get_feature_cols  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

CONFIDENCE_LEVEL = 0.90
RANDOM_STATE = 42


def main() -> dict:
    model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_features.parquet"))
    feature_cols = [c for c in feature_cols if c in test.columns] or get_feature_cols(test)

    # Held-out test set -> split into calibration + evaluation (exchangeable).
    X = test[feature_cols].fillna(0).values
    y = test["is_fraud"].astype(int).values
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE
    )

    alpha = 1.0 - CONFIDENCE_LEVEL

    # ---- Manual LAC conformal quantile (the textbook split-conformal rule) ----
    p1_cal = model.predict_proba(X_cal)[:, 1]
    p_true = np.where(y_cal == 1, p1_cal, 1.0 - p1_cal)
    conformity = 1.0 - p_true
    n = len(conformity)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(conformity, q_level, method="higher"))
    threshold = 1.0 - q_hat  # class k in set iff p_k >= threshold

    # ---- Validate empirical coverage with MAPIE (authentic library check) ----
    mapie_coverage = None
    try:
        from mapie.classification import SplitConformalClassifier

        scc = SplitConformalClassifier(
            estimator=model, confidence_level=CONFIDENCE_LEVEL,
            conformity_score="lac", prefit=True,
        )
        scc.conformalize(X_cal, y_cal)
        _, y_set = scc.predict_set(X_eval)
        y_set = np.asarray(y_set)
        if y_set.ndim == 3:  # (n, n_classes, n_levels)
            y_set = y_set[:, :, 0]
        covered = y_set[np.arange(len(y_eval)), y_eval]
        mapie_coverage = float(covered.mean())
    except Exception as exc:
        print(f"[conformal] MAPIE validation skipped ({exc})")

    # ---- Independent coverage + band stats using our exported threshold ----
    p1_eval = model.predict_proba(X_eval)[:, 1]
    in_fraud = p1_eval >= threshold
    in_legit = (1.0 - p1_eval) >= threshold
    set_size = in_fraud.astype(int) + in_legit.astype(int)
    # true label in set?
    label_in_set = np.where(y_eval == 1, in_fraud, in_legit)
    threshold_coverage = float(label_in_set.mean())
    uncertain_rate = float((set_size != 1).mean())

    band = [round(1.0 - threshold, 4), round(threshold, 4)]
    print(f"[conformal] confidence_level={CONFIDENCE_LEVEL}  n_calib={n}")
    print(f"[conformal] q_hat={q_hat:.4f}  threshold t={threshold:.4f}")
    print(f"[conformal] uncertain band (fraud_score in): {band}")
    print(f"[conformal] empirical coverage  threshold={threshold_coverage:.4f}"
          + (f"  mapie={mapie_coverage:.4f}" if mapie_coverage is not None else ""))
    print(f"[conformal] uncertain rate on eval: {uncertain_rate:.4f}")

    out = {
        "method": "lac_split_conformal",
        "confidence_level": CONFIDENCE_LEVEL,
        "alpha": alpha,
        "q_hat": q_hat,
        "threshold": threshold,
        "uncertain_band": band,
        "n_calib": n,
        "empirical_coverage": threshold_coverage,
        "mapie_coverage": mapie_coverage,
        "uncertain_rate": uncertain_rate,
    }
    with open(os.path.join(MODELS_DIR, "conformal.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[conformal] Saved models/conformal.json")
    return out


if __name__ == "__main__":
    main()
