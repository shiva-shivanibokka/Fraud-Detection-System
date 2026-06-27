"""
Ensemble Fraud Classifier — proven XGBoost + Optuna-tuned CatBoost soft-vote
===========================================================================
Sprint 3 model upgrade. Builds on the single-XGBoost baseline (train.py).

Design rationale:
  Soft-vote averaging only helps when the two members are comparably strong
  AND make different errors. An earlier LightGBM member was noticeably weaker
  than XGBoost (~0.874 vs 0.906 AUC-PR), so averaging dragged the ensemble
  BELOW the baseline. CatBoost is a true peer of XGBoost, so it has a real
  chance of adding diversity without diluting quality.

  1. Anchor on the baseline's PROVEN XGBoost params (train.get_xgb_params).
  2. Use Optuna to tune a CatBoost (AUC-PR objective).
  3. Soft-vote the two (averaged probabilities).
  4. Isotonic-calibrate via a prefit holdout (FrozenEstimator).
  5. Evaluate against the CURRENT baseline on the same production metrics, so
     we only promote a genuinely better model.

Saved as fraud_model_ensemble.pkl. NOT promoted over fraud_model.pkl here.

Run:  python -m src.model.train_ensemble
"""

import json
import os
import sys
import warnings

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.train import (  # noqa: E402  reuse production-metric helpers
    dollar_value_captured,
    get_feature_cols,
    get_xgb_params,
    precision_at_k,
    recall_at_fpr,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

HPO_SUBSAMPLE = 200_000
HPO_TRIALS = 20  # CatBoost fits are slower than XGB/LGB
RANDOM_STATE = 42


def _evaluate(y_true, scores, amounts) -> dict:
    """Production-metric bundle for a set of scores."""
    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "auc_pr": float(average_precision_score(y_true, scores)),
        "precision_at_1pct": float(precision_at_k(y_true, scores, 0.01)),
        "precision_at_05pct": float(precision_at_k(y_true, scores, 0.005)),
        "recall_at_01fpr": float(recall_at_fpr(y_true, scores, 0.001)),
        "dollar_capture_rate": float(
            dollar_value_captured(y_true, scores, amounts)["dollar_capture_rate"]
        ),
    }


def _tune_catboost(X, y, scale_pos_weight) -> tuple[dict, int]:
    """Optuna search for CatBoost params on a stratified subsample.

    Returns (best_params, best_iterations).
    """
    if len(y) > HPO_SUBSAMPLE:
        X, _, y, _ = train_test_split(
            X, y, train_size=HPO_SUBSAMPLE, stratify=y, random_state=RANDOM_STATE
        )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "iterations": 1000,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "PRAUC",
            "random_seed": RANDOM_STATE,
            "thread_count": -1,
            "verbose": 0,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=0)
        scores = model.predict_proba(X_val)[:, 1]
        trial.set_user_attr("best_iteration", int(model.get_best_iteration() or 500))
        return average_precision_score(y_val, scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=HPO_TRIALS, show_progress_bar=False)
    best = study.best_params
    best_iter = study.best_trial.user_attrs.get("best_iteration", 500)
    print(f"[ensemble] Optuna CatBoost best AUC-PR={study.best_value:.4f} "
          f"(iterations={best_iter}) params={best}")
    return best, best_iter


def main() -> dict:
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_features.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test_features.parquet"))

    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    feature_cols = [c for c in feature_cols if c in train.columns] or get_feature_cols(train)

    X_train = train[feature_cols].fillna(0).values
    y_train = train["is_fraud"].astype(int).values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["is_fraud"].astype(int).values
    amounts_test = test["amt"].fillna(0).values

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[ensemble] Train {len(y_train):,} rows, {len(feature_cols)} features, "
          f"fraud rate {n_pos / len(y_train):.4f}")

    # ---- Baseline: evaluate the currently-deployed model on the test set ----
    baseline_metrics = None
    try:
        baseline = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
        baseline_metrics = _evaluate(y_test, baseline.predict_proba(X_test)[:, 1], amounts_test)
        print(f"[ensemble] Baseline AUC-PR={baseline_metrics['auc_pr']:.4f}")
    except Exception as exc:
        print(f"[ensemble] Could not evaluate baseline: {exc}")

    # ---- Member 1: proven baseline XGBoost params ----
    xgb_clf = xgb.XGBClassifier(**get_xgb_params(n_pos, n_neg))

    # ---- Member 2: Optuna-tuned CatBoost ----
    print(f"[ensemble] Tuning CatBoost ({HPO_TRIALS} trials)...")
    cb_params, cb_iters = _tune_catboost(X_train, y_train, scale_pos_weight)
    cb_clf = CatBoostClassifier(
        **cb_params,
        iterations=cb_iters,
        scale_pos_weight=scale_pos_weight,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        verbose=0,
    )

    # ---- Soft-vote ensemble: fit on a holdout split, calibrate on the rest ----
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    print("[ensemble] Fitting soft-vote ensemble (proven XGBoost + tuned CatBoost)...")
    voting = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("cat", cb_clf)], voting="soft", n_jobs=1
    )
    voting.fit(X_fit, y_fit)

    print("[ensemble] Calibrating (isotonic, prefit holdout)...")
    calibrated = CalibratedClassifierCV(FrozenEstimator(voting), method="isotonic")
    calibrated.fit(X_cal, y_cal)

    # ---- Evaluate ensemble on the test set ----
    ensemble_metrics = _evaluate(y_test, calibrated.predict_proba(X_test)[:, 1], amounts_test)

    # ---- Report comparison ----
    print("\n[ensemble] ===== Baseline vs Ensemble (test set) =====")
    keys = ["auc_pr", "auc_roc", "precision_at_1pct", "precision_at_05pct",
            "recall_at_01fpr", "dollar_capture_rate"]
    print(f"  {'metric':22s} {'baseline':>10s} {'ensemble':>10s} {'delta':>10s}")
    wins = 0
    for k in keys:
        b = baseline_metrics[k] if baseline_metrics else float("nan")
        e = ensemble_metrics[k]
        if e > b:
            wins += 1
        print(f"  {k:22s} {b:>10.4f} {e:>10.4f} {e - b:>+10.4f}")
    base_pr = baseline_metrics["auc_pr"] if baseline_metrics else 0.0
    print(f"\n[ensemble] ensemble wins {wins}/{len(keys)} metrics; "
          f"AUC-PR delta {ensemble_metrics['auc_pr'] - base_pr:+.4f}")

    # ---- Save ensemble artifacts (NOT promoted over fraud_model.pkl) ----
    joblib.dump(calibrated, os.path.join(MODELS_DIR, "fraud_model_ensemble.pkl"))
    with open(os.path.join(MODELS_DIR, "ensemble_metrics.json"), "w") as f:
        json.dump(
            {"baseline": baseline_metrics, "ensemble": ensemble_metrics,
             "catboost_params": cb_params, "catboost_iterations": cb_iters},
            f, indent=2,
        )
    print("[ensemble] Saved fraud_model_ensemble.pkl + ensemble_metrics.json")
    return {"baseline": baseline_metrics, "ensemble": ensemble_metrics}


if __name__ == "__main__":
    main()
