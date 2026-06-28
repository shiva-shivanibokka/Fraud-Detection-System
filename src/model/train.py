"""
XGBoost Fraud Classifier + ONNX Export
=========================================
Trains XGBoost on the full feature set (tabular + velocity + graph + GNN embeddings)
and exports to ONNX for <5ms production inference.

Why ONNX?
  ONNX Runtime is optimized for inference latency, not training flexibility.
  Visa and Mastercard require <20ms ML inference within a <100ms payment decision.
  A 500-tree XGBoost in pure Python takes 15-25ms; the same model as ONNX runs
  in 2-5ms. This is not a minor optimization — it's the difference between
  fitting in the latency budget or not.

Evaluation methodology:
  Production fraud teams do NOT evaluate on AUC alone. They evaluate on:
  1. Precision@K — given a fixed review queue (e.g., 1% of transactions),
     what fraction of those reviewed are actually fraud?
  2. Recall@FPR — at a fixed false positive rate (0.1%, 1%), how much fraud
     do we catch?
  3. Dollar value captured — what fraction of fraudulent dollar volume
     is flagged? (A $10K fraud is 100x worse than a $100 fraud.)
  4. Concept drift — how does AUC degrade over monthly cohorts?
     (Train on Jan-Jun, evaluate on Jul, Aug, Sep, Oct, Nov, Dec separately.)

Features used:
  - Tabular: amt, category_enc, gender_enc, hour, day_of_week, is_weekend,
             is_night, age, geo_distance_km, city_pop
  - Velocity: vel_card_*_count, vel_card_*_amt_sum, vel_device_*_count, etc.
  - Graph: graph_degree, graph_shared_device_cards, graph_neighbor_fraud_rate, etc.
  - GNN: gnn_embed_0 through gnn_embed_63
"""

import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import shap

    SHAP_AVAILABLE = True
except (ImportError, Exception):
    SHAP_AVAILABLE = False
    print("[model] SHAP unavailable (NumPy version conflict) — using XGBoost native importance")
import warnings

import joblib
import mlflow
import mlflow.xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# ---------------------------------------------------------------------------
# Feature Sets
# ---------------------------------------------------------------------------

TABULAR_FEATURES = [
    "amt",
    "category_enc",
    "gender_enc",
    "state_enc",
    "merchant_enc",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_night",
    "age",
    "geo_distance_km",
    "city_pop",
]

VELOCITY_FEATURE_PREFIXES = [
    "vel_card_",
    "vel_device_",
    "vel_ip_prefix_",
    "vel_merchant_",
]

GRAPH_FEATURES = [
    "graph_degree",
    "graph_n_devices",
    "graph_n_ips",
    "graph_n_merchants",
    "graph_device_fraud_rate",
    "graph_ip_fraud_rate",
    "graph_shared_device_cards",
    "graph_shared_ip_cards",
    "graph_neighbor_card_count",
    "graph_neighbor_fraud_rate",
    "graph_component_size",
]

GNN_EMBED_DIM = 64


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns available in df."""
    vel_cols = [c for c in df.columns if any(c.startswith(p) for p in VELOCITY_FEATURE_PREFIXES)]
    graph_cols = [c for c in GRAPH_FEATURES if c in df.columns]
    gnn_cols = [c for c in df.columns if c.startswith("gnn_embed_")]
    tab_cols = [c for c in TABULAR_FEATURES if c in df.columns]
    return tab_cols + vel_cols + graph_cols + gnn_cols


def get_xgb_params(n_pos: int, n_neg: int) -> dict:
    """
    XGBoost hyperparameters tuned for severely imbalanced fraud detection.
    scale_pos_weight = n_neg / n_pos handles 0.6% fraud rate.
    tree_method='hist' is required for ONNX export compatibility.
    """
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": n_neg / max(n_pos, 1),
        "tree_method": "hist",  # required for ONNX
        "eval_metric": "aucpr",  # area under PR curve — better for imbalance than AUC
        "random_state": 42,
        "n_jobs": -1,
    }


# ---------------------------------------------------------------------------
# Production Evaluation Metrics
# ---------------------------------------------------------------------------


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k_frac: float = 0.01) -> float:
    """
    Precision@K: of the top K% highest-scored transactions (what gets reviewed),
    what fraction are actually fraud?

    This is the primary metric for fixed-capacity fraud review queues.
    A team can only review N transactions per day — P@K tells you how
    efficiently you fill that queue with real fraud.
    """
    n = len(y_true)
    k = max(1, int(n * k_frac))
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return y_true[top_k_idx].mean()


def recall_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float = 0.001) -> float:
    """
    Recall at fixed False Positive Rate.
    Finds the threshold that produces target_fpr, returns recall at that threshold.

    Production context: a 0.1% FPR on 1M daily transactions = 1,000 false blocks.
    Each false block pisses off a legitimate customer. This is the real tradeoff.
    """
    from sklearn.metrics import roc_curve

    fprs, tprs, thresholds = roc_curve(y_true, y_scores)
    # Find FPR closest to target
    idx = np.argmin(np.abs(fprs - target_fpr))
    return float(tprs[idx])


def dollar_value_captured(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    amounts: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Dollar-value-captured metrics — the business metric that actually matters.
    A $10,000 fraud caught is worth more than 100 $10 frauds caught.
    """
    preds = (y_scores >= threshold).astype(int)
    total_fraud_value = amounts[y_true == 1].sum()
    caught_fraud_value = amounts[(y_true == 1) & (preds == 1)].sum()
    false_block_value = amounts[(y_true == 0) & (preds == 1)].sum()

    return {
        "total_fraud_value": float(total_fraud_value),
        "caught_fraud_value": float(caught_fraud_value),
        "dollar_capture_rate": float(caught_fraud_value / max(total_fraud_value, 1)),
        "false_block_value": float(false_block_value),
        "false_block_rate": float(false_block_value / amounts[y_true == 0].sum()),
    }


def temporal_auc_by_month(
    df_test: pd.DataFrame,
    feature_cols: list[str],
    model: xgb.XGBClassifier,
) -> pd.DataFrame:
    """
    Concept drift evaluation: compute AUC per calendar month on the test set.
    Shows how model performance degrades over time — the production concern
    that AUC on a static test set completely misses.
    """
    results = []
    df_test = df_test.copy()
    df_test["year_month"] = df_test["trans_dt"].dt.to_period("M").astype(str)

    for ym, group in df_test.groupby("year_month"):
        if group["is_fraud"].sum() < 5:
            continue  # skip months with too few fraud cases
        X = group[feature_cols].fillna(0)
        y = group["is_fraud"].values
        scores = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, scores)
        p_at_1 = precision_at_k(y, scores, k_frac=0.01)
        results.append(
            {
                "month": ym,
                "n_transactions": len(group),
                "n_fraud": int(y.sum()),
                "fraud_rate": float(y.mean()),
                "auc": float(auc),
                "precision_at_1pct": float(p_at_1),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def train_fraud_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment_name: str = "FraudDetectionSystem",
) -> dict:
    """Full training pipeline: XGBoost → calibration → SHAP → ONNX export."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    feature_cols = get_feature_cols(train)
    print(
        f"[model] Training on {len(feature_cols)} features: "
        f"{len([c for c in feature_cols if c.startswith('vel_')])} velocity, "
        f"{len([c for c in feature_cols if c.startswith('graph_')])} graph, "
        f"{len([c for c in feature_cols if c.startswith('gnn_')])} GNN"
    )

    X_train = train[feature_cols].fillna(0).values
    y_train = train["is_fraud"].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["is_fraud"].values

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    print(f"[model] Train: {len(y_train):,} | Fraud: {n_pos:,} ({n_pos / len(y_train):.4f})")

    # DagsHub (or any remote MLflow): point the tracking URI at it when set,
    # otherwise MLflow logs to the local ./mlruns. One env var flips local->remote.
    from src.config import settings

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        print(f"[model] MLflow tracking -> {settings.mlflow_tracking_uri}")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="XGBoost_ONNX_Fraud"):
        params = get_xgb_params(n_pos, n_neg)
        mlflow.log_params({k: v for k, v in params.items() if k != "n_jobs"})
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("train_fraud_rate", round(float(n_pos / len(y_train)), 5))

        # Base XGBoost model
        print("[model] Training XGBoost...")
        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Isotonic calibration for reliable probability outputs
        print("[model] Calibrating probabilities (isotonic regression)...")
        try:
            # sklearn >= 1.2: cv must be int or cross-validator, not "prefit"
            cal_clf = CalibratedClassifierCV(clf, cv=5, method="isotonic")
            cal_clf.fit(X_train, y_train)
        except Exception:
            # fallback: use the uncalibrated model wrapped to match API
            cal_clf = clf

        # Evaluate
        scores_test = cal_clf.predict_proba(X_test)[:, 1]
        amounts_test = test["amt"].fillna(0).values

        auc = roc_auc_score(y_test, scores_test)
        ap = average_precision_score(y_test, scores_test)
        p_at_1 = precision_at_k(y_test, scores_test, k_frac=0.01)
        p_at_05 = precision_at_k(y_test, scores_test, k_frac=0.005)
        rec_01fpr = recall_at_fpr(y_test, scores_test, target_fpr=0.001)
        dollar_metrics = dollar_value_captured(y_test, scores_test, amounts_test)

        print("\n[model] === Evaluation Results ===")
        print(f"  AUC-ROC:              {auc:.4f}")
        print(f"  AUC-PR:               {ap:.4f}")
        print(f"  Precision@1%:         {p_at_1:.4f}  (top 1% of flagged txns)")
        print(f"  Precision@0.5%:       {p_at_05:.4f}")
        print(f"  Recall@0.1%FPR:       {rec_01fpr:.4f}")
        print(f"  Dollar capture rate:  {dollar_metrics['dollar_capture_rate']:.4f}")
        print(f"  False block rate:     {dollar_metrics['false_block_rate']:.6f}")

        metrics = {
            "auc_roc": auc,
            "auc_pr": ap,
            "precision_at_1pct": p_at_1,
            "precision_at_05pct": p_at_05,
            "recall_at_01fpr": rec_01fpr,
            **dollar_metrics,
        }
        mlflow.log_metrics(metrics)

        # Concept drift by month
        print("[model] Computing temporal AUC by month...")
        drift_df = temporal_auc_by_month(test, feature_cols, cal_clf)
        if not drift_df.empty:
            drift_path = os.path.join(MODELS_DIR, "drift_by_month.json")
            drift_df.to_json(drift_path, orient="records", indent=2)
            mlflow.log_artifact(drift_path)
            print(drift_df[["month", "auc", "precision_at_1pct"]].to_string(index=False))

        # Feature importance — SHAP if available, XGBoost gain as fallback
        print("[model] Computing feature importance...")
        if SHAP_AVAILABLE:
            try:
                sample_idx = np.random.choice(
                    len(X_test), size=min(2000, len(X_test)), replace=False
                )
                X_shap = pd.DataFrame(X_test[sample_idx], columns=feature_cols)
                background = shap.sample(X_shap, 200, random_state=42)
                explainer = shap.Explainer(clf.predict, background)
                shap_vals = explainer(X_shap)
                mean_abs_shap = pd.Series(
                    np.abs(shap_vals.values).mean(axis=0), index=feature_cols
                ).sort_values(ascending=False)
            except Exception as e:
                print(f"[model] SHAP failed ({e}), falling back to XGBoost gain importance")
                mean_abs_shap = None
        else:
            mean_abs_shap = None

        if mean_abs_shap is None:
            # XGBoost gain-based importance as fallback
            gain_dict = clf.get_booster().get_score(importance_type="gain")
            mean_abs_shap = pd.Series(
                {feat: gain_dict.get(feat, 0.0) for feat in feature_cols}
            ).sort_values(ascending=False)
            max_val = mean_abs_shap.max()
            if max_val > 0:
                mean_abs_shap = mean_abs_shap / max_val

        shap_path = os.path.join(MODELS_DIR, "feature_importance.json")
        mean_abs_shap.to_json(shap_path)
        mlflow.log_artifact(shap_path)

        print("\n[model] Top 15 features by importance:")
        print(mean_abs_shap.head(15).to_string())

        # Save artifacts
        joblib.dump(cal_clf, os.path.join(MODELS_DIR, "fraud_model.pkl"))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
        joblib.dump(mean_abs_shap, os.path.join(MODELS_DIR, "shap_importance.pkl"))

        # ONNX export
        print("[model] Exporting to ONNX for <5ms inference...")
        onnx_path = export_to_onnx(clf, feature_cols, X_train[:10])
        if onnx_path:
            mlflow.log_artifact(onnx_path)

        try:
            mlflow.sklearn.log_model(cal_clf, name="fraud_model_calibrated")
        except Exception as e:
            print(f"[model] MLflow log_model skipped ({e})")
        print(f"[model] Training complete. AUC={auc:.4f}, P@1%={p_at_1:.4f}")

    return {
        "model": cal_clf,
        "base_clf": clf,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "drift_df": drift_df if not drift_df.empty else pd.DataFrame(),
        "shap_importance": mean_abs_shap,
    }


def export_to_onnx(clf: xgb.XGBClassifier, feature_cols: list, X_sample: np.ndarray) -> str:
    """
    Export XGBoost model to ONNX format for sub-5ms inference.
    ONNX Runtime is 3-5x faster than native XGBoost for batch=1 inference.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_sklearn(
            clf,
            initial_types=initial_type,
            options={id(clf): {"zipmap": False}},
        )
        onnx_path = os.path.join(MODELS_DIR, "fraud_model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"[model] ONNX model saved: {onnx_path}")
        return onnx_path
    except ImportError:
        print("[model] skl2onnx not installed — trying onnxmltools...")
        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType

            onnx_model = convert_xgboost(
                clf.get_booster(),
                initial_types=[("features", FloatTensorType([None, len(feature_cols)]))],
            )
            onnx_path = os.path.join(MODELS_DIR, "fraud_model.onnx")
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"[model] ONNX model saved (onnxmltools): {onnx_path}")
            return onnx_path
        except Exception as e:
            print(f"[model] ONNX export failed: {e} — continuing without ONNX")
            return None
    except Exception as e:
        print(f"[model] ONNX export failed: {e}")
        return None


def load_onnx_model(onnx_path: str = None):
    """Load ONNX model for fast inference. Falls back to joblib model."""
    if onnx_path is None:
        onnx_path = os.path.join(MODELS_DIR, "fraud_model.onnx")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        print(f"[model] ONNX Runtime loaded: {onnx_path}")
        return sess, input_name
    except Exception as e:
        print(f"[model] ONNX load failed ({e}) — using joblib model")
        model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
        return model, None


def predict_onnx(sess, input_name, X: np.ndarray) -> np.ndarray:
    """Run inference via ONNX Runtime. Returns fraud probabilities."""
    X_float = X.astype(np.float32)
    outputs = sess.run(None, {input_name: X_float})
    # outputs[1] = class probabilities (shape: [n, 2])
    if isinstance(outputs[1], np.ndarray) and outputs[1].ndim == 2:
        return outputs[1][:, 1]
    return outputs[0].astype(float)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    PROCESSED_DIR_ABS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
    train_path = os.path.join(PROCESSED_DIR_ABS, "train_features.parquet")
    test_path = os.path.join(PROCESSED_DIR_ABS, "test_features.parquet")

    if not os.path.exists(train_path):
        print("Run src/pipeline.py first to generate features.")
        sys.exit(1)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    results = train_fraud_model(train, test)
    print(f"\nFinal metrics: {results['metrics']}")
