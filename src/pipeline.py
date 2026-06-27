"""
Full Pipeline Orchestrator
===========================
Runs all 5 stages end-to-end and caches artifacts to disk.
The FastAPI serving layer loads from cache — no retraining on startup.

Stages:
  1. Data preparation + entity field synthesis
  2. Velocity feature computation (offline replay)
  3. Entity graph construction + GraphSAGE training
  4. Fraud ring detection (Stripe similarity approach)
  5. FP-Growth rule mining (Uber RADAR approach)
  6. XGBoost training + ONNX export + evaluation

Run: python src/pipeline.py
     python src/pipeline.py --subsample 0.2   (for quick dev run)
     python src/pipeline.py --force            (force retrain everything)
"""

import argparse
import os
import sys
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")


def stage1_data(subsample: float, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")

    if not force and os.path.exists(train_path) and os.path.exists(test_path):
        print("[pipeline] Stage 1: Loading cached data...")
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    print("\n[pipeline] Stage 1: Data Preparation")
    from data_prep import run_data_pipeline

    return run_data_pipeline(subsample=subsample)


def stage2_velocity(
    train: pd.DataFrame, test: pd.DataFrame, force: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_vel_path = os.path.join(PROCESSED_DIR, "train_velocity.parquet")
    test_vel_path = os.path.join(PROCESSED_DIR, "test_velocity.parquet")

    if not force and os.path.exists(train_vel_path):
        print("[pipeline] Stage 2: Loading cached velocity features...")
        return pd.read_parquet(train_vel_path), pd.read_parquet(test_vel_path)

    print("\n[pipeline] Stage 2: Velocity Features")
    from velocity.feature_store import compute_velocity_features_offline

    train_vel = compute_velocity_features_offline(train)
    # For test set, use the same offline computation (no data leakage:
    # test transactions are after train, so window lookbacks into test don't
    # see train data — we compute them independently per entity)
    test_vel = compute_velocity_features_offline(test)

    train_vel.to_parquet(train_vel_path, index=False)
    test_vel.to_parquet(test_vel_path, index=False)
    return train_vel, test_vel


def stage3_graph(
    train: pd.DataFrame, test: pd.DataFrame, force: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_graph_path = os.path.join(PROCESSED_DIR, "train_graph.parquet")
    test_graph_path = os.path.join(PROCESSED_DIR, "test_graph.parquet")

    if not force and os.path.exists(train_graph_path):
        print("[pipeline] Stage 3: Loading cached graph features + embeddings...")
        return pd.read_parquet(train_graph_path), pd.read_parquet(test_graph_path)

    print("\n[pipeline] Stage 3: Entity Graph + GraphSAGE")

    from graph.entity_graph import (
        attach_gnn_embeddings,
        build_entity_graph,
        compute_graph_features,
        train_graphsage,
    )

    G = build_entity_graph(train)
    # Save graph for visualization in React dashboard
    graph_data = {
        "nodes": [
            {
                "id": n,
                "type": G.nodes[n].get("node_type", "unknown"),
                "fraud_rate": G.nodes[n].get("fraud_rate", 0),
                "is_fraud": G.nodes[n].get("is_fraud_node", 0),
                "txn_count": G.nodes[n].get("txn_count", 0),
            }
            for n in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, "type": G.edges[u, v].get("edge_type", "")}
            for u, v in G.edges()
        ],
    }
    import json

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "entity_graph.json"), "w") as f:
        json.dump(graph_data, f)

    train = compute_graph_features(G, train)
    test = compute_graph_features(G, test)

    # Train GraphSAGE on training graph only (inductive — test nodes unseen)
    card_embeddings = train_graphsage(G, epochs=50)

    train = attach_gnn_embeddings(train, card_embeddings)
    test = attach_gnn_embeddings(test, card_embeddings)

    train.to_parquet(train_graph_path, index=False)
    test.to_parquet(test_graph_path, index=False)
    return train, test


def stage4_fraud_rings(train: pd.DataFrame, force: bool) -> pd.DataFrame:
    rings_path = os.path.join(PROCESSED_DIR, "train_rings.parquet")
    model_path = os.path.join(MODELS_DIR, "similarity_model.pkl")

    if not force and os.path.exists(rings_path) and os.path.exists(model_path):
        print("[pipeline] Stage 4: Loading cached fraud rings...")
        return pd.read_parquet(rings_path)

    print("\n[pipeline] Stage 4: Fraud Ring Detection (Stripe Similarity Approach)")
    from graph.fraud_rings import (
        detect_fraud_rings,
        generate_training_pairs,
        get_ring_stats,
        save_similarity_model,
        train_similarity_model,
    )

    # fraud_rings.py expects 'card_id' column — our data uses 'cc_num'
    train_rings_input = train.copy()
    train_rings_input["card_id"] = train_rings_input["cc_num"].astype(str)

    pairs_df = generate_training_pairs(train_rings_input)
    sim_model = train_similarity_model(pairs_df)
    save_similarity_model(sim_model, model_path)

    train_rings = detect_fraud_rings(train_rings_input, sim_model)
    # Propagate ring_id back to original train index
    train["ring_id"] = (
        train_rings["ring_id"].values if "ring_id" in train_rings.columns else "NO_RING"
    )

    ring_stats = get_ring_stats(train_rings if "ring_id" in train_rings.columns else train)
    if not ring_stats.empty:
        ring_stats.to_json(os.path.join(MODELS_DIR, "ring_stats.json"), orient="records", indent=2)
        print(f"[pipeline] Found {len(ring_stats)} fraud rings")
        print(ring_stats.head(5).to_string())

    train.to_parquet(rings_path, index=False)
    return train


def stage5_rules(train: pd.DataFrame, force: bool) -> None:
    rules_path = os.path.join(MODELS_DIR, "fraud_rules.json")

    if not force and os.path.exists(rules_path):
        print("[pipeline] Stage 5: Loading cached FP-Growth rules...")
        return

    print("\n[pipeline] Stage 5: FP-Growth Auto-Rule Mining (Uber RADAR Approach)")
    from rules.fp_growth_rules import (
        discretize_for_fpgrowth,
        format_rule_for_display,
        mine_fraud_rules,
        save_rules_to_json,
    )

    fraud_df = train[train["is_fraud"] == 1]
    disc_df = discretize_for_fpgrowth(train, fraud_df)
    rules_df = mine_fraud_rules(disc_df, min_support=0.02, min_confidence=0.5, min_lift=1.5)

    if not rules_df.empty:
        save_rules_to_json(rules_df, rules_path)
        print(f"[pipeline] Mined {len(rules_df)} fraud rules. Top 5:")
        for _, row in rules_df.head(5).iterrows():
            print(f"  {format_rule_for_display(row)}")
    else:
        print("[pipeline] No rules mined (try lowering min_support/min_confidence)")


def stage6_model(train: pd.DataFrame, test: pd.DataFrame, force: bool) -> dict:
    model_path = os.path.join(MODELS_DIR, "fraud_model.pkl")

    if not force and os.path.exists(model_path):
        print("[pipeline] Stage 6: Loading cached model...")
        model = joblib.load(model_path)
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        return {"model": model, "feature_cols": feature_cols}

    print("\n[pipeline] Stage 6: XGBoost Training + ONNX Export")
    from model.train import train_fraud_model

    return train_fraud_model(train, test)


def run_pipeline(subsample: float = 1.0, force: bool = False) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("FRAUD DETECTION SYSTEM — FULL PIPELINE")
    print("=" * 65)

    # Stage 1: Data
    train, test = stage1_data(subsample, force)

    # Stage 2: Velocity features
    train, test = stage2_velocity(train, test, force)

    # Stage 3: Graph + GNN
    train, test = stage3_graph(train, test, force)

    # Stage 4: Fraud rings (train only — rings are mined from labeled data)
    train = stage4_fraud_rings(train, force)

    # Merge ring labels into test (for evaluation only — not a model feature)
    if "ring_id" in train.columns:
        ring_map = train.set_index("cc_num")["ring_id"].to_dict()
        test["ring_id"] = test["cc_num"].map(ring_map).fillna("NO_RING")

    # Stage 5: FP-Growth rules
    stage5_rules(train, force)

    # Save combined feature sets for the model
    train_features_path = os.path.join(PROCESSED_DIR, "train_features.parquet")
    test_features_path = os.path.join(PROCESSED_DIR, "test_features.parquet")
    if force or not os.path.exists(train_features_path):
        train.to_parquet(train_features_path, index=False)
        test.to_parquet(test_features_path, index=False)

    # Stage 6: Model
    results = stage6_model(train, test, force)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    if "metrics" in results:
        m = results["metrics"]
        print(f"  AUC-ROC:             {m.get('auc_roc', 0):.4f}")
        print(f"  AUC-PR:              {m.get('auc_pr', 0):.4f}")
        print(f"  Precision@1%:        {m.get('precision_at_1pct', 0):.4f}")
        print(f"  Recall@0.1%FPR:      {m.get('recall_at_01fpr', 0):.4f}")
        print(f"  Dollar capture rate: {m.get('dollar_capture_rate', 0):.4f}")
    print("\nStart the API server: uvicorn src.api.main:app --reload")
    print("Start the React dashboard: cd frontend && npm run dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection System Pipeline")
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.2,
        help="Fraction of data to use (default: 0.2 for dev)",
    )
    parser.add_argument("--force", action="store_true", help="Force retrain all stages")
    args = parser.parse_args()
    run_pipeline(subsample=args.subsample, force=args.force)
