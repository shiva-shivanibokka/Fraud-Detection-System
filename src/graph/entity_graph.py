"""
Entity Graph Construction + GraphSAGE Fraud Embeddings
=======================================================
Builds a heterogeneous entity graph from transaction data:
  - Nodes: cards, devices, IP prefixes, merchants
  - Edges: "card used device", "card used IP", "card transacted at merchant"

Then trains a GraphSAGE model (PyTorch Geometric) to produce 64-dim
fraud risk embeddings per card node. These embeddings are stored as
features alongside tabular and velocity features in the main XGBoost model.

Architecture mirrors Uber's RGCN fraud paper and Airbnb's payment network GNN:
  - Semi-supervised: card nodes labeled by fraud rate
  - Inductive: GraphSAGE generates embeddings for unseen nodes
    (new cards have no history, but share a device/IP with known fraudsters)
  - Embeddings stored in a dict (simulating Redis embedding lookup at serve time)

Why GraphSAGE over standard GCN?
  GCN requires the full graph at inference time (transductive).
  GraphSAGE generates embeddings by sampling a fixed-size neighborhood
  — works for new nodes that appear after training.
  This is the inductive learning requirement for production fraud detection.
"""

import os
import warnings

import joblib
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

try:
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("[graph] PyTorch Geometric not available — using NetworkX graph features only")


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_entity_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build a bipartite entity graph from transaction records.

    Node types:
      card_{cc_num}, device_{device_id}, ip_{ip_prefix}, merchant_{merchant}

    Edge types (undirected):
      card — device  (card used this device)
      card — ip      (card originated from this IP prefix)
      card — merchant (card transacted at this merchant)

    Node attributes:
      - is_fraud_node: 1 if this card has any confirmed fraud transaction
      - fraud_rate: fraction of transactions that are fraud (cards only)
      - txn_count: number of transactions
      - avg_amount: average transaction amount
    """
    G = nx.Graph()

    # --- Card nodes ---
    card_stats = (
        df.groupby("cc_num")
        .agg(
            txn_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            avg_amount=("amt", "mean"),
            max_amount=("amt", "max"),
        )
        .reset_index()
    )
    card_stats["fraud_rate"] = card_stats["fraud_count"] / card_stats["txn_count"]
    card_stats["is_fraud_node"] = (card_stats["fraud_count"] > 0).astype(int)

    for _, row in card_stats.iterrows():
        # int() guards against pandas upcasting the big-integer cc_num to a
        # float in mixed-dtype iterrows (which would name the node
        # "card_<n>.0" and break edge matching + embedding lookup).
        G.add_node(
            f"card_{int(row.cc_num)}",
            node_type="card",
            is_fraud_node=int(row.is_fraud_node),
            fraud_rate=float(row.fraud_rate),
            txn_count=int(row.txn_count),
            avg_amount=float(row.avg_amount),
            max_amount=float(row.max_amount),
        )

    # --- Device nodes ---
    device_stats = (
        df.groupby("device_id")
        .agg(
            txn_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            n_cards=("cc_num", "nunique"),
        )
        .reset_index()
    )
    device_stats["fraud_rate"] = device_stats["fraud_count"] / device_stats["txn_count"]

    for _, row in device_stats.iterrows():
        G.add_node(
            f"device_{row.device_id}",
            node_type="device",
            is_fraud_node=int(row.fraud_count > 0),
            fraud_rate=float(row.fraud_rate),
            txn_count=int(row.txn_count),
            n_cards=int(row.n_cards),
            avg_amount=0.0,
            max_amount=0.0,
        )

    # --- IP prefix nodes ---
    ip_stats = (
        df.groupby("ip_prefix")
        .agg(
            txn_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            n_cards=("cc_num", "nunique"),
        )
        .reset_index()
    )
    ip_stats["fraud_rate"] = ip_stats["fraud_count"] / ip_stats["txn_count"]

    for _, row in ip_stats.iterrows():
        G.add_node(
            f"ip_{row.ip_prefix}",
            node_type="ip",
            is_fraud_node=int(row.fraud_count > 0),
            fraud_rate=float(row.fraud_rate),
            txn_count=int(row.txn_count),
            n_cards=int(row.n_cards),
            avg_amount=0.0,
            max_amount=0.0,
        )

    # --- Merchant nodes ---
    merch_stats = (
        df.groupby("merchant")
        .agg(
            txn_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            n_cards=("cc_num", "nunique"),
            avg_amount=("amt", "mean"),
        )
        .reset_index()
    )
    merch_stats["fraud_rate"] = merch_stats["fraud_count"] / merch_stats["txn_count"]

    for _, row in merch_stats.iterrows():
        G.add_node(
            f"merchant_{row.merchant}",
            node_type="merchant",
            is_fraud_node=int(row.fraud_count > 0),
            fraud_rate=float(row.fraud_rate),
            txn_count=int(row.txn_count),
            n_cards=int(row.n_cards),
            avg_amount=float(row.avg_amount),
            max_amount=0.0,
        )

    # --- Edges ---
    # card — device (deduplicated)
    card_device = df[["cc_num", "device_id"]].drop_duplicates()
    for _, row in card_device.iterrows():
        cn = f"card_{row.cc_num}"
        dn = f"device_{row.device_id}"
        if G.has_node(cn) and G.has_node(dn):
            G.add_edge(cn, dn, edge_type="card_device")

    # card — IP prefix
    card_ip = df[["cc_num", "ip_prefix"]].drop_duplicates()
    for _, row in card_ip.iterrows():
        cn = f"card_{row.cc_num}"
        ip = f"ip_{row.ip_prefix}"
        if G.has_node(cn) and G.has_node(ip):
            G.add_edge(cn, ip, edge_type="card_ip")

    # card — merchant (top-5 merchants per card to avoid dense graph)
    top_merchants = (
        df.groupby(["cc_num", "merchant"])
        .size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .groupby("cc_num")
        .head(5)
    )
    for _, row in top_merchants.iterrows():
        cn = f"card_{row.cc_num}"
        mn = f"merchant_{row.merchant}"
        if G.has_node(cn) and G.has_node(mn):
            G.add_edge(cn, mn, edge_type="card_merchant")

    print(f"[graph] Entity graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def compute_graph_features(G: nx.Graph, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-card graph features from the entity graph.
    These are the features passed to XGBoost alongside tabular/velocity features.

    Features (inspired by Uber RGCN and Stripe similarity paper):
      - degree: number of connected entities
      - n_devices: unique devices used
      - n_ips: unique IP prefixes
      - n_merchants: unique merchants
      - device_fraud_rate: avg fraud rate of connected devices
      - ip_fraud_rate: avg fraud rate of connected IP prefixes
      - shared_device_cards: total cards sharing same device (fraud ring signal)
      - shared_ip_cards: total cards sharing same IP prefix
      - neighbor_fraud_rate: fraction of neighboring cards that are fraudulent
      - max_component_size: size of connected component (fraud ring size)
    """
    print("[graph] Computing graph features per card...")

    # Connected components — fraud rings cluster into large components
    components = list(nx.connected_components(G))
    node_to_component = {}
    for comp in components:
        size = len(comp)
        for node in comp:
            node_to_component[node] = size

    features = []
    cards = df["cc_num"].unique()

    for card in cards:
        node = f"card_{card}"
        if not G.has_node(node):
            features.append({"cc_num": card})
            continue

        neighbors = list(G.neighbors(node))
        devices = [n for n in neighbors if n.startswith("device_")]
        ips = [n for n in neighbors if n.startswith("ip_")]
        merchants = [n for n in neighbors if n.startswith("merchant_")]

        # Fraud rates of neighboring entity nodes
        dev_fraud_rates = [G.nodes[d].get("fraud_rate", 0) for d in devices]
        ip_fraud_rates = [G.nodes[i].get("fraud_rate", 0) for i in ips]

        # Cards sharing same device/IP (fraud ring signal)
        shared_device_cards = sum(G.nodes[d].get("n_cards", 1) - 1 for d in devices)
        shared_ip_cards = sum(G.nodes[i].get("n_cards", 1) - 1 for i in ips)

        # Neighbor cards' fraud rates
        card_neighbors = []
        for dev in devices:
            card_neighbors.extend(
                [n for n in G.neighbors(dev) if n.startswith("card_") and n != node]
            )
        for ip in ips:
            card_neighbors.extend(
                [n for n in G.neighbors(ip) if n.startswith("card_") and n != node]
            )
        card_neighbors = list(set(card_neighbors))
        neighbor_fraud_rates = [G.nodes[n].get("fraud_rate", 0) for n in card_neighbors]

        feat = {
            "cc_num": card,
            "graph_degree": G.degree(node),
            "graph_n_devices": len(devices),
            "graph_n_ips": len(ips),
            "graph_n_merchants": len(merchants),
            "graph_device_fraud_rate": np.mean(dev_fraud_rates) if dev_fraud_rates else 0.0,
            "graph_ip_fraud_rate": np.mean(ip_fraud_rates) if ip_fraud_rates else 0.0,
            "graph_shared_device_cards": shared_device_cards,
            "graph_shared_ip_cards": shared_ip_cards,
            "graph_neighbor_card_count": len(card_neighbors),
            "graph_neighbor_fraud_rate": np.mean(neighbor_fraud_rates)
            if neighbor_fraud_rates
            else 0.0,
            "graph_component_size": node_to_component.get(node, 1),
        }
        features.append(feat)

    graph_df = pd.DataFrame(features)
    # Merge back to transactions
    df_out = df.merge(graph_df, on="cc_num", how="left")
    graph_cols = [c for c in graph_df.columns if c != "cc_num"]
    df_out[graph_cols] = df_out[graph_cols].fillna(0)

    print(f"[graph] Added {len(graph_cols)} graph features")
    return df_out


# ---------------------------------------------------------------------------
# GraphSAGE Embedding Model
# ---------------------------------------------------------------------------


class GraphSAGEFraud(nn.Module):
    """
    Two-layer GraphSAGE for fraud node classification.
    Produces 64-dim embeddings per card node.

    Architecture: SAGEConv(in → 128) → ReLU → Dropout → SAGEConv(128 → 64) → output
    Uses mean aggregation (default SAGE) — proven effective for fraud graphs
    where super-nodes (high-degree entities) would distort sum aggregation.
    """

    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden, aggr="mean")
        self.conv2 = SAGEConv(hidden, out_channels, aggr="mean")
        self.classifier = nn.Linear(out_channels, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        embeddings = x
        logits = self.classifier(x).squeeze(-1)
        return logits, embeddings


def train_graphsage(
    G: nx.Graph,
    epochs: int = 50,
    lr: float = 0.01,
    hidden: int = 128,
    embed_dim: int = 64,
    device_str: str = "cpu",
) -> dict:
    """
    Train GraphSAGE on the entity graph.
    Returns a dict: card_id -> 64-dim embedding numpy array.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("[graph] PyTorch Geometric not available — skipping GNN training")
        return {}

    import torch

    device = torch.device(device_str)

    # Build node feature matrix from graph attributes
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Feature vector per node: [fraud_rate, txn_count_log, avg_amount_log,
    #                            is_fraud_node, n_cards_log, degree_norm]
    feats = np.zeros((n_nodes, 6), dtype=np.float32)
    labels = np.full(n_nodes, -1, dtype=np.float32)  # -1 = unlabeled (non-card)
    card_mask = np.zeros(n_nodes, dtype=bool)

    for node, i in node_idx.items():
        attrs = G.nodes[node]
        feats[i, 0] = attrs.get("fraud_rate", 0.0)
        feats[i, 1] = np.log1p(attrs.get("txn_count", 0))
        feats[i, 2] = np.log1p(attrs.get("avg_amount", 0))
        feats[i, 3] = float(attrs.get("is_fraud_node", 0))
        feats[i, 4] = np.log1p(attrs.get("n_cards", 1))
        feats[i, 5] = np.log1p(G.degree(node))

        if node.startswith("card_"):
            labels[i] = float(attrs.get("is_fraud_node", 0))
            card_mask[i] = True

    # Build edge index
    edges = list(G.edges())
    if not edges:
        print("[graph] No edges in graph — skipping GNN")
        return {}

    src = [node_idx[e[0]] for e in edges]
    dst = [node_idx[e[1]] for e in edges]
    # Undirected: add both directions
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long).to(device)

    x = torch.tensor(feats, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)
    card_mask_t = torch.tensor(card_mask, dtype=torch.bool).to(device)

    model = GraphSAGEFraud(in_channels=6, hidden=hidden, out_channels=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Class weight for imbalanced fraud labels
    fraud_count = card_mask.sum() and (labels[card_mask] == 1).sum()
    legit_count = card_mask.sum() and (labels[card_mask] == 0).sum()
    pos_weight = torch.tensor([legit_count / max(fraud_count, 1)], dtype=torch.float32).to(device)

    print(f"[graph] Training GraphSAGE: {n_nodes} nodes, {len(edges)} edges, {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits, _ = model(x, edge_index)

        # Only compute loss on card nodes with known labels
        labeled_mask = card_mask_t & (y >= 0)
        loss = F.binary_cross_entropy_with_logits(
            logits[labeled_mask],
            y[labeled_mask],
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = torch.sigmoid(logits[labeled_mask]) > 0.5
                acc = (preds == y[labeled_mask].bool()).float().mean()
            print(
                f"  Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | "
                f"Card Acc: {acc.item():.4f}"
            )

    # Extract embeddings for all card nodes
    model.eval()
    with torch.no_grad():
        _, embeddings = model(x, edge_index)
        embeddings_np = embeddings.cpu().numpy()

    card_embeddings = {}
    for node, i in node_idx.items():
        if node.startswith("card_"):
            card_id = node[len("card_") :]
            card_embeddings[card_id] = embeddings_np[i]

    print(f"[graph] Generated embeddings for {len(card_embeddings)} cards (shape: {embed_dim}d)")

    # Save model + embeddings
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "graphsage.pt"))
    joblib.dump(card_embeddings, os.path.join(MODELS_DIR, "card_embeddings.pkl"))
    joblib.dump(node_idx, os.path.join(MODELS_DIR, "graph_node_idx.pkl"))

    return card_embeddings


def attach_gnn_embeddings(
    df: pd.DataFrame,
    card_embeddings: dict,
    embed_dim: int = 64,
) -> pd.DataFrame:
    """
    Attach GNN embeddings to each transaction row.
    Each card gets a 64-dim vector; embed_0 through embed_63 become features.
    """
    if not card_embeddings:
        # No embeddings available — add zero columns as placeholders
        for i in range(embed_dim):
            df[f"gnn_embed_{i}"] = 0.0
        return df

    zero_embed = np.zeros(embed_dim)
    embed_matrix = np.array([card_embeddings.get(str(cc), zero_embed) for cc in df["cc_num"]])

    for i in range(embed_dim):
        df[f"gnn_embed_{i}"] = embed_matrix[:, i]

    print(f"[graph] Attached {embed_dim}-dim GNN embeddings to {len(df):,} rows")
    return df


def get_graph_feature_names(embed_dim: int = 64) -> list[str]:
    """Return all graph feature names (NetworkX + GNN embedding columns)."""
    nx_feats = [
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
    gnn_feats = [f"gnn_embed_{i}" for i in range(embed_dim)]
    return nx_feats + gnn_feats


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data_prep import run_data_pipeline

    print("Running data pipeline (30% sample)...")
    train, test = run_data_pipeline(subsample=0.1)

    print("\nBuilding entity graph...")
    G = build_entity_graph(train)

    print("\nComputing graph features...")
    train = compute_graph_features(G, train)

    print("\nTraining GraphSAGE...")
    embeddings = train_graphsage(G, epochs=30)

    if embeddings:
        train = attach_gnn_embeddings(train, embeddings)
        print("\nSample graph features:")
        graph_cols = [c for c in train.columns if c.startswith("graph_")][:5]
        print(train[graph_cols + ["is_fraud"]].head())
