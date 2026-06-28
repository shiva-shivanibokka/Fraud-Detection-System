"""
TGAT-style temporal GNN on Elliptic
===================================
Adds TGAT's signature idea to the GAT baseline: a learnable *functional time
encoding* of each edge's time-gap (Delta t = step[dst] - step[src]), fed into
graph attention so the attention weights become time-aware. Same data, split,
loss, and metrics as train_gat.py — a clean A/B (does temporal modeling beat the
static GAT?).

Run:  python -m src.graph_fraud.train_tgat
"""

import os

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.nn import GATConv

ROOT = os.path.join("data", "elliptic")
RAW_FEATURES = os.path.join(ROOT, "raw", "elliptic_txs_features.csv")
EPOCHS = 200
HIDDEN = 64
HEADS = 4
TIME_DIM = 16
LR = 0.005
SEED = 42


class TimeEncoder(torch.nn.Module):
    """TGAT functional time encoding: cos of a learnable-frequency projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.w = torch.nn.Linear(1, dim)
        # init with geometrically-spaced frequencies (as in the TGAT paper)
        self.w.weight = torch.nn.Parameter(
            (1.0 / 10 ** torch.linspace(0, 9, dim)).reshape(dim, 1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.w(dt.view(-1, 1).float()))


class TGAT(torch.nn.Module):
    def __init__(self, in_ch: int, hidden: int, heads: int, time_dim: int, dropout: float = 0.5):
        super().__init__()
        self.time_enc = TimeEncoder(time_dim)
        self.g1 = GATConv(in_ch, hidden, heads=heads, edge_dim=time_dim, dropout=dropout)
        self.g2 = GATConv(hidden * heads, 2, heads=1, concat=False, edge_dim=time_dim,
                          dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, dt):
        te = self.time_enc(dt)
        x = F.elu(self.g1(x, edge_index, edge_attr=te))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.g2(x, edge_index, edge_attr=te)


def evaluate(logits, y, mask) -> dict:
    prob = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
    pred = logits[mask].argmax(1).cpu().numpy()
    true = y[mask].cpu().numpy()
    return {
        "illicit_f1": f1_score(true, pred, pos_label=1, zero_division=0),
        "illicit_precision": precision_score(true, pred, pos_label=1, zero_division=0),
        "illicit_recall": recall_score(true, pred, pos_label=1, zero_division=0),
        "auc": roc_auc_score(true, prob),
    }


def main() -> dict:
    from torch_geometric.datasets import EllipticBitcoinDataset

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tgat] device: {device}")

    data = EllipticBitcoinDataset(root=ROOT)[0]
    # per-node time-step (raw features col 1, aligned to PyG node order)
    node_time = torch.tensor(
        pd.read_csv(RAW_FEATURES, header=None, usecols=[1])[1].to_numpy(), dtype=torch.float
    )
    data = data.to(device)
    node_time = node_time.to(device)
    y = data.y
    dt = node_time[data.edge_index[1]] - node_time[data.edge_index[0]]

    ytr = y[data.train_mask]
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    print(f"[tgat] train illicit={n_pos}, licit={n_neg}, pos_weight={weight[1]:.1f}")

    model = TGAT(data.num_node_features, HIDDEN, HEADS, TIME_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index, dt)
        loss = F.cross_entropy(out[data.train_mask], y[data.train_mask], weight=weight)
        loss.backward()
        opt.step()
        if (epoch + 1) % 40 == 0:
            model.eval()
            with torch.no_grad():
                m = evaluate(model(data.x, data.edge_index, dt), y, data.test_mask)
            print(f"  epoch {epoch + 1:>3} | loss {loss.item():.4f} | "
                  f"test illicit-F1 {m['illicit_f1']:.4f} | AUC {m['auc']:.4f}")

    model.eval()
    with torch.no_grad():
        metrics = evaluate(model(data.x, data.edge_index, dt), y, data.test_mask)
    print("\n[tgat] ===== TGAT (test: steps 35-49) =====")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.4f}")
    print("\n[tgat] GAT baseline was: illicit_f1 0.3362, auc 0.8589")
    return metrics


if __name__ == "__main__":
    main()
