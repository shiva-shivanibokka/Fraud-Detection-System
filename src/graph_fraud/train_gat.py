"""
GAT baseline on Elliptic (static GNN — no temporal modeling)
============================================================
Node classification: illicit (1) vs licit (0). Transductive full-graph message
passing; train on time-steps 1-29, early-stop on validation (steps 30-34), report
test (steps 35-49) at the best-validation epoch. The bar for the temporal model.

Run:  python -m src.graph_fraud.train_gat
"""

import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.graph_fraud.common import eval_logits, load_elliptic

EPOCHS = 200
HIDDEN = 64
HEADS = 4
LR = 0.005
SEED = 42


class GAT(torch.nn.Module):
    def __init__(self, in_ch: int, hidden: int, heads: int, dropout: float = 0.5):
        super().__init__()
        self.g1 = GATConv(in_ch, hidden, heads=heads, dropout=dropout)
        self.g2 = GATConv(hidden * heads, 2, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.g1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.g2(x, edge_index)


def main() -> dict:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gat] device: {device}")

    data, _ = load_elliptic(device)
    y = data.y

    ytr = y[data.train_mask]
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    weight = torch.tensor([1.0, math.sqrt(n_neg / max(n_pos, 1))], device=device)
    print(f"[gat] train illicit={n_pos}, licit={n_neg}, pos_weight={weight[1]:.1f}")

    model = GAT(data.num_node_features, HIDDEN, HEADS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    best_val_auc, best_test = 0.0, {}
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], y[data.train_mask], weight=weight)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
            val = eval_logits(logits, y, data.val_mask)
            if val["auc"] > best_val_auc:
                best_val_auc = val["auc"]
                best_test = eval_logits(logits, y, data.test_mask)

    print("\n[gat] ===== GAT (test steps 35-49, best-validation epoch) =====")
    for k, v in best_test.items():
        print(f"  {k:18s}: {v:.4f}")
    return best_test


if __name__ == "__main__":
    main()
