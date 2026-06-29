"""
EvolveGCN-O on Elliptic — snapshot-based temporal GNN
=====================================================
The temporal model that fits Elliptic: it evolves the GCN *weight matrices*
across the 49 time-step snapshots via GRU cells, capturing how fraud patterns
change over time (e.g. the dark-market regime shift) — rather than attending
over edge time-gaps (TGAT, the wrong bias here).

Each snapshot's subgraph = the step-t nodes PLUS their neighbors from time <= t
(past money-flow context), since Elliptic's edges are mostly cross-time-step.
Train on steps 1-29, early-stop on validation (30-34), report test (35-49) at the
best-validation epoch. Implemented from scratch (torch-geometric-temporal needs
torch-sparse, which won't build on Windows) with native torch sparse ops.

Run:  python -m src.graph_fraud.train_evolvegcn
"""

import math

import torch
import torch.nn.functional as F

from src.graph_fraud.common import illicit_metrics, load_elliptic

EPOCHS = 200
HIDDEN = 64
LR = 0.005
SEED = 42


def _norm_adj(ei_local, n, device):
    sl = torch.arange(n, device=device)
    ei = torch.cat([ei_local, ei_local.flip(0), torch.stack([sl, sl])], dim=1)
    deg = torch.zeros(n, device=device).scatter_add_(
        0, ei[0], torch.ones(ei.size(1), device=device)
    )
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    vals = dinv[ei[0]] * dinv[ei[1]]
    return torch.sparse_coo_tensor(ei, vals, (n, n)).coalesce()


def build_snapshots(data, node_time, device) -> list:
    n_total = data.num_nodes
    ei = data.edge_index
    snaps = []
    for t in range(int(node_time.min()), int(node_time.max()) + 1):
        center = node_time == t
        if int(center.sum()) == 0:
            continue
        inc = (center[ei[0]] & (node_time[ei[1]] <= t)) | (
            center[ei[1]] & (node_time[ei[0]] <= t)
        )
        e = ei[:, inc]
        involved = torch.cat([center.nonzero().flatten(), e.reshape(-1)]).unique()
        n = involved.numel()
        gmap = torch.full((n_total,), -1, dtype=torch.long, device=device)
        gmap[involved] = torch.arange(n, device=device)
        is_center = center[involved]
        snaps.append({
            "x": data.x[involved],
            "y": data.y[involved],
            "adj": _norm_adj(gmap[e], n, device),
            "train": data.train_mask[involved] & is_center,
            "val": data.val_mask[involved] & is_center,
            "test": data.test_mask[involved] & is_center,
            # Global node indices + center mask, so the prediction exporter can
            # map each snapshot's center-node logits back to global node ids.
            "involved": involved,
            "is_center": is_center,
        })
    return snaps


class EvolveGCNO(torch.nn.Module):
    def __init__(self, in_dim: int, hid: int, n_classes: int):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.empty(in_dim, hid))
        self.W2 = torch.nn.Parameter(torch.empty(hid, n_classes))
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)
        self.gru1 = torch.nn.GRUCell(in_dim, in_dim)
        self.gru2 = torch.nn.GRUCell(hid, hid)

    def forward(self, snaps: list) -> list:
        w1, w2 = self.W1, self.W2
        outs = []
        for s in snaps:
            w1 = self.gru1(w1.t(), w1.t()).t()
            w2 = self.gru2(w2.t(), w2.t()).t()
            h = F.relu(torch.sparse.mm(s["adj"], s["x"] @ w1))
            outs.append(torch.sparse.mm(s["adj"], h @ w2))
        return outs


def _eval(outs, snaps, key) -> dict:
    logits = torch.cat([o[s[key]] for o, s in zip(outs, snaps) if s[key].any()])
    y = torch.cat([s["y"][s[key]] for s in snaps if s[key].any()])
    prob = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()
    return illicit_metrics(prob, pred, y.cpu().numpy())


def main() -> dict:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evolvegcn] device: {device}")

    data, node_time = load_elliptic(device)
    snaps = build_snapshots(data, node_time, device)
    print(f"[evolvegcn] {len(snaps)} snapshots built")

    ytr = torch.cat([s["y"][s["train"]] for s in snaps if s["train"].any()])
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    weight = torch.tensor([1.0, math.sqrt(n_neg / max(n_pos, 1))], device=device)
    print(f"[evolvegcn] train illicit={n_pos}, licit={n_neg}, pos_weight={weight[1]:.1f}")

    model = EvolveGCNO(data.num_node_features, HIDDEN, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    best_val_auc, best_test = 0.0, {}
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        outs = model(snaps)
        logits = torch.cat([o[s["train"]] for o, s in zip(outs, snaps) if s["train"].any()])
        labels = torch.cat([s["y"][s["train"]] for s in snaps if s["train"].any()])
        loss = F.cross_entropy(logits, labels, weight=weight)
        loss.backward()
        opt.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                outs = model(snaps)
            val = _eval(outs, snaps, "val")
            if val["auc"] > best_val_auc:
                best_val_auc = val["auc"]
                best_test = _eval(outs, snaps, "test")

    print("\n[evolvegcn] ===== EvolveGCN-O (test steps 35-49, best-validation epoch) =====")
    for k, v in best_test.items():
        print(f"  {k:18s}: {v:.4f}")
    return best_test


if __name__ == "__main__":
    main()
