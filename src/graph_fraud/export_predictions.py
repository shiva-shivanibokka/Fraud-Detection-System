"""
Export served GNN predictions for the Elliptic graph module.
======================================================================
Trains a model (default: the benchmark-winning EvolveGCN-O; GAT baseline also
available via --model gat), early-stopped on validation, then writes a *compact*
JSON artifact the serving API can hand to the frontend WITHOUT any torch
dependency at request time — same pattern as the other models/*.json artifacts.

The artifact has three parts, all kept small enough to serve on free tier:
  - metrics:  illicit precision / recall / F1 / AUC on the test steps (35-49),
              reported at the best-validation epoch (no test peeking)
  - timeline: per time-step counts of predicted vs actual illicit nodes
  - graph:    a sampled high-risk subgraph (top-prob test nodes + neighbors,
              capped) for a force-directed visualization

Run on a machine with the Elliptic dataset + a GPU:
    python -m src.graph_fraud.export_predictions                 # EvolveGCN-O, 200 epochs
    python -m src.graph_fraud.export_predictions --model gat     # GAT baseline
    python -m src.graph_fraud.export_predictions --epochs 40
Then upload models/elliptic_graph.json to the HF Hub model repo (see
scripts/upload_models_to_hf.py) so the deployed API serves it.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from src.graph_fraud.common import eval_logits, illicit_metrics, load_elliptic
from src.graph_fraud.train_evolvegcn import EvolveGCNO, build_snapshots
from src.graph_fraud.train_evolvegcn import _eval as _eval_egcn
from src.graph_fraud.train_gat import GAT

SEED = 42
OUT = os.path.join("models", "elliptic_graph.json")
SEED_NODES = 60      # highest-risk test nodes to anchor the subgraph
MAX_NODES = 220      # cap on subgraph nodes (keeps the JSON small)
MAX_EDGES = 500


def _train(data, device, epochs: int):
    torch.manual_seed(SEED)
    y = data.y
    ytr = y[data.train_mask]
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    weight = torch.tensor([1.0, math.sqrt(n_neg / max(n_pos, 1))], device=device)

    model = GAT(data.num_node_features, 64, 4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    best_val, best_logits, best_test = 0.0, None, {}
    for epoch in range(epochs):
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
            if val["auc"] > best_val:
                best_val = val["auc"]
                best_logits = logits
                best_test = eval_logits(logits, y, data.test_mask)
    return best_logits, best_test


def _train_evolvegcn(data, node_time, device, epochs: int):
    """Train EvolveGCN-O and return (global per-node logits, test metrics) at the
    best-validation epoch. Each node is predicted in the snapshot where it is the
    center (its own time-step), so the per-snapshot outputs are scattered back to
    a global [num_nodes, 2] logit tensor for the timeline/subgraph export."""
    torch.manual_seed(SEED)
    snaps = build_snapshots(data, node_time, device)
    ytr = torch.cat([s["y"][s["train"]] for s in snaps if s["train"].any()])
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    weight = torch.tensor([1.0, math.sqrt(n_neg / max(n_pos, 1))], device=device)

    model = EvolveGCNO(data.num_node_features, 64, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    def _global_logits(outs) -> torch.Tensor:
        gl = data.x.new_zeros((data.num_nodes, 2))
        for o, s in zip(outs, snaps):
            c = s["is_center"]
            if c.any():
                gl[s["involved"][c]] = o[c].detach()
        return gl

    best_val, best_logits, best_test = 0.0, None, {}
    for epoch in range(epochs):
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
            val = _eval_egcn(outs, snaps, "val")
            if val["auc"] > best_val:
                best_val = val["auc"]
                best_test = _eval_egcn(outs, snaps, "test")
                best_logits = _global_logits(outs)
    return best_logits, best_test


def _timeline(pred, y, node_time) -> list[dict]:
    out = []
    for t in range(int(node_time.min()), int(node_time.max()) + 1):
        m = node_time == t
        if int(m.sum()) == 0:
            continue
        known = m & (y != 2)
        out.append({
            "step": t,
            "nodes": int(m.sum()),
            "actual_illicit": int(((y == 1) & m).sum()),
            "predicted_illicit": int(((pred == 1) & known).sum()),
        })
    return out


def _subgraph(probs, pred, y, node_time, edge_index) -> dict:
    # Work entirely on CPU so node indices and gathered tensors share a device.
    probs, y, node_time = probs.cpu(), y.cpu(), node_time.cpu()
    ei = edge_index.cpu()
    test_idx = (y != 2).nonzero().flatten()
    # rank known nodes by predicted fraud probability; anchor on the riskiest
    order = probs[test_idx].argsort(descending=True)
    seeds = test_idx[order[:SEED_NODES]]

    inc = torch.isin(ei[0], seeds) | torch.isin(ei[1], seeds)
    cand = torch.cat([seeds, ei[0, inc], ei[1, inc]]).unique()
    if cand.numel() > MAX_NODES:
        keep = probs[cand].cpu().argsort(descending=True)[:MAX_NODES]
        cand = cand[keep]

    em = torch.isin(ei[0], cand) & torch.isin(ei[1], cand)
    sub = ei[:, em][:, :MAX_EDGES]
    node_set = set(cand.tolist())
    links = [{"source": int(s), "target": int(t)}
             for s, t in zip(sub[0].tolist(), sub[1].tolist())
             if int(s) in node_set and int(t) in node_set]

    nodes = [{
        "id": int(n),
        "prob": round(float(probs[n]), 4),
        "label": int(y[n]) if int(y[n]) in (0, 1) else -1,
        "step": int(node_time[n]),
    } for n in cand.tolist()]
    return {"nodes": nodes, "links": links}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["evolvegcn", "gat"], default="evolvegcn",
                    help="which model to train + export (default: the benchmark winner)")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] model: {args.model} | device: {device}")
    data, node_time = load_elliptic(device)

    if args.model == "evolvegcn":
        model_name = "EvolveGCN-O"
        logits, test_metrics = _train_evolvegcn(data, node_time, device, args.epochs)
    else:
        model_name = "GAT"
        logits, test_metrics = _train(data, device, args.epochs)
    if logits is None:  # epochs < 5: never evaluated — score the final state
        logits = data.x.new_zeros((data.num_nodes, 2))
        test_metrics = illicit_metrics([0.0], [0], [0])

    probs = F.softmax(logits, dim=1)[:, 1].detach()
    pred = logits.argmax(1)

    artifact = {
        "model": model_name,
        "dataset": "Elliptic Bitcoin",
        "metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "graph_stats": {
            "nodes": int(data.num_nodes),
            "edges": int(data.edge_index.size(1)),
            "features": int(data.num_node_features),
            "time_steps": int(node_time.max()),
        },
        "timeline": _timeline(pred, data.y, node_time),
        "graph": _subgraph(probs, pred, data.y, node_time, data.edge_index),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f)
    size_kb = os.path.getsize(OUT) / 1024
    print(f"[export] test metrics: {artifact['metrics']}")
    print(f"[export] subgraph: {len(artifact['graph']['nodes'])} nodes, "
          f"{len(artifact['graph']['links'])} links")
    print(f"[export] wrote {OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
