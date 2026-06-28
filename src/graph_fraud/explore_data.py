"""
Elliptic Bitcoin dataset — exploration (DATA FIRST, before any modeling)
=======================================================================
Loads the dataset via PyTorch Geometric's built-in loader (no raw CSVs) and
prints the structure we need to understand before building a GNN:
  - graph size (nodes / edges / features)
  - label distribution (licit / illicit / unknown)
  - the temporal structure (49 time-steps) and the train/test split

Run:  python -m src.graph_fraud.explore_data
"""

import os
import socket

import torch

# This machine has a broken IPv6 path to some CDNs — force IPv4 so the
# PyG dataset download doesn't hang. Harmless where IPv6 works.
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only

ROOT = os.path.join("data", "elliptic")


def main() -> None:
    from torch_geometric.datasets import EllipticBitcoinDataset

    print("[elliptic] loading (downloads on first run)...", flush=True)
    ds = EllipticBitcoinDataset(root=ROOT)
    data = ds[0]
    print("\n[elliptic] Data object:", data)
    print(f"  num_nodes        : {data.num_nodes:,}")
    print(f"  num_edges        : {data.num_edges:,}")
    print(f"  num_node_features: {data.num_node_features}")
    print(f"  num_classes      : {ds.num_classes}")

    # ---- Labels: PyG maps licit=0, illicit=1; unlabeled handled via masks ----
    y = data.y
    uniq, counts = torch.unique(y, return_counts=True)
    print("\n[elliptic] label values + counts (y):")
    for v, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  y={v}: {c:,}")
    if hasattr(data, "train_mask"):
        tr, te = int(data.train_mask.sum()), int(data.test_mask.sum())
        print(f"\n[elliptic] train_mask labeled nodes: {tr:,}")
        print(f"[elliptic] test_mask  labeled nodes: {te:,}")
        # fraud rate within each split
        ytr = y[data.train_mask]
        yte = y[data.test_mask]
        print(f"  train illicit rate: {ytr.float().mean().item():.4f}")
        print(f"  test  illicit rate: {yte.float().mean().item():.4f}")

    # ---- Temporal structure: the time-step is the first node feature ----
    col0 = data.x[:, 0]
    print("\n[elliptic] feature column 0 (candidate time-step):")
    print(f"  min={col0.min().item()}, max={col0.max().item()}, "
          f"unique={torch.unique(col0).numel()}")
    ts = torch.unique(col0)
    if ts.numel() <= 60:
        print(f"  unique values: {ts.tolist()}")
    print("\n[elliptic] feature matrix shape:", tuple(data.x.shape))
    print("[elliptic] edge_index shape:", tuple(data.edge_index.shape))


if __name__ == "__main__":
    main()
