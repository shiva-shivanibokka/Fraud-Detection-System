"""
Elliptic — temporal structure exploration.

PyG drops the raw time-step from data.x, so we read it (cheaply, 2 columns) from
the raw features file PyG already downloaded, verify it aligns with PyG's node
ordering + train/test split, and show the per-time-step label dynamics that
motivate temporal modeling (TGAT).

Run:  python -m src.graph_fraud.explore_temporal
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.join("data", "elliptic")
RAW_FEATURES = os.path.join(ROOT, "raw", "elliptic_txs_features.csv")


def main() -> None:
    from torch_geometric.datasets import EllipticBitcoinDataset

    data = EllipticBitcoinDataset(root=ROOT)[0]
    y = data.y.numpy()

    # time-step = column 1 of the raw features file; node index == row order in PyG
    feat = pd.read_csv(RAW_FEATURES, header=None, usecols=[0, 1])
    feat.columns = ["txId", "timestep"]
    ts = feat["timestep"].to_numpy()
    assert len(ts) == data.num_nodes, f"{len(ts)} vs {data.num_nodes}"

    print(f"[temporal] time-steps: {ts.min()}..{ts.max()} ({len(np.unique(ts))} unique)")

    # Verify alignment: PyG's train/test split should be a clean temporal cut.
    tr_ts = ts[data.train_mask.numpy()]
    te_ts = ts[data.test_mask.numpy()]
    print(f"[temporal] train_mask time-step range: {tr_ts.min()}..{tr_ts.max()}")
    print(f"[temporal] test_mask  time-step range: {te_ts.min()}..{te_ts.max()}")
    print("[temporal] -> clean temporal split confirmed" if tr_ts.max() < te_ts.min()
          else "[temporal] -> WARNING: train/test overlap in time")

    print("\n[temporal] per-time-step label counts (illicit / licit / unknown):")
    print(f"  {'t':>3} {'total':>7} {'illicit':>8} {'licit':>7} {'unknown':>8} {'illicit%':>9}")
    for t in range(int(ts.min()), int(ts.max()) + 1):
        m = ts == t
        yt = y[m]
        n_ill = int((yt == 1).sum())
        n_lic = int((yt == 0).sum())
        n_unk = int((yt == 2).sum())
        lab = n_ill + n_lic
        rate = (n_ill / lab) if lab else 0.0
        print(f"  {t:>3} {int(m.sum()):>7} {n_ill:>8} {n_lic:>7} {n_unk:>8} {rate:>8.1%}")


if __name__ == "__main__":
    main()
