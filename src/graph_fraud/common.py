"""
Shared utilities for the Elliptic graph-fraud models.

Splits PyG's training nodes (time-steps 1-34) into an actual training set
(steps 1-29) and a held-out validation set (steps 30-34) so we can do honest
early stopping — pick the epoch by validation AUC, then report test (steps
35-49) once. No test peeking.
"""

import os

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = os.path.join("data", "elliptic")
RAW_FEATURES = os.path.join(ROOT, "raw", "elliptic_txs_features.csv")
VAL_LO, VAL_HI = 30, 34  # validation = last training time-steps


def load_elliptic(device):
    """Return (data, node_time). data.train_mask = steps 1-29, data.val_mask =
    steps 30-34, data.test_mask = steps 35-49 (unchanged)."""
    from torch_geometric.datasets import EllipticBitcoinDataset

    data = EllipticBitcoinDataset(root=ROOT)[0]
    node_time = torch.tensor(
        pd.read_csv(RAW_FEATURES, header=None, usecols=[1])[1].to_numpy(), dtype=torch.long
    )
    in_val = (node_time >= VAL_LO) & (node_time <= VAL_HI)
    data.val_mask = data.train_mask & in_val
    data.train_mask = data.train_mask & ~in_val
    return data.to(device), node_time.to(device)


def illicit_metrics(prob, pred, true) -> dict:
    return {
        "illicit_f1": f1_score(true, pred, pos_label=1, zero_division=0),
        "illicit_precision": precision_score(true, pred, pos_label=1, zero_division=0),
        "illicit_recall": recall_score(true, pred, pos_label=1, zero_division=0),
        "auc": roc_auc_score(true, prob),
    }


def eval_logits(logits, y, mask) -> dict:
    prob = F.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    pred = logits[mask].argmax(1).cpu().numpy()
    return illicit_metrics(prob, pred, y[mask].cpu().numpy())
