"""Regression tests for label leakage in synthesized entity fields.

Background
----------
``synthesize_entity_fields`` invents device_id / ip_address, because the raw
Sparkov dataset has neither. An earlier version invented them *using the label
in a deterministic way*:

    ring_devices = rng.choice(device_pool[: n_devices // 3], ...)   # fraud only
    primary      = rng.choice(device_pool[n_devices // 3 :])        # legit only
    card_to_prefix[card] = f"192.168.{...}"                         # fraud only

Fraud cards were confined to one slice of the device pool and handed a dedicated
``192.168.*`` range, while everyone else got ``10.*``. The graph stage then
aggregates those columns into graph_device_fraud_rate / graph_ip_fraud_rate and
feeds them to XGBoost — which is how the pipeline reported an implausible
AUC-ROC of ~0.997.

What these tests actually measure
---------------------------------
The leak is measured the way the pipeline builds it: per-entity fraud rate is
fit on TRAIN (``build_entity_graph(train)``) and applied to TEST. Two subtleties
cost real debugging time and are worth keeping in mind before editing:

  1. Card-level, not transaction-level. A fraud card also has legit
     transactions, so its device shows up on both sides. Transaction-level
     overlap looks healthy even when the card pools are fully disjoint.

  2. device_id carries irreducible signal that is *not* a synthesis artifact.
     Most cards have their own device, and the same cards span the temporal
     split, so device fraud rate acts as card reputation — a real and
     legitimately predictive production feature. Measured on the toy below, it
     scores ~0.94 even with rings switched off entirely (ring_fraction=0.0),
     versus ~0.95 at the 0.25 default. Rings contribute ~0.01 of it. So there is
     no useful AUC bound to assert on device_id; the structural check is that
     fraud and legit cards draw from one shared pool.

     ip_prefix is the honest canary: with rings off it sits at chance (~0.49),
     and the pre-fix generator scored ~0.98. That is the artifact these tests
     exist to catch.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data_prep import synthesize_entity_fields

N_CARDS = 3000
TXNS_PER_CARD = 20
FRAUD_CARD_RATE = 0.06
SPLIT = pd.Timestamp("2020-01-01")


def _toy() -> pd.DataFrame:
    """Toy transactions spanning 2019-2020, so a temporal split is possible."""
    rng = np.random.default_rng(0)
    cards = np.repeat(np.arange(N_CARDS) + 4_000_000_000_000, TXNS_PER_CARD)
    n = len(cards)
    fraud_cards = set(
        rng.choice(np.unique(cards), size=int(N_CARDS * FRAUD_CARD_RATE), replace=False)
    )
    is_fraud = np.array(
        [1 if (c in fraud_cards and rng.random() < 0.4) else 0 for c in cards], dtype=int
    )
    start = pd.Timestamp("2019-01-01").value // 10**9
    end = pd.Timestamp("2020-07-01").value // 10**9
    return pd.DataFrame(
        {
            "cc_num": cards,
            "is_fraud": is_fraud,
            "trans_dt": pd.to_datetime(rng.integers(start, end, size=n), unit="s"),
            "merchant": rng.choice([f"m_{i}" for i in range(60)], size=n),
            "lat": rng.uniform(25.0, 48.0, size=n),
            "long": rng.uniform(-124.0, -68.0, size=n),
        }
    )


def _temporal_leak_auc(out: pd.DataFrame, column: str) -> float:
    """Fit per-entity fraud rate on train, score test — mirrors graph_*_fraud_rate."""
    train = out[out["trans_dt"] < SPLIT]
    test = out[out["trans_dt"] >= SPLIT]
    rate = train.groupby(column)["is_fraud"].mean()
    scored = test[column].map(rate).fillna(train["is_fraud"].mean())
    return roc_auc_score(test["is_fraud"], scored)


def _card_pools(out: pd.DataFrame, column: str) -> tuple[set, set]:
    """Values used by cards that ever commit fraud, vs cards that never do."""
    card_fraud = out.groupby("cc_num")["is_fraud"].max()
    fraud_cards = set(card_fraud[card_fraud == 1].index)
    per_card = out.groupby("cc_num")[column].agg(set)
    fraud_vals: set = set()
    legit_vals: set = set()
    for card, vals in per_card.items():
        (fraud_vals if card in fraud_cards else legit_vals).update(vals)
    return fraud_vals, legit_vals


@pytest.fixture(scope="module")
def entities() -> pd.DataFrame:
    return synthesize_entity_fields(_toy(), seed=7)


def test_no_dedicated_fraud_address_space(entities):
    """Fraud must not get its own IP range — that alone recovers the label."""
    assert not entities["ip_prefix"].str.startswith("192.168").any(), (
        "fraud cards were given a dedicated 192.168.* range while legit cards "
        "got 10.* — ip_prefix alone recovers the label"
    )


@pytest.mark.parametrize("column", ["device_id", "ip_prefix"])
def test_card_pools_are_shared_not_disjoint(entities, column):
    """Fraud and legit cards must draw from one shared pool.

    Checked at card level: transaction-level overlap is confounded, because a
    fraud card also has legit transactions.
    """
    fraud_vals, legit_vals = _card_pools(entities, column)
    shared = len(fraud_vals & legit_vals) / len(fraud_vals)
    assert shared > 0.5, (
        f"only {shared:.0%} of the fraud {column} pool is shared with legit "
        f"cards — the generator is partitioning the pool by label"
    )


def test_ip_prefix_does_not_recover_the_label(entities):
    """ip_prefix is the canary: the pre-fix generator scored ~0.98 here."""
    auc = _temporal_leak_auc(entities, "ip_prefix")
    assert auc < 0.80, (
        f"ip_prefix recovers the test label at AUC={auc:.3f} — the generator is "
        "encoding is_fraud into the address space"
    )


def test_ip_prefix_is_pure_noise_when_rings_are_disabled():
    """With no rings, a synthesized IP must carry no label signal whatsoever.

    This isolates artifact from earned signal: any AUC above chance here is the
    generator leaking, not ring structure being detected.
    """
    out = synthesize_entity_fields(_toy(), seed=7, ring_fraction=0.0)
    auc = _temporal_leak_auc(out, "ip_prefix")
    assert 0.40 < auc < 0.60, (
        f"with rings disabled ip_prefix still scores AUC={auc:.3f}; it should sit "
        "at chance, so the address scheme itself is encoding the label"
    )
