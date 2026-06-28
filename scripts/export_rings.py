"""
Regenerate models/ring_stats.json as MULTIPLE distinct fraud rings.

The shipped artifact had a single 467-card "ring" — the similarity-model
detection collapsed every fraud card into one connected component. Real fraud
rings are cards that share hardware: here we link two fraud cards when they
share a `device_id` (the strongest co-occurrence signal; merchants/IPs are too
broadly shared), then each connected component of >= MIN_RING cards is a ring.

Writes the same schema the API + frontend already expect:
  ring_id, n_cards, n_txns, fraud_rate, total_amt, avg_amt, n_states,
  n_merchants, first_txn (ms), last_txn (ms), span_days

Run:  python -m scripts.export_rings   (then upload models/ring_stats.json to HF)
"""

from __future__ import annotations

import itertools
import json
import os

import networkx as nx
import pandas as pd

SRC = os.path.join("data", "processed", "train.parquet")
OUT = os.path.join("models", "ring_stats.json")
MIN_RING = 3   # a ring needs at least this many cards
MAX_RINGS = 25


def main() -> None:
    cols = ["cc_num", "device_id", "merchant", "amt", "is_fraud", "state", "unix_time"]
    df = pd.read_parquet(SRC, columns=cols)

    fraud_cards = set(df.groupby("cc_num")["is_fraud"].sum().pipe(lambda s: s[s > 0]).index)
    sub = df[df["cc_num"].isin(fraud_cards)]
    print(f"[rings] {len(fraud_cards)} fraud cards")

    # Link fraud cards that share a device (each device -> clique of its cards).
    g = nx.Graph()
    g.add_nodes_from(fraud_cards)
    for _, cards in sub.groupby("device_id")["cc_num"].unique().items():
        if len(cards) >= 2:
            g.add_edges_from(itertools.combinations(cards, 2))

    components = [c for c in nx.connected_components(g) if len(c) >= MIN_RING]
    components.sort(key=len, reverse=True)
    print(f"[rings] {len(components)} components with >= {MIN_RING} cards")

    rings = []
    for i, cards in enumerate(components[:MAX_RINGS], start=1):
        t = df[df["cc_num"].isin(cards)]
        first, last = int(t["unix_time"].min()), int(t["unix_time"].max())
        rings.append({
            "ring_id": f"RING_{i:04d}",
            "n_cards": int(len(cards)),
            "n_txns": int(len(t)),
            "fraud_rate": round(float(t["is_fraud"].mean()), 6),
            "total_amt": round(float(t["amt"].sum()), 2),
            "avg_amt": round(float(t["amt"].mean()), 4),
            "n_states": int(t["state"].nunique()),
            "n_merchants": int(t["merchant"].nunique()),
            "first_txn": first * 1000,
            "last_txn": last * 1000,
            "span_days": int((last - first) / 86400),
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rings, f)
    sizes = [r["n_cards"] for r in rings]
    print(f"[rings] wrote {len(rings)} rings -> {OUT} "
          f"(card counts: {sizes[:10]}{'...' if len(sizes) > 10 else ''})")


if __name__ == "__main__":
    main()
