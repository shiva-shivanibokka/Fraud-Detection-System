"""
Regenerate models/entity_graph.json as the detected fraud RINGS.

Earlier this exported a generic, dense entity network that didn't actually show
"rings". It now mirrors scripts/export_rings.py: fraud cards that share a device
form a ring (a connected component), and the graph is the union of the top rings
— each card plus the device(s) it shares with the rest of its ring. The force
layout lays these out as separate clusters, and every node carries its ring_id /
ring_index so the graph and the Ring Case Report dropdown line up exactly.

Output: {"nodes": [...], "links": [...]} with node fields
  id, type ("card"|"device"), ring_id, ring_index, is_fraud, txn_count, fraud_rate

Run:  python -m scripts.export_entity_graph   (then upload models/entity_graph.json to HF)
"""

from __future__ import annotations

import itertools
import json
import os

import networkx as nx
import pandas as pd

SRC = os.path.join("data", "processed", "train.parquet")
OUT = os.path.join("models", "entity_graph.json")
MIN_RING = 3      # must match scripts/export_rings.py so ring_ids align
MAX_RINGS = 25


def main() -> None:
    cols = ["cc_num", "device_id", "amt", "is_fraud"]
    df = pd.read_parquet(SRC, columns=cols)
    fraud_cards = set(df.groupby("cc_num")["is_fraud"].sum().pipe(lambda s: s[s > 0]).index)
    sub = df[df["cc_num"].isin(fraud_cards)]

    # Link fraud cards that share a device, then each component is a ring.
    g = nx.Graph()
    g.add_nodes_from(fraud_cards)
    for _, cards in sub.groupby("device_id")["cc_num"].unique().items():
        if len(cards) >= 2:
            g.add_edges_from(itertools.combinations(cards, 2))
    components = sorted((c for c in nx.connected_components(g) if len(c) >= MIN_RING),
                        key=len, reverse=True)[:MAX_RINGS]

    nodes, links = [], []
    for idx, cards in enumerate(components):
        ring_id = f"RING_{idx + 1:04d}"
        rt = sub[sub["cc_num"].isin(cards)]
        for card in cards:
            ct = df[df["cc_num"] == card]
            nodes.append({
                "id": f"card_{int(card)}", "type": "card", "ring_id": ring_id, "ring_index": idx,
                "is_fraud": 1, "txn_count": int(len(ct)),
                "fraud_rate": round(float(ct["is_fraud"].mean()), 4),
            })
        # shared devices (used by >=2 cards of this ring) become the ring's hubs
        dev_cards = rt.groupby("device_id")["cc_num"].apply(lambda s: set(s.unique()))
        for dev, dcards in dev_cards.items():
            ring_dcards = dcards & set(cards)
            if len(ring_dcards) >= 2:
                dev_id = f"dev_{dev}_{ring_id}"
                nodes.append({
                    "id": dev_id, "type": "device", "ring_id": ring_id, "ring_index": idx,
                    "is_fraud": 0, "txn_count": int(len(ring_dcards)), "fraud_rate": 0.0,
                })
                links.extend({"source": f"card_{int(c)}", "target": dev_id} for c in ring_dcards)

    artifact = {"nodes": nodes, "links": links}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f)
    n_cards = sum(1 for n in nodes if n["type"] == "card")
    print(f"[entity-graph] {len(components)} rings -> {n_cards} cards, "
          f"{len(nodes) - n_cards} shared devices, {len(links)} links "
          f"({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
