"""
Regenerate models/entity_graph.json for the Fraud Ring Graph tab.

The previously published artifact predated the edge-building fix, so it had 0
edges (just a scatter of dots). This rebuilds the entity graph from the training
data and exports a fraud-centric *sampled* subgraph — the riskiest cards plus
the devices/IPs/merchants they share (which is where rings show up) — small
enough to render smoothly in D3.

Output keys match what the frontend reads: {"nodes": [...], "links": [...]}.

Run locally (needs data/processed/train.parquet), then upload to HF Hub:
    python -m scripts.export_entity_graph
"""

from __future__ import annotations

import json
import os

import pandas as pd

from src.graph.entity_graph import build_entity_graph

OUT = os.path.join("models", "entity_graph.json")
SEED_CARDS = 140      # riskiest cards to anchor on
CARDS_PER_ENTITY = 6  # how many sharers to pull in per device/ip/merchant
MAX_NODES = 500
MAX_LINKS = 1300


def main() -> None:
    cols = ["cc_num", "device_id", "ip_prefix", "merchant", "amt", "is_fraud"]
    df = pd.read_parquet("data/processed/train.parquet", columns=cols)
    print(f"[entity-graph] loaded {len(df):,} rows")

    g = build_entity_graph(df)
    print(f"[entity-graph] full graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    cards = [(n, d) for n, d in g.nodes(data=True) if d.get("node_type") == "card"]
    cards.sort(key=lambda x: (x[1].get("fraud_rate", 0), x[1].get("txn_count", 0)), reverse=True)
    seeds = [n for n, _ in cards[:SEED_CARDS]]

    keep: set = set(seeds)
    for s in seeds:                       # add the entities each risky card touches
        keep.update(g.neighbors(s))
    for ent in [n for n in list(keep) if g.nodes[n].get("node_type") != "card"]:
        for c in list(g.neighbors(ent))[:CARDS_PER_ENTITY]:  # other cards on that entity (rings)
            keep.add(c)
        if len(keep) >= MAX_NODES:
            break
    keep = set(list(keep)[:MAX_NODES])

    h = g.subgraph(keep)
    nodes = [{
        "id": n,
        "type": d.get("node_type", "unknown"),
        "fraud_rate": round(float(d.get("fraud_rate", 0) or 0), 4),
        "is_fraud": int(d.get("is_fraud_node", 0) or 0),
        "txn_count": int(d.get("txn_count", 0) or 0),
    } for n, d in h.nodes(data=True)]
    links = [{"source": u, "target": v, "type": h.edges[u, v].get("edge_type", "")}
             for u, v in h.edges()][:MAX_LINKS]

    artifact = {"nodes": nodes, "links": links}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(artifact, f)
    size_kb = os.path.getsize(OUT) / 1024
    fraud_nodes = sum(1 for n in nodes if n["is_fraud"])
    print(f"[entity-graph] exported {len(nodes)} nodes ({fraud_nodes} fraud), "
          f"{len(links)} links -> {OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
