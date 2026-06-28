"""
Retrain trigger — aggregate analyst feedback and decide whether a retrain is due.

This is the *trigger* half of the active-learning loop (Sprint 4, item 20). It
reads the ✓/✗ labels analysts submit via POST /feedback — from Supabase if
configured, else the local JSONL sink — and reports how many new labels have
accumulated. When the count clears a threshold it emits a retrain signal that a
scheduled workflow (or a human) acts on.

It deliberately does NOT run training here: retraining needs the full labelled
dataset, which does not live on the free-tier serving box. In CI the script runs
in "report" mode and is safe to run with zero feedback. To wire the actual
retrain, have the workflow call `python -m src.model.train` after this script
exits 0 with a signal, in an environment that has the data mounted.

Usage:
    python -m scripts.retrain_from_feedback              # report only
    python -m scripts.retrain_from_feedback --threshold 50
Exit code 0 always (so CI never breaks); the retrain signal is on stdout and in
the GITHUB_OUTPUT file (`retrain=true|false`) for a downstream workflow step.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from src.config import settings


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _load_supabase() -> list[dict]:
    if not (settings.supabase_url and settings.supabase_key):
        return []
    try:
        import httpx

        url = f"{settings.supabase_url.rstrip('/')}/rest/v1/analyst_feedback"
        headers = {
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}",
        }
        r = httpx.get(url, headers=headers, params={"select": "label"}, timeout=15.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001 — report-only, never fail the job
        print(f"[retrain] supabase read skipped: {exc}")
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=50,
                    help="new labels required to signal a retrain")
    args = ap.parse_args()

    rows = _load_supabase() or _load_jsonl(settings.feedback_path)
    counts = Counter((r.get("label") or "").lower() for r in rows)
    total = sum(counts.values())
    fraud, legit = counts.get("fraud", 0), counts.get("legit", 0)

    print(f"[retrain] feedback labels: total={total} fraud={fraud} legit={legit}")
    print(f"[retrain] threshold={args.threshold}")

    retrain = total >= args.threshold and fraud > 0 and legit > 0
    print(f"[retrain] signal: {'RETRAIN' if retrain else 'hold'}")

    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"retrain={'true' if retrain else 'false'}\n")
            f.write(f"label_count={total}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
