"""
Export the categorical label-encoder maps the model was trained on.

The model expects category_enc / gender_enc / state_enc / merchant_enc, but the
serving API had no way to compute them, so it fed 0 — a train/serve skew that
made the model ignore category/state/merchant (audit F1). The training parquet
carries both the raw value and its encoded value, so we recover the *exact*
mapping the model learned (no re-fit, no risk of a different encoding).

Writes models/encoders.json: {field: {"map": {value: int}, "fallback": int}}.
The fallback (mode's encoding) is used for values unseen at training time
(e.g. a free-text merchant name).

Run:  python -m scripts.export_encoders   (then upload models/encoders.json to HF)
"""

from __future__ import annotations

import json
import os

import pandas as pd

SRC = os.path.join("data", "processed", "train_features.parquet")
OUT = os.path.join("models", "encoders.json")
FIELDS = ["category", "gender", "state", "merchant"]


def main() -> None:
    cols = [c for f in FIELDS for c in (f, f"{f}_enc")]
    df = pd.read_parquet(SRC, columns=cols)
    encoders: dict = {}
    for f in FIELDS:
        sub = df[[f, f"{f}_enc"]].dropna()
        mapping = {
            str(k): int(v)
            for k, v in sub.drop_duplicates(subset=[f]).set_index(f)[f"{f}_enc"].items()
        }
        fallback = int(sub[f"{f}_enc"].mode().iloc[0])
        encoders[f] = {"map": mapping, "fallback": fallback}
        print(f"[encoders] {f}: {len(mapping)} values, fallback={fallback}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(encoders, fh)
    print(f"[encoders] wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
