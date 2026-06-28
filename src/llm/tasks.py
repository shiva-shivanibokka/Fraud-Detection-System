"""
Prompt construction + lightweight grounded retrieval for the three LLM
features: the analyst copilot, one-click fraud-ring case reports, and the
natural-language rule editor.

These are pure functions over plain data (no FastAPI, no live LLM), so they are
unit-testable in isolation. "Retrieval" here means selecting the most relevant
slices of the system's own fraud knowledge — FP-Growth rules, ring stats,
metrics, feature importances — and grounding the model on that JSON. On the
free-tier serving box we deliberately avoid an embeddings/pgvector dependency
(it would pull torch); the structured fraud knowledge is small enough to ground
directly, and the model is instructed to answer only from it.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Analyst copilot
# ---------------------------------------------------------------------------
COPILOT_SYSTEM = (
    "You are a fraud-analyst copilot embedded in a real-time card-fraud "
    "detection system. Answer the analyst's question using ONLY the JSON CONTEXT "
    "provided. Be concise and concrete: cite rule lift/confidence, ring sizes, "
    "and metric values where relevant. If the context does not contain the "
    "answer, say so plainly instead of guessing. Never invent numbers."
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 2}


def build_context(
    *,
    question: str,
    metrics: dict,
    rules: list[dict],
    rings: list[dict],
    feature_importance: list[dict],
    blocklist_size: int,
    thresholds: dict,
    max_rules: int = 8,
    max_rings: int = 5,
) -> dict:
    """Select the most question-relevant fraud knowledge to ground the model."""
    q = _tokens(question)

    def rule_relevance(r: dict) -> tuple:
        overlap = len(q & _tokens(" ".join(r.get("antecedent", []))))
        return (overlap, float(r.get("lift", 0.0)))

    top_rules = sorted(rules or [], key=rule_relevance, reverse=True)[:max_rules]

    def ring_size(r: dict) -> int:
        return len(r.get("cards", []) or []) if isinstance(r, dict) else 0

    top_rings = sorted(rings or [], key=ring_size, reverse=True)[:max_rings]
    ring_summ = [
        {
            "ring_id": r.get("ring_id", i),
            "cards": ring_size(r),
            "shared_devices": len(r.get("devices", []) or []),
            "fraud_rate": r.get("fraud_rate"),
            "total_amount": r.get("total_amount"),
        }
        for i, r in enumerate(top_rings)
        if isinstance(r, dict)
    ]

    return {
        "metrics": metrics or {},
        "decision_thresholds": thresholds or {},
        "blocklist_size": blocklist_size,
        "top_rules": [
            {
                "antecedent": r.get("antecedent", []),
                "confidence": r.get("confidence"),
                "lift": r.get("lift"),
                "support": r.get("support"),
            }
            for r in top_rules
        ],
        "fraud_rings": ring_summ,
        "top_features": (feature_importance or [])[:10],
    }


def copilot_messages(context: dict, question: str) -> list[dict]:
    return [
        {"role": "system", "content": COPILOT_SYSTEM},
        {
            "role": "user",
            "content": f"CONTEXT:\n{json.dumps(context, default=str)}\n\nQUESTION: {question}",
        },
    ]


# ---------------------------------------------------------------------------
# Fraud-ring case report
# ---------------------------------------------------------------------------
CASE_REPORT_SYSTEM = (
    "You are a senior fraud investigator. Write a concise case report in "
    "markdown (~180 words) for the fraud ring described in the JSON. Structure: "
    "a one-line **Summary**, **Scale** (cards, shared devices/IPs, total "
    "exposure), **Risk signals** (the strongest indicators present in the data), "
    "and a **Recommended action**. Use only the data provided — do not invent "
    "card numbers, names, or amounts that are not present."
)


def case_report_messages(ring: dict) -> list[dict]:
    return [
        {"role": "system", "content": CASE_REPORT_SYSTEM},
        {"role": "user", "content": f"FRAUD RING DATA:\n{json.dumps(ring, default=str)}"},
    ]


# ---------------------------------------------------------------------------
# Natural-language rule editor
# ---------------------------------------------------------------------------
ALLOWED_RULE_FIELDS = [
    "amt_gt",
    "amt_lt",
    "category",
    "merchant",
    "hour_gt",
    "hour_lt",
    "is_night",
    "geo_distance_km_gt",
    "vel_card_1min_count_gt",
    "state",
]

RULE_SYSTEM = (
    "Convert the analyst's plain-English fraud rule into a STRICT JSON object and "
    "output ONLY that JSON (no prose, no markdown fences). Schema:\n"
    '{"antecedent": ["field=value", ...], "action": "DECLINE" | "REVIEW" | '
    '"FLAG", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}\n'
    f"Allowed fields for antecedent terms: {', '.join(ALLOWED_RULE_FIELDS)}. "
    "Encode thresholds in the field name, e.g. amt_gt=500 means amount greater "
    "than 500, hour_lt=6 means before 6am. If the request is ambiguous, pick "
    "sensible defaults and state them in the rationale."
)


def rule_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": RULE_SYSTEM},
        {"role": "user", "content": text},
    ]


def parse_rule(content: str) -> dict:
    """Extract and validate the rule JSON from a model response.

    Tolerates code fences or stray prose around the JSON object. Returns a
    normalized rule dict; raises ValueError if no valid object is found.
    """
    if not content:
        raise ValueError("Empty response from model.")
    text = content.strip()
    # strip ```json ... ``` fences if present
    fence = re.search(r"\{.*\}", text, re.DOTALL)
    if not fence:
        raise ValueError("No JSON object found in model response.")
    obj = json.loads(fence.group(0))

    antecedent = obj.get("antecedent") or []
    if isinstance(antecedent, str):
        antecedent = [antecedent]
    action = str(obj.get("action", "REVIEW")).upper()
    if action not in {"DECLINE", "REVIEW", "FLAG"}:
        action = "REVIEW"
    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "antecedent": [str(a) for a in antecedent],
        "action": action,
        "confidence": round(confidence, 3),
        "rationale": str(obj.get("rationale", "")),
    }
