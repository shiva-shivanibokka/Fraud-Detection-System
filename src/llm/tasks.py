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


_READABLE = {
    "hour_night": "nighttime", "hour_day": "daytime", "weekend": "weekend",
    "weekday": "weekday", "amt_low": "low amount", "amt_medium": "medium amount",
    "amt_high": "high amount", "geo_far": "far from home", "geo_near": "near home",
}


def _readable(token: str) -> str:
    """FP-Growth item -> plain English so the model understands the conditions."""
    if not token:
        return ""
    if token.startswith("cat_"):
        return f"category {token[4:].replace('_', ' ')}"
    if token.startswith("state_"):
        s = token[6:]
        return "other state" if s == "other" else f"state {s.upper()}"
    return _READABLE.get(token, token.replace("_", " "))


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
    rules = rules or []

    def rule_relevance(r: dict) -> tuple:
        overlap = len(q & _tokens(" ".join(_readable(a) for a in r.get("antecedents", []))))
        return (overlap, float(r.get("lift", 0.0)))

    top_rules = sorted(rules, key=rule_relevance, reverse=True)[:max_rules]

    # Aggregate category-level fraud lift so questions like "which categories have
    # the highest fraud lift?" are directly answerable from the grounding.
    cat_signals: dict[str, dict] = {}
    for r in rules:
        lift = float(r.get("lift", 0) or 0)
        for a in r.get("antecedents", []):
            if a.startswith("cat_"):
                cat = a[4:].replace("_", " ")
                if lift > cat_signals.get(cat, {}).get("max_lift", 0):
                    cat_signals[cat] = {
                        "category": cat, "max_lift": round(lift, 2),
                        "confidence": round(float(r.get("confidence", 0) or 0), 3),
                    }
    top_categories = sorted(cat_signals.values(), key=lambda x: x["max_lift"], reverse=True)[:10]

    def ring_size(r: dict) -> int:
        # ring_stats.json carries aggregate counts: n_cards / n_merchants /
        # total_amt (no per-card lists) — read those, not "cards" (audit F3).
        return int(r.get("n_cards", 0) or 0) if isinstance(r, dict) else 0

    top_rings = sorted(rings or [], key=ring_size, reverse=True)[:max_rings]
    ring_summ = [
        {
            "ring_id": r.get("ring_id", i),
            "cards": ring_size(r),
            "merchants": int(r.get("n_merchants", 0) or 0),
            "fraud_rate": r.get("fraud_rate"),
            "total_amount": r.get("total_amt"),
            "span_days": r.get("span_days"),
        }
        for i, r in enumerate(top_rings)
        if isinstance(r, dict)
    ]

    return {
        "metrics": metrics or {},
        "decision_thresholds": thresholds or {},
        "blocklist_size": blocklist_size,
        "category_fraud_signals": top_categories,
        "top_rules": [
            {
                "conditions": [_readable(a) for a in r.get("antecedents", [])],
                "predicts": "fraud" if "FRAUD" in (r.get("consequents") or []) else "unknown",
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
