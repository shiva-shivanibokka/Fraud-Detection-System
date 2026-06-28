"""Tests for the BYOK LLM copilot layer.

The provider relay (network call to OpenAI/Groq) is monkeypatched so these run
offline and without any API key — we test our prompt construction, header
handling, validation, parsing, and the feedback sink, not the vendor.
"""

import json

import pytest

from src.llm import tasks


# ---------------------------------------------------------------------------
# Pure units (no app)
# ---------------------------------------------------------------------------
def test_parse_rule_normalizes_and_clamps():
    rule = tasks.parse_rule(
        '```json\n{"antecedent": "amt_gt=500", "action": "decline", '
        '"confidence": 1.7, "rationale": "big amount"}\n```'
    )
    assert rule["antecedent"] == ["amt_gt=500"]  # string coerced to list
    assert rule["action"] == "DECLINE"  # upper-cased
    assert rule["confidence"] == 1.0  # clamped into [0,1]
    assert rule["rationale"] == "big amount"


def test_parse_rule_rejects_non_json():
    with pytest.raises(ValueError):
        tasks.parse_rule("I could not turn that into a rule, sorry.")


def test_build_context_grounds_category_lift():
    """The grounding must read the real `antecedents` field, humanize the tokens,
    and surface a category->lift summary so the copilot can answer category
    questions (the bug was reading singular `antecedent` -> empty conditions)."""
    rules = [
        {"antecedents": ["cat_grocery_pos"], "consequents": ["FRAUD"], "lift": 1.2, "confidence": 0.3},
        {"antecedents": ["cat_gas_transport", "state_other"], "consequents": ["FRAUD"],
         "lift": 9.0, "confidence": 0.9},
    ]
    ctx = tasks.build_context(
        question="which merchant categories have the highest fraud lift?",
        metrics={"total_scored": 5},
        rules=rules,
        rings=[{"cards": ["a", "b"], "devices": ["d1"]}],
        feature_importance=[{"name": "amt", "importance": 0.3}],
        blocklist_size=7,
        thresholds={"review": 0.4, "decline": 0.8},
    )
    # gas transport (lift 9) ranks above grocery in the category summary
    assert ctx["category_fraud_signals"][0]["category"] == "gas transport"
    assert ctx["category_fraud_signals"][0]["max_lift"] == 9.0
    # rule conditions are humanized, not raw tokens or empty
    conds = [c for r in ctx["top_rules"] for c in r["conditions"]]
    assert any("category gas transport" in c for c in conds)
    assert ctx["blocklist_size"] == 7
    assert ctx["fraud_rings"][0]["cards"] == 2


# ---------------------------------------------------------------------------
# Endpoints (app, chat() monkeypatched)
# ---------------------------------------------------------------------------
def test_llm_providers_registry(client):
    r = client.get("/llm/providers")
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["providers"]}
    assert {"openai", "groq"} <= ids
    for p in r.json()["providers"]:
        assert p["models"]  # each provider exposes a model allow-list


def test_copilot_rejects_unknown_provider(client):
    # No X-LLM-* headers -> empty provider fails validation (400) before network
    r = client.post("/llm/copilot", json={"question": "what's risky?"})
    assert r.status_code == 400


def test_copilot_requires_key(client):
    # Valid provider but no key -> 401 before any network call
    r = client.post(
        "/llm/copilot",
        json={"question": "what's risky?"},
        headers={"X-LLM-Provider": "openai", "X-LLM-Model": "gpt-4o"},
    )
    assert r.status_code == 401


def test_copilot_with_mocked_provider(client, monkeypatch):
    async def fake_chat(provider, model, key, messages, **kw):
        assert provider == "openai" and key == "sk-test"
        return "Casino merchants show 9x lift."

    monkeypatch.setattr("src.api.main.llm_providers.chat", fake_chat)
    r = client.post(
        "/llm/copilot",
        json={"question": "what is risky?"},
        headers={"X-LLM-Provider": "openai", "X-LLM-Model": "gpt-4o", "X-LLM-Key": "sk-test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Casino merchants show 9x lift."
    assert "grounded_on" in body


def test_rule_from_text_with_mocked_provider(client, monkeypatch):
    async def fake_chat(provider, model, key, messages, **kw):
        return json.dumps(
            {"antecedent": ["amt_gt=1000", "hour_lt=6"], "action": "REVIEW",
             "confidence": 0.7, "rationale": "large late-night charge"}
        )

    monkeypatch.setattr("src.api.main.llm_providers.chat", fake_chat)
    r = client.post(
        "/llm/rule-from-text",
        json={"text": "flag charges over $1000 before 6am"},
        headers={"X-LLM-Provider": "groq", "X-LLM-Model": "llama-3.3-70b-versatile",
                 "X-LLM-Key": "gsk_test"},
    )
    assert r.status_code == 200
    rule = r.json()["rule"]
    assert rule["action"] == "REVIEW"
    assert "amt_gt=1000" in rule["antecedent"]


def test_case_report_with_direct_ring(client, monkeypatch):
    async def fake_chat(provider, model, key, messages, **kw):
        return "## Summary\nA 12-card ring sharing 2 devices."

    monkeypatch.setattr("src.api.main.llm_providers.chat", fake_chat)
    r = client.post(
        "/llm/case-report",
        json={"ring": {"cards": list(range(12)), "devices": ["d1", "d2"]}},
        headers={"X-LLM-Provider": "openai", "X-LLM-Key": "sk-test"},
    )
    assert r.status_code == 200
    assert "Summary" in r.json()["report"]


def test_feedback_persists_to_jsonl(client, monkeypatch, tmp_path):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr("src.api.main.settings.feedback_path", str(path))
    r = client.post(
        "/feedback",
        json={"trans_id": "t1", "decision": "REVIEW", "fraud_score": 0.55, "label": "fraud"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    line = json.loads(path.read_text().strip())
    assert line["trans_id"] == "t1" and line["label"] == "fraud"


def test_feedback_rejects_bad_label(client):
    r = client.post("/feedback", json={"trans_id": "t2", "label": "maybe"})
    assert r.status_code == 422
