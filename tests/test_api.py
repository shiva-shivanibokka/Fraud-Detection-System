"""Integration tests for the fraud-detection API endpoints."""


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    for key in ("model_loaded", "feature_cols", "rules_loaded", "redis_mode"):
        assert key in data
    # feature_cols is always a positive count (real model: 137, demo fallback: 11)
    assert data["feature_cols"] > 0


def test_score_returns_valid_decision(client):
    r = client.post("/score", json={"cc_num": "4111111111111111", "amt": 42.50})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert 0.0 <= d["fraud_score"] <= 1.0
    assert d["trans_id"]
    assert isinstance(d["reasons"], list) and d["reasons"]
    assert d["latency_ms"] >= 0.0


def test_score_includes_conformal_fields(client):
    """Every score carries a conformal uncertainty signal (MAPIE LAC)."""
    d = client.post("/score", json={"cc_num": "4111111111111111", "amt": 42.50}).json()
    assert d["confidence_label"] in {
        "confident_fraud", "confident_legit", "uncertain", "unknown"
    }
    assert isinstance(d["prediction_set"], list)
    assert all(c in {"fraud", "legit"} for c in d["prediction_set"])
    assert 0.0 <= d["conformal_coverage"] <= 1.0


def test_score_requires_amount_and_card(client):
    # cc_num and amt are required fields -> 422 when missing
    r = client.post("/score", json={"cc_num": "4111111111111111"})
    assert r.status_code == 422


def test_velocity_hard_cap_declines(client):
    """6 rapid transactions on one card must trip the Layer-1 velocity cap."""
    card = "9000000000000006"
    last = None
    for _ in range(6):
        last = client.post("/score", json={"cc_num": card, "amt": 50.0}).json()
    assert last["decision"] == "DECLINE"
    assert last["layer_triggered"] == "rules"
    assert last["fraud_score"] == 1.0


def test_metrics_shape(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    for key in ("total_scored", "decline_rate", "latency_p50_ms"):
        assert key in data


def test_fraud_rules_endpoint(client):
    r = client.get("/fraud-rules")
    assert r.status_code == 200
    assert "rules" in r.json()


def test_feature_importance_endpoint(client):
    r = client.get("/feature-importance")
    assert r.status_code == 200
    assert "features" in r.json()
