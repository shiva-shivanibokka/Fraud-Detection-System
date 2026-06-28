"""
Fraud Detection API - 3-layer decision architecture
Layer 1: Hard rules engine  (<1ms)
Layer 2: ML model scoring   (<20ms) via ONNX / joblib fallback
Layer 3: SHAP explanation generation
"""

import asyncio
import collections
import datetime as dt
import json
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.download_models import ensure_models
from src.llm import providers as llm_providers
from src.llm import tasks as llm_tasks
from src.velocity.feature_store import VelocityFeatureStore

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("fraud_api")

# ---------------------------------------------------------------------------
# Sentry error tracking (optional — no-ops unless SENTRY_DSN is set).
# ---------------------------------------------------------------------------
if settings.sentry_dsn:
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            traces_sample_rate=0.1,
            # Never let PII (card numbers, keys) ride along in event payloads.
            send_default_pii=False,
        )
        logger.info("sentry_initialized", environment=settings.environment)
    except Exception as exc:  # noqa: BLE001
        logger.warning("sentry_init_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", settings.model_dir)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
class AppState:
    onnx_session = None
    joblib_model = None
    feature_cols: list[str] = []
    card_embeddings: dict = {}
    fraud_rules: list[dict] = []
    known_fraud_cards: set[str] = set()
    velocity_store: VelocityFeatureStore | None = None
    shap_explainer = None
    latency_history: collections.deque = collections.deque(maxlen=1000)
    model_latency_history: collections.deque = collections.deque(maxlen=1000)
    total_scored: int = 0
    decline_count: int = 0
    onnx_available: bool = False
    redis_available: bool = False
    conformal: dict = {}
    cf_ranges: dict = {}
    dice_explainer = None


state = AppState()


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Inf) with None.

    Model JSON files (e.g. fraud_rules.json) can contain NaN values, which
    Python's json writes but Starlette's response encoder (allow_nan=False)
    rejects with a 500. Sanitizing on load keeps every endpoint serializable.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r") as f:
            return _json_safe(json.load(f))
    except Exception as exc:
        logger.warning("json_load_failed", path=path, error=str(exc))
        return default if default is not None else {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Ensure model artifacts exist (pull from HF Hub if not cached) ----
    ensure_models(settings.hf_repo_id, settings.model_dir)

    # ---- Load ONNX model ----
    try:
        import onnxruntime as ort

        onnx_path = os.path.join(MODELS_DIR, "fraud_model.onnx")
        state.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        state.onnx_available = True
        logger.info("model_loaded", backend="onnx")
    except Exception as exc:
        logger.warning("onnx_load_failed", error=str(exc))
        try:
            import joblib

            state.joblib_model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.pkl"))
            logger.info("model_loaded", backend="joblib")
        except Exception as exc2:
            logger.error("model_load_failed", error=str(exc2))

    # ---- Load feature columns ----
    try:
        import joblib

        state.feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
    except Exception:
        state.feature_cols = [
            "amt",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_night",
            "age",
            "geo_distance_km",
            "city_pop",
            "vel_card_1min_count",
            "vel_card_1hr_count",
            "vel_ip_prefix_1min_count",
        ]

    # ---- Load card embeddings ----
    try:
        import joblib

        state.card_embeddings = joblib.load(os.path.join(MODELS_DIR, "card_embeddings.pkl")) or {}
    except Exception:
        state.card_embeddings = {}

    # ---- Load FP-Growth rules ----
    state.fraud_rules = _load_json(os.path.join(MODELS_DIR, "fraud_rules.json"), default=[])

    # ---- Load conformal calibration (uncertainty band) ----
    state.conformal = _load_json(os.path.join(MODELS_DIR, "conformal.json"), default={})

    # ---- Load DICE counterfactual feature ranges (explainer built lazily) ----
    state.cf_ranges = _load_json(os.path.join(MODELS_DIR, "cf_ranges.json"), default={})

    # ---- Build blocklist from ring stats ----
    # ring_stats.json may be either a bare list of ring objects or a dict with a
    # "rings" key. Normalize to a list so startup never crashes on either shape.
    ring_stats = _load_json(os.path.join(MODELS_DIR, "ring_stats.json"), default={})
    rings = ring_stats.get("rings", []) if isinstance(ring_stats, dict) else ring_stats
    for ring in rings or []:
        cards = ring.get("cards") if isinstance(ring, dict) else None
        for card in cards or []:
            state.known_fraud_cards.add(str(card))

    # ---- Velocity store (canonical: Redis or in-memory fallback) ----
    state.velocity_store = VelocityFeatureStore()
    state.redis_available = state.velocity_store.use_redis

    # ---- SHAP explainer (best-effort background init) ----
    try:
        import shap

        model = state.joblib_model
        if model is not None and hasattr(model, "predict_proba"):
            state.shap_explainer = shap.TreeExplainer(model)
            logger.info("shap_explainer_ready")
    except Exception as exc:
        logger.warning("shap_init_failed", error=str(exc))

    logger.info(
        "startup_complete",
        rules=len(state.fraud_rules),
        blocklist=len(state.known_fraud_cards),
    )
    yield
    logger.info("shutdown")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    trans_id: str = ""
    cc_num: str
    device_id: str = ""
    ip_prefix: str = "192.168"
    merchant: str = ""
    category: str = ""
    amt: float
    hour: int = 12
    day_of_week: int = 1
    is_weekend: int = 0
    is_night: int = 0
    age: float = 35.0
    geo_distance_km: float = 0.0
    city_pop: int = 50000
    state: str = ""
    gender: str = ""
    timestamp: float = 0.0


class FraudRule(BaseModel):
    antecedent: list[str]
    confidence: float
    lift: float
    support: float


class ScoreResponse(BaseModel):
    trans_id: str
    decision: str
    fraud_score: float
    layer_triggered: str
    reasons: list[str]
    latency_ms: float
    model_latency_ms: float
    triggered_rules: list[dict]
    # Conformal uncertainty (MAPIE LAC); empty/defaults when calibration absent.
    confidence_label: str = "unknown"
    prediction_set: list[str] = []
    conformal_coverage: float = 0.0


class CounterfactualChange(BaseModel):
    feature: str
    original: float
    suggested: float


class Counterfactual(BaseModel):
    changes: list[CounterfactualChange]
    resulting_class: str  # "legit" or "fraud"


class CounterfactualResponse(BaseModel):
    trans_id: str
    original_decision: str
    original_fraud_score: float
    counterfactuals: list[Counterfactual]
    available: bool = True
    message: str = ""


# ---- LLM copilot (BYOK) request bodies ----
class CopilotRequest(BaseModel):
    question: str


class CaseReportRequest(BaseModel):
    ring_id: int | None = None  # index into ring_stats.json
    ring: dict | None = None  # or pass the ring object directly


class RuleFromTextRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    trans_id: str
    decision: str = ""
    fraud_score: float = 0.0
    label: str  # analyst ground-truth: "fraud" or "legit"
    note: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    arr = sorted(data)
    idx = int(len(arr) * pct / 100)
    return arr[min(idx, len(arr) - 1)]


def _raw_feature_dict(req: TransactionRequest, vel: dict) -> dict:
    return {
        "amt": req.amt,
        "hour": req.hour,
        "day_of_week": req.day_of_week,
        "is_weekend": req.is_weekend,
        "is_night": req.is_night,
        "age": req.age,
        "geo_distance_km": req.geo_distance_km,
        "city_pop": req.city_pop,
        **vel,  # all canonical velocity features from VelocityFeatureStore
    }


def _build_feature_vector(req: TransactionRequest, vel: dict) -> np.ndarray:
    raw = _raw_feature_dict(req, vel)
    vec = [raw.get(col, 0.0) for col in state.feature_cols]
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def _run_model(features: np.ndarray) -> tuple[float, float]:
    t0 = time.perf_counter()
    score = 0.5
    if state.onnx_session is not None:
        input_name = state.onnx_session.get_inputs()[0].name
        out = state.onnx_session.run(None, {input_name: features})
        proba = out[1] if len(out) > 1 else out[0]
        arr = np.array(proba).flatten()
        score = float(arr[-1]) if len(arr) > 1 else float(arr[0])
    elif state.joblib_model is not None:
        if hasattr(state.joblib_model, "predict_proba"):
            score = float(state.joblib_model.predict_proba(features)[0][1])
        else:
            score = float(state.joblib_model.predict(features)[0])
    else:
        # Demo mode: deterministic score from features
        amt = features[0][0]
        vel = features[0][8] if features.shape[1] > 8 else 0
        score = min(1.0, (amt / 5000.0) * 0.4 + (vel / 6.0) * 0.6)
    latency_ms = (time.perf_counter() - t0) * 1000
    return float(score), latency_ms


def _conformal_fields(score: float) -> dict:
    """Map a fraud score to a conformal prediction set + confidence label.

    Uses the exported LAC threshold t (models/conformal.json): a class is in
    the 90%-coverage prediction set iff its probability >= t. For binary fraud:
        score >= t        -> {fraud}  confident_fraud
        score <= 1 - t    -> {legit}  confident_legit
        otherwise         -> {}       uncertain (route to review)
    """
    cfg = state.conformal or {}
    t = cfg.get("threshold")
    if t is None:
        return {"confidence_label": "unknown", "prediction_set": [],
                "conformal_coverage": 0.0}
    pred_set = []
    if score >= t:
        pred_set.append("fraud")
    if (1.0 - score) >= t:
        pred_set.append("legit")
    if pred_set == ["fraud"]:
        label = "confident_fraud"
    elif pred_set == ["legit"]:
        label = "confident_legit"
    else:
        label = "uncertain"
    return {
        "confidence_label": label,
        "prediction_set": pred_set,
        "conformal_coverage": float(cfg.get("confidence_level", 0.0)),
    }


def _cf_query_df(req: TransactionRequest, vel: dict):
    import pandas as pd

    raw = _raw_feature_dict(req, vel)
    row = {col: float(raw.get(col, 0.0)) for col in state.feature_cols}
    return pd.DataFrame([row], columns=state.feature_cols)


def _get_dice_explainer():
    """Build (and cache) the DICE explainer lazily.

    dice-ml is imported only on first use, so it never costs startup memory or
    cold-start time. Returns None if the model or feature ranges are absent.
    """
    if state.dice_explainer is not None:
        return state.dice_explainer
    if state.joblib_model is None or not state.cf_ranges.get("ranges"):
        return None
    try:
        import dice_ml
        from dice_ml import Dice

        data = dice_ml.Data(
            features=state.cf_ranges["ranges"],
            outcome_name=state.cf_ranges.get("outcome_name", "is_fraud"),
        )
        model = dice_ml.Model(model=state.joblib_model, backend="sklearn")
        state.dice_explainer = Dice(data, model, method="random")
        return state.dice_explainer
    except Exception as exc:
        logger.warning("dice_init_failed", error=str(exc))
        return None


def _match_rules(req: TransactionRequest) -> list[dict]:
    matched = []
    for rule in state.fraud_rules:
        ante = rule.get("antecedent", [])
        hit = any(
            (item == f"category={req.category}")
            or (item == "hour=night" and req.is_night)
            or (item == f"merchant={req.merchant}")
            for item in ante
        )
        if hit:
            matched.append(rule)
        if len(matched) >= 3:
            break
    return matched


def _shap_reasons(features: np.ndarray, vel: dict, score: float) -> list[str]:
    reasons: list[str] = []
    if state.shap_explainer is not None:
        try:
            shap_vals = state.shap_explainer.shap_values(features)
            arr = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            top_idx = np.argsort(np.abs(arr))[::-1][:5]
            for i in top_idx:
                col = state.feature_cols[i] if i < len(state.feature_cols) else f"feature_{i}"
                val = features[0][i]
                reasons.append(f"{col}={val:.2f} (SHAP impact: {arr[i]:+.3f})")
            return reasons
        except Exception:
            pass

    # Fallback: rule-based reasons from feature values
    cols = state.feature_cols

    def fval(name):
        idx = cols.index(name) if name in cols else -1
        return float(features[0][idx]) if idx >= 0 else 0.0

    vel_1min = vel.get("vel_card_1min_count", 0)
    if vel_1min > 2:
        reasons.append(f"High transaction velocity on this card (1-min count: {vel_1min})")
    if fval("amt") > 1000:
        reasons.append(f"Unusually high transaction amount (${fval('amt'):.2f})")
    if fval("geo_distance_km") > 500:
        reasons.append(f"Large geographic distance ({fval('geo_distance_km'):.0f} km from home)")
    if fval("is_night") == 1:
        reasons.append("Transaction occurred during nighttime hours")
    ip_count = int(vel.get("vel_ip_prefix_1min_count", 0))
    if ip_count > 3:
        reasons.append(f"IP prefix shared with {ip_count} recent transactions")
    if not reasons:
        reasons.append(f"Model score: {score:.2%} fraud probability")
    return reasons[:5]


# ---------------------------------------------------------------------------
# Live feed — best-effort publish of each decision to a Supabase table the
# frontend subscribes to via Realtime. Fire-and-forget so it never adds latency
# to /score, and a no-op unless Supabase is configured and the feed is enabled.
# ---------------------------------------------------------------------------
async def _publish_decision(req: "TransactionRequest", trans_id: str, decision: str,
                            score: float, layer: str) -> None:
    if not (settings.live_feed_enabled and settings.supabase_url and settings.supabase_key):
        return
    record = {
        "trans_id": trans_id,
        "decision": decision,
        "fraud_score": round(float(score), 4),
        "amount": req.amt,
        "merchant": req.merchant,
        "category": req.category,
        "hour": req.hour,
        "layer": layer,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        import httpx

        url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{settings.live_feed_table}"
        headers = {
            "apikey": settings.supabase_key,
            "Authorization": f"Bearer {settings.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            await client.post(url, json=record, headers=headers)
    except Exception as exc:  # noqa: BLE001 — never let the feed affect scoring
        logger.warning("live_feed_publish_failed", error=str(exc))


def _emit_live(req: "TransactionRequest", trans_id: str, decision: str,
               score: float, layer: str) -> None:
    """Schedule a fire-and-forget publish without blocking the response."""
    try:
        asyncio.create_task(_publish_decision(req, trans_id, decision, score, layer))
    except RuntimeError:
        pass  # no running loop (e.g. sync test context) — skip silently


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/score", response_model=ScoreResponse)
async def score_transaction(req: TransactionRequest, request: Request):
    t_start = time.perf_counter()
    trace_id = str(uuid.uuid4())
    ts = req.timestamp or time.time()
    trans_id = req.trans_id or trace_id

    log = logger.bind(trace_id=trace_id, trans_id=trans_id) if hasattr(logger, "bind") else logger

    # ---- Layer 1a: blocklist ----
    if req.cc_num in state.known_fraud_cards:
        elapsed = (time.perf_counter() - t_start) * 1000
        state.total_scored += 1
        state.decline_count += 1
        state.latency_history.append(elapsed)
        _emit_live(req, trans_id, "DECLINE", 1.0, "rules")
        return ScoreResponse(
            trans_id=trans_id,
            decision="DECLINE",
            fraud_score=1.0,
            layer_triggered="rules",
            reasons=["Card is on the fraud blocklist"],
            latency_ms=round(elapsed, 2),
            model_latency_ms=0.0,
            triggered_rules=[],
            **_conformal_fields(1.0),
        )

    # ---- Velocity: record transaction then fetch canonical features ----
    state.velocity_store.record_transaction(
        card_id=req.cc_num,
        device_id=req.device_id,
        ip_prefix=req.ip_prefix,
        merchant=req.merchant,
        amount=req.amt,
        timestamp=ts,
        trans_id=trans_id,
    )
    vel = state.velocity_store.get_velocity_features(
        card_id=req.cc_num,
        device_id=req.device_id,
        ip_prefix=req.ip_prefix,
        merchant=req.merchant,
        now=ts,
    )

    # ---- Layer 1b: velocity hard cap ----
    if vel.get("vel_card_1min_count", 0) > 5:
        elapsed = (time.perf_counter() - t_start) * 1000
        state.total_scored += 1
        state.decline_count += 1
        state.latency_history.append(elapsed)
        _emit_live(req, trans_id, "DECLINE", 1.0, "rules")
        return ScoreResponse(
            trans_id=trans_id,
            decision="DECLINE",
            fraud_score=1.0,
            layer_triggered="rules",
            reasons=[f"Velocity limit exceeded: {vel['vel_card_1min_count']} txns in last 60s"],
            latency_ms=round(elapsed, 2),
            model_latency_ms=0.0,
            triggered_rules=[],
            **_conformal_fields(1.0),
        )

    # ---- Layer 2: ML scoring ----
    features = _build_feature_vector(req, vel)
    fraud_score, model_latency_ms = _run_model(features)
    state.model_latency_history.append(model_latency_ms)

    # ---- Threshold -> decision ----
    if fraud_score >= settings.fraud_threshold_decline:
        decision = "DECLINE"
        state.decline_count += 1
    elif fraud_score >= settings.fraud_threshold_review:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    # ---- Layer 3: SHAP explanations ----
    reasons = _shap_reasons(features, vel, fraud_score)

    # ---- Matched FP-Growth rules ----
    triggered_rules = _match_rules(req)

    elapsed = (time.perf_counter() - t_start) * 1000
    state.latency_history.append(elapsed)
    state.total_scored += 1

    resp = ScoreResponse(
        trans_id=trans_id,
        decision=decision,
        fraud_score=round(fraud_score, 4),
        layer_triggered="model",
        reasons=reasons,
        latency_ms=round(elapsed, 2),
        model_latency_ms=round(model_latency_ms, 2),
        triggered_rules=triggered_rules,
        **_conformal_fields(fraud_score),
    )
    _emit_live(req, trans_id, decision, fraud_score, "model")
    if hasattr(log, "info"):
        log.info("scored", decision=decision, score=fraud_score, latency_ms=elapsed)
    return resp


@app.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual(req: TransactionRequest):
    """DICE counterfactuals: minimal changes to actionable features (amount,
    geo distance, hour) that flip the model's prediction."""
    trans_id = req.trans_id or str(uuid.uuid4())
    ts = req.timestamp or time.time()

    # Read-only velocity (do NOT record — this is a hypothetical query).
    vel = {}
    if state.velocity_store is not None:
        vel = state.velocity_store.get_velocity_features(
            card_id=req.cc_num, device_id=req.device_id, ip_prefix=req.ip_prefix,
            merchant=req.merchant, now=ts,
        )

    features = _build_feature_vector(req, vel)
    score, _ = _run_model(features)
    if score >= settings.fraud_threshold_decline:
        decision = "DECLINE"
    elif score >= settings.fraud_threshold_review:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    base = dict(trans_id=trans_id, original_decision=decision,
                original_fraud_score=round(score, 4))

    explainer = _get_dice_explainer()
    if explainer is None:
        return CounterfactualResponse(
            **base, counterfactuals=[], available=False,
            message="Counterfactuals unavailable (model or feature ranges not loaded).",
        )

    vary = state.cf_ranges.get("vary_features", ["amt", "geo_distance_km", "hour"])
    outcome = state.cf_ranges.get("outcome_name", "is_fraud")
    query_df = _cf_query_df(req, vel)
    try:
        cf = explainer.generate_counterfactuals(
            query_df, total_CFs=3, desired_class="opposite", features_to_vary=vary,
        )
        cfs_df = cf.cf_examples_list[0].final_cfs_df
    except Exception as exc:
        return CounterfactualResponse(
            **base, counterfactuals=[], available=False,
            message=f"No counterfactuals found: {exc}",
        )

    orig = {f: float(query_df[f].iloc[0]) for f in vary}
    results: list[Counterfactual] = []
    for _, r in cfs_df.iterrows():
        changes = [
            CounterfactualChange(
                feature=f, original=round(orig[f], 2), suggested=round(float(r[f]), 2)
            )
            for f in vary
            if abs(float(r[f]) - orig[f]) > 1e-9
        ]
        cls = int(r[outcome]) if outcome in cfs_df.columns else 0
        results.append(
            Counterfactual(changes=changes, resulting_class="fraud" if cls == 1 else "legit")
        )

    return CounterfactualResponse(**base, counterfactuals=results, available=True)


@app.get("/health")
async def health():
    redis_connected = False
    if state.velocity_store is not None and state.velocity_store.use_redis:
        try:
            state.velocity_store.redis.ping()
            redis_connected = True
        except Exception:
            redis_connected = False

    redis_mode = "upstash" if "upstash.io" in settings.redis_url else "local"

    return {
        "status": "ok",
        "model_loaded": state.onnx_session is not None or state.joblib_model is not None,
        "onnx_available": state.onnx_available,
        "redis_available": state.redis_available,
        "redis_connected": redis_connected,
        "redis_mode": redis_mode,
        "blocklist_size": len(state.known_fraud_cards),
        "rules_loaded": len(state.fraud_rules),
        "feature_cols": len(state.feature_cols),
    }


@app.get("/metrics")
async def metrics():
    lat = list(state.latency_history)
    ml_lat = list(state.model_latency_history)
    decline_rate = (state.decline_count / state.total_scored) if state.total_scored else 0.0
    return {
        "total_scored": state.total_scored,
        "decline_rate": round(decline_rate, 4),
        "avg_score": 0.0,
        "latency_p50_ms": round(_percentile(lat, 50), 2),
        "latency_p95_ms": round(_percentile(lat, 95), 2),
        "latency_p99_ms": round(_percentile(lat, 99), 2),
        "onnx_latency_p50_ms": round(_percentile(ml_lat, 50), 2),
        "onnx_latency_p95_ms": round(_percentile(ml_lat, 95), 2),
        "onnx_latency_p99_ms": round(_percentile(ml_lat, 99), 2),
    }


@app.get("/fraud-rings")
async def fraud_rings():
    return _load_json(os.path.join(MODELS_DIR, "ring_stats.json"), default={"rings": []})


@app.get("/fraud-rules")
async def fraud_rules_endpoint():
    return {"rules": state.fraud_rules}


@app.get("/drift")
async def drift():
    return _load_json(os.path.join(MODELS_DIR, "drift_by_month.json"), default={"months": []})


@app.get("/entity-graph")
async def entity_graph():
    return _load_json(
        os.path.join(MODELS_DIR, "entity_graph.json"),
        default={"nodes": [], "links": []},
    )


@app.get("/graph/elliptic")
async def elliptic_graph():
    """Served predictions from the offline Elliptic GNN (GAT): test metrics,
    per-time-step illicit counts, and a sampled high-risk subgraph. Returns
    empty defaults until models/elliptic_graph.json is published to HF Hub
    (generated by `python -m src.graph_fraud.export_predictions`)."""
    return _load_json(
        os.path.join(MODELS_DIR, "elliptic_graph.json"),
        default={"model": "GAT", "metrics": {}, "graph_stats": {},
                 "timeline": [], "graph": {"nodes": [], "links": []}},
    )


def _feature_importance_list() -> list[dict]:
    cols = state.feature_cols or []
    if state.joblib_model is not None and hasattr(state.joblib_model, "feature_importances_"):
        imp = state.joblib_model.feature_importances_
        return [{"name": c, "importance": float(v)} for c, v in zip(cols, imp)]
    defaults = {
        "amt": 0.28,
        "geo_distance_km": 0.18,
        "vel_card_1min_count": 0.16,
        "hour": 0.09,
        "age": 0.08,
        "city_pop": 0.07,
        "vel_card_1hr_count": 0.06,
        "vel_ip_prefix_1min_count": 0.05,
        "is_night": 0.02,
        "is_weekend": 0.01,
    }
    return [{"name": c, "importance": defaults.get(c, 0.01)} for c in cols]


@app.get("/feature-importance")
async def feature_importance():
    return {"features": _feature_importance_list()}


# ---------------------------------------------------------------------------
# LLM copilot (BYOK) — provider/model/key arrive per-request via X-LLM-* headers
# and are never stored or logged. The server only relays the call.
# ---------------------------------------------------------------------------
def _llm_creds(request: Request) -> tuple[str, str, str]:
    return (
        request.headers.get("X-LLM-Provider", "").lower().strip(),
        request.headers.get("X-LLM-Model", "").strip(),
        request.headers.get("X-LLM-Key", "").strip(),
    )


def _metrics_snapshot() -> dict:
    lat = list(state.latency_history)
    decline_rate = (state.decline_count / state.total_scored) if state.total_scored else 0.0
    return {
        "total_scored": state.total_scored,
        "decline_rate": round(decline_rate, 4),
        "latency_p95_ms": round(_percentile(lat, 95), 2),
    }


@app.get("/llm/providers")
async def get_llm_providers():
    """Provider/model catalog for the frontend Settings dropdowns (no secrets)."""
    return llm_providers.public_registry()


@app.post("/llm/copilot")
async def llm_copilot(body: CopilotRequest, request: Request):
    provider, model, key = _llm_creds(request)
    rings = _load_json(os.path.join(MODELS_DIR, "ring_stats.json"), default=[])
    if isinstance(rings, dict):
        rings = rings.get("rings", [])
    context = llm_tasks.build_context(
        question=body.question,
        metrics=_metrics_snapshot(),
        rules=state.fraud_rules,
        rings=rings or [],
        feature_importance=_feature_importance_list(),
        blocklist_size=len(state.known_fraud_cards),
        thresholds={
            "review": settings.fraud_threshold_review,
            "decline": settings.fraud_threshold_decline,
        },
    )
    messages = llm_tasks.copilot_messages(context, body.question)
    try:
        answer = await llm_providers.chat(provider, model, key, messages, max_tokens=700)
    except llm_providers.LLMError as exc:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})
    return {"answer": answer, "grounded_on": {
        "rules": len(context["top_rules"]),
        "rings": len(context["fraud_rings"]),
        "features": len(context["top_features"]),
    }}


@app.post("/llm/case-report")
async def llm_case_report(body: CaseReportRequest, request: Request):
    provider, model, key = _llm_creds(request)
    ring = body.ring
    if ring is None:
        rings = _load_json(os.path.join(MODELS_DIR, "ring_stats.json"), default=[])
        if isinstance(rings, dict):
            rings = rings.get("rings", [])
        idx = body.ring_id or 0
        if not rings or idx < 0 or idx >= len(rings):
            return JSONResponse(status_code=404, content={"detail": "Fraud ring not found."})
        ring = rings[idx]
    messages = llm_tasks.case_report_messages(ring)
    try:
        report = await llm_providers.chat(provider, model, key, messages, max_tokens=600)
    except llm_providers.LLMError as exc:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})
    return {"report": report}


@app.post("/llm/rule-from-text")
async def llm_rule_from_text(body: RuleFromTextRequest, request: Request):
    provider, model, key = _llm_creds(request)
    messages = llm_tasks.rule_messages(body.text)
    try:
        content = await llm_providers.chat(
            provider, model, key, messages, temperature=0.0, max_tokens=400, json_mode=True
        )
    except llm_providers.LLMError as exc:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})
    try:
        rule = llm_tasks.parse_rule(content)
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Could not parse a rule from the response: {exc}"},
        )
    return {"rule": rule, "raw": content}


# ---------------------------------------------------------------------------
# Analyst feedback loop — ✓/✗ labels on REVIEW/DECLINE decisions feed active
# learning. Always appended to a local JSONL; best-effort mirrored to Supabase.
# ---------------------------------------------------------------------------
async def _store_feedback(record: dict) -> dict:
    sinks = {"jsonl": False, "supabase": False}
    # Local JSONL (always-on sink)
    try:
        path = settings.feedback_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        sinks["jsonl"] = True
    except Exception as exc:
        logger.warning("feedback_jsonl_failed", error=str(exc))
    # Best-effort Supabase REST insert (no SDK; plain HTTPS)
    if settings.supabase_url and settings.supabase_key:
        try:
            import httpx

            url = f"{settings.supabase_url.rstrip('/')}/rest/v1/analyst_feedback"
            headers = {
                "apikey": settings.supabase_key,
                "Authorization": f"Bearer {settings.supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(url, json=record, headers=headers)
            sinks["supabase"] = r.status_code < 300
        except Exception as exc:
            logger.warning("feedback_supabase_failed", error=str(exc))
    return sinks


@app.post("/feedback")
async def submit_feedback(body: FeedbackRequest):
    label = body.label.lower().strip()
    if label not in {"fraud", "legit"}:
        return JSONResponse(
            status_code=422, content={"detail": "label must be 'fraud' or 'legit'."}
        )
    record = {
        "trans_id": body.trans_id,
        "decision": body.decision,
        "fraud_score": body.fraud_score,
        "label": label,
        "note": body.note,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": settings.model_version,
    }
    sinks = await _store_feedback(record)
    return {"ok": sinks["jsonl"] or sinks["supabase"], "stored": sinks}
