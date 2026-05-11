"""
Fraud Detection API - 3-layer decision architecture
Layer 1: Hard rules engine  (<1ms)
Layer 2: ML model scoring   (<20ms) via ONNX / joblib fallback
Layer 3: SHAP explanation generation
"""

import json
import os
import sys
import time
import uuid
import collections
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("fraud_api")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ---------------------------------------------------------------------------
# Velocity store (Redis if available, else in-memory)
# ---------------------------------------------------------------------------
class InMemoryVelocityStore:
    """Sliding-window counter backed by a deque per key."""

    def __init__(self):
        self._store: dict[str, collections.deque] = collections.defaultdict(
            collections.deque
        )

    def incr_window(self, key: str, window_seconds: int, ts: float) -> int:
        dq = self._store[key]
        cutoff = ts - window_seconds
        while dq and dq[0] < cutoff:
            dq.popleft()
        dq.append(ts)
        return len(dq)

    def get_count(self, key: str, window_seconds: int, ts: float) -> int:
        dq = self._store[key]
        cutoff = ts - window_seconds
        return sum(1 for t in dq if t >= cutoff)


class VelocityFeatureStore:
    def __init__(self):
        self.redis = None
        self._mem = InMemoryVelocityStore()
        self._use_redis = False
        try:
            import redis as _redis

            r = _redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
            r.ping()
            self.redis = r
            self._use_redis = True
            logger.info("velocity_store", backend="redis")
        except Exception:
            logger.info("velocity_store", backend="in_memory")

    def record_and_count(
        self, cc_num: str, ip_prefix: str, ts: float
    ) -> dict[str, int]:
        if self._use_redis:
            pipe = self.redis.pipeline()
            for key, ttl in [
                (f"vel:card:{cc_num}", 60),
                (f"vel:card:{cc_num}", 3600),
                (f"vel:ip:{ip_prefix}", 60),
            ]:
                pipe.zadd(key, {str(ts): ts})
                pipe.zremrangebyscore(key, "-inf", ts - ttl)
                pipe.zcard(key)
                pipe.expire(key, ttl)
            results = pipe.execute()
            return {
                "vel_card_1min": results[2],
                "vel_card_1hr": results[6],
                "vel_ip_1min": results[10],
            }
        else:
            return {
                "vel_card_1min": self._mem.incr_window(f"card:{cc_num}", 60, ts),
                "vel_card_1hr": self._mem.incr_window(f"card_hr:{cc_num}", 3600, ts),
                "vel_ip_1min": self._mem.incr_window(f"ip:{ip_prefix}", 60, ts),
            }


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


state = AppState()


def _load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("json_load_failed", path=path, error=str(exc))
        return default if default is not None else {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Load ONNX model ----
    try:
        import onnxruntime as ort

        onnx_path = os.path.join(MODELS_DIR, "fraud_model.onnx")
        state.onnx_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        state.onnx_available = True
        logger.info("model_loaded", backend="onnx")
    except Exception as exc:
        logger.warning("onnx_load_failed", error=str(exc))
        try:
            import joblib

            state.joblib_model = joblib.load(
                os.path.join(MODELS_DIR, "fraud_model.pkl")
            )
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
            "vel_card_1min",
            "vel_card_1hr",
            "vel_ip_1min",
        ]

    # ---- Load card embeddings ----
    try:
        import joblib

        state.card_embeddings = (
            joblib.load(os.path.join(MODELS_DIR, "card_embeddings.pkl")) or {}
        )
    except Exception:
        state.card_embeddings = {}

    # ---- Load FP-Growth rules ----
    state.fraud_rules = _load_json(
        os.path.join(MODELS_DIR, "fraud_rules.json"), default=[]
    )

    # ---- Build blocklist from ring stats ----
    ring_stats = _load_json(os.path.join(MODELS_DIR, "ring_stats.json"), default={})
    for ring in ring_stats.get("rings") or []:
        for card in ring.get("cards") or []:
            state.known_fraud_cards.add(str(card))

    # ---- Velocity store ----
    state.velocity_store = VelocityFeatureStore()
    state.redis_available = state.velocity_store._use_redis

    # ---- SHAP explainer (best-effort background init) ----
    try:
        import shap

        model = state.joblib_model
        if model is not None and hasattr(model, "predict_proba"):
            bg = np.zeros((10, len(state.feature_cols)))
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
    allow_origins=["*"],
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    arr = sorted(data)
    idx = int(len(arr) * pct / 100)
    return arr[min(idx, len(arr) - 1)]


def _build_feature_vector(req: TransactionRequest, vel: dict[str, int]) -> np.ndarray:
    raw = {
        "amt": req.amt,
        "hour": req.hour,
        "day_of_week": req.day_of_week,
        "is_weekend": req.is_weekend,
        "is_night": req.is_night,
        "age": req.age,
        "geo_distance_km": req.geo_distance_km,
        "city_pop": req.city_pop,
        "vel_card_1min": vel.get("vel_card_1min", 0),
        "vel_card_1hr": vel.get("vel_card_1hr", 0),
        "vel_ip_1min": vel.get("vel_ip_1min", 0),
    }
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


def _match_rules(req: TransactionRequest) -> list[dict]:
    matched = []
    for rule in state.fraud_rules:
        ante = rule.get("antecedent", [])
        hit = any(
            (item == f"category={req.category}")
            or (item == f"hour=night" and req.is_night)
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
                col = (
                    state.feature_cols[i]
                    if i < len(state.feature_cols)
                    else f"feature_{i}"
                )
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

    vel_1min = vel.get("vel_card_1min", 0)
    if vel_1min > 2:
        reasons.append(
            f"High transaction velocity on this card (1-min count: {vel_1min})"
        )
    if fval("amt") > 1000:
        reasons.append(f"Unusually high transaction amount (${fval('amt'):.2f})")
    if fval("geo_distance_km") > 500:
        reasons.append(
            f"Large geographic distance ({fval('geo_distance_km'):.0f} km from home)"
        )
    if fval("is_night") == 1:
        reasons.append("Transaction occurred during nighttime hours")
    if fval("vel_ip_1min") > 3:
        ip_count = int(vel.get("vel_ip_1min", 0))
        reasons.append(f"IP prefix shared with {ip_count} recent transactions")
    if not reasons:
        reasons.append(f"Model score: {score:.2%} fraud probability")
    return reasons[:5]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/score", response_model=ScoreResponse)
async def score_transaction(req: TransactionRequest, request: Request):
    t_start = time.perf_counter()
    trace_id = str(uuid.uuid4())
    ts = req.timestamp or time.time()
    trans_id = req.trans_id or trace_id

    log = (
        logger.bind(trace_id=trace_id, trans_id=trans_id)
        if hasattr(logger, "bind")
        else logger
    )

    # ---- Layer 1a: blocklist ----
    if req.cc_num in state.known_fraud_cards:
        elapsed = (time.perf_counter() - t_start) * 1000
        state.total_scored += 1
        state.decline_count += 1
        state.latency_history.append(elapsed)
        return ScoreResponse(
            trans_id=trans_id,
            decision="DECLINE",
            fraud_score=1.0,
            layer_triggered="rules",
            reasons=["Card is on the fraud blocklist"],
            latency_ms=round(elapsed, 2),
            model_latency_ms=0.0,
            triggered_rules=[],
        )

    # ---- Velocity: record + fetch counts ----
    vel = state.velocity_store.record_and_count(req.cc_num, req.ip_prefix, ts)

    # ---- Layer 1b: velocity hard cap ----
    if vel.get("vel_card_1min", 0) > 5:
        elapsed = (time.perf_counter() - t_start) * 1000
        state.total_scored += 1
        state.decline_count += 1
        state.latency_history.append(elapsed)
        return ScoreResponse(
            trans_id=trans_id,
            decision="DECLINE",
            fraud_score=1.0,
            layer_triggered="rules",
            reasons=[
                f"Velocity limit exceeded: {vel['vel_card_1min']} txns in last 60s"
            ],
            latency_ms=round(elapsed, 2),
            model_latency_ms=0.0,
            triggered_rules=[],
        )

    # ---- Layer 2: ML scoring ----
    features = _build_feature_vector(req, vel)
    fraud_score, model_latency_ms = _run_model(features)
    state.model_latency_history.append(model_latency_ms)

    # ---- Threshold -> decision ----
    if fraud_score >= 0.8:
        decision = "DECLINE"
        state.decline_count += 1
    elif fraud_score >= 0.4:
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
    )
    if hasattr(log, "info"):
        log.info("scored", decision=decision, score=fraud_score, latency_ms=elapsed)
    return resp


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.onnx_session is not None
        or state.joblib_model is not None,
        "onnx_available": state.onnx_available,
        "redis_available": state.redis_available,
        "blocklist_size": len(state.known_fraud_cards),
        "rules_loaded": len(state.fraud_rules),
        "feature_cols": len(state.feature_cols),
    }


@app.get("/metrics")
async def metrics():
    lat = list(state.latency_history)
    ml_lat = list(state.model_latency_history)
    decline_rate = (
        (state.decline_count / state.total_scored) if state.total_scored else 0.0
    )
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
    return _load_json(
        os.path.join(MODELS_DIR, "ring_stats.json"), default={"rings": []}
    )


@app.get("/fraud-rules")
async def fraud_rules_endpoint():
    return {"rules": state.fraud_rules}


@app.get("/drift")
async def drift():
    return _load_json(
        os.path.join(MODELS_DIR, "drift_by_month.json"), default={"months": []}
    )


@app.get("/entity-graph")
async def entity_graph():
    return _load_json(
        os.path.join(MODELS_DIR, "entity_graph.json"),
        default={"nodes": [], "links": []},
    )


@app.get("/feature-importance")
async def feature_importance():
    cols = state.feature_cols or []
    if state.joblib_model is not None and hasattr(
        state.joblib_model, "feature_importances_"
    ):
        imp = state.joblib_model.feature_importances_
        return {
            "features": [{"name": c, "importance": float(v)} for c, v in zip(cols, imp)]
        }
    # Heuristic fallback
    defaults = {
        "amt": 0.28,
        "geo_distance_km": 0.18,
        "vel_card_1min": 0.16,
        "hour": 0.09,
        "age": 0.08,
        "city_pop": 0.07,
        "vel_card_1hr": 0.06,
        "vel_ip_1min": 0.05,
        "is_night": 0.02,
        "is_weekend": 0.01,
    }
    return {
        "features": [{"name": c, "importance": defaults.get(c, 0.01)} for c in cols]
    }
