# Architecture Decision Record

Concise record of the non-obvious engineering decisions in this project and the
reasoning behind them. Each entry: **Context → Decision → Consequences**.

---

## ADR-001 — Split serving deps from training deps

**Context.** The full stack needs torch + torch-geometric (~2–3 GB) for the GNN
research module, but the deploy target is Render's free tier (512 MB).

**Decision.** Maintain `requirements-api.txt` (lean serving: FastAPI, sklearn,
xgboost, onnxruntime, httpx, redis) separate from `requirements.txt` (full
training/research). The serving box never imports torch.

**Consequences.** Deploy fits free tier; anything torch-bound (sentence-
transformers, GNN) is offline-only by construction. This constraint recurs in
ADR-005.

---

## ADR-002 — Single calibrated XGBoost over an ensemble

**Context.** An XGBoost+LightGBM / XGBoost+CatBoost ensemble was A/B-tested
against a single isotonic-calibrated XGBoost.

**Decision.** Keep the single calibrated XGBoost; reject the ensemble.

**Consequences.** The ensemble did not beat the calibrated single model on the
temporal test split and doubled inference cost and memory. The experiment is
retained as `src/model/train_ensemble.py` for transparency, not shipped.

---

## ADR-003 — Serving-light conformal & counterfactuals

**Context.** MAPIE (conformal) and dice-ml add value but MAPIE shouldn't be a
hot-path dependency.

**Decision.** Compute the conformal LAC threshold offline, export it to
`conformal.json`, and apply it with plain arithmetic at serving. DICE is
lazy-imported only when `/counterfactual` is called.

**Consequences.** 90%-coverage uncertainty labels on every score with no MAPIE
import at request time; counterfactuals cost nothing until used.

---

## ADR-004 — EvolveGCN-O as the temporal GNN

**Context.** On Elliptic, fraud regimes shift over time (e.g. a dark-market
shutdown). Candidates: static GAT, continuous-time TGAT, snapshot-based
EvolveGCN-O.

**Decision.** Use EvolveGCN-O as the headline temporal model.

**Consequences.** TGAT's continuous-time edge attention is the wrong inductive
bias here (Elliptic edges are mostly cross-time-step money flow, not timestamped
interactions); EvolveGCN-O — which evolves the GCN weights across snapshots —
matches the regime-shift structure and wins on illicit F1. Both best-epoch and
validation-early-stopped numbers are reported.

---

## ADR-005 — BYOK LLM relay with grounded (not pgvector) retrieval

**Context.** The analyst copilot needs an LLM, but the public portfolio repo has
no budget for a hosted key, and pgvector RAG needs an embeddings model
(sentence-transformers → torch, excluded by ADR-001).

**Decision.** Bring-your-own-key: provider/model/key arrive per request via
`X-LLM-*` headers from browser localStorage and are never stored or logged; the
server only relays to OpenAI/Groq (both OpenAI-compatible) over httpx. Retrieval
grounds the model on the system's own structured fraud knowledge (rules, rings,
metrics, importances) rather than a vector DB.

**Consequences.** Zero server-side secrets and zero hosting cost; deployable on
free tier. pgvector semantic RAG is a documented future upgrade, not a claimed
current capability.

---

## ADR-006 — Fire-and-forget live feed

**Context.** The live dashboard feed needs each decision in Supabase, but a
synchronous insert would add network latency to every `/score`.

**Decision.** Publish decisions to Supabase via `asyncio.create_task`
(fire-and-forget), gated on Supabase being configured, with errors swallowed and
logged.

**Consequences.** `/score` latency is unaffected and the feed never breaks
scoring; the feed is best-effort (a dropped publish loses one feed row, never a
decision). All observability (Sentry, DagsHub, live feed) is env-gated and
no-ops cleanly when unconfigured.
