# Fraud Detection System — Claude Instructions

## Git Workflow

**All changes go directly to `main`.** No feature branches. No pushes unless explicitly requested.

- Commit directly to `main` after each logical change
- Do NOT create branches (`git checkout -b`) unless the user explicitly asks
- Do NOT `git push` unless the user explicitly asks
- Do NOT create pull requests unless the user explicitly asks

## Project Context

Production overhaul of a fraud detection ML system. Portfolio project for Shivani Bokka — public repo, free-tier infrastructure only.

**Active sprint plan:** See `memory/feature_roadmap.md` for the 5-sprint overhaul plan. Always check which sprint is next before starting work.

## Stack

- **Backend:** FastAPI + Uvicorn, Python 3.11, pydantic-settings for config
- **ML:** XGBoost + LightGBM ensemble (Optuna HPO), ONNX inference, TGAT graph neural network
- **Feature store:** `src/velocity/feature_store.py` (canonical) — use this everywhere, never inline a duplicate
- **Database:** Supabase (PostgreSQL + pgvector + Realtime)
- **Cache:** Upstash Redis (dual-mode: `redis://` local, `rediss://` prod)
- **Frontend:** React 18 + Vite + D3 + Recharts
- **Hosting:** Render.com (backend), Vercel (frontend)
- **CI/CD:** GitHub Actions
- **LLM:** OpenAI GPT-4o or Groq Llama-3.3-70B — user BYOK via browser localStorage

## Code Rules

- Config always via `src/config.py` (`settings.*`) — never hardcode URLs, ports, or thresholds
- Velocity features: always use canonical names (`vel_card_1min_count`, not `vel_card_1min`)
- Ruff must pass (`ruff check src/`) before any commit
- No duplicate classes or logic — if it exists in a module, import it
