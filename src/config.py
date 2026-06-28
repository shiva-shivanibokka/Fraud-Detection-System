"""
Application configuration via pydantic-settings.

All settings can be overridden by environment variables (case-insensitive).
Create a .env file in the project root for local development.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Redis (velocity feature store)
    redis_url: str = "redis://localhost:6379"

    # Supabase (operational DB + pgvector)
    supabase_url: str = ""
    supabase_key: str = ""

    # CORS — restrict in production to your actual frontend origin(s)
    cors_origins: list[str] = ["http://localhost:5173"]

    # Model artifacts directory (relative to repo root or absolute)
    model_dir: str = "models"

    # Fraud decision thresholds
    fraud_threshold_review: float = 0.4
    fraud_threshold_decline: float = 0.8

    # Hugging Face Hub (model registry)
    hf_repo_id: str = ""

    # LLM copilot backend: "openai" or "groq". Keys are BYOK (per-request
    # X-LLM-Key header from the browser) — never read from the server env.
    llm_provider: str = "openai"

    # Analyst feedback loop: where ✓/✗ labels land. JSONL is the always-on local
    # sink; if supabase_url/key are set, each label is also best-effort inserted
    # into the Supabase "feedback" table for the retrain trigger.
    feedback_path: str = "data/feedback.jsonl"

    # Deployed model version tag (used in /decisions records)
    model_version: str = "1.0.0"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Suppress pydantic's default protection of the "model_" namespace so
        # that fields named model_dir and model_version are allowed without warnings.
        "protected_namespaces": (),
    }


settings = Settings()
