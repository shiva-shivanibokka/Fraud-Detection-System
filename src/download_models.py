"""
Pull model artifacts from the Hugging Face Hub into the local models/ dir.

Idempotent: if the artifacts are already present (e.g. baked into the
Render build slug) this is a no-op, so it is safe to call both at build
time and again at app startup ("pull at startup if not cached").

Deliberately depends only on the standard library + huggingface_hub (NOT
src.config) so it can run as a plain script in the Render build command:

    python -m src.download_models

Environment variables (same names/defaults as src/config.py):
    HF_REPO_ID   required — e.g. your-username/fraud-detection-model
    MODEL_DIR    optional — defaults to "models"
    HF_TOKEN     optional — only needed if the HF model repo is private
"""

import os
from pathlib import Path

# The file we treat as proof that the artifacts are present.
SENTINEL = "fraud_model.pkl"


def ensure_models(
    repo_id: str | None = None,
    model_dir: str | None = None,
    token: str | None = None,
) -> bool:
    """Download artifacts from HF Hub if the sentinel file is missing.

    Returns True if models are present after the call, False otherwise.
    Never raises on a missing repo_id or network error — the API falls
    back to demo mode if models cannot be obtained.
    """
    repo_id = (repo_id if repo_id is not None else os.environ.get("HF_REPO_ID", "")).strip()
    model_dir = (model_dir if model_dir is not None else os.environ.get("MODEL_DIR", "models")).strip()
    token = (token if token is not None else os.environ.get("HF_TOKEN", "")).strip() or None

    dest = Path(model_dir)
    sentinel_path = dest / SENTINEL

    if sentinel_path.is_file():
        print(f"[download_models] '{sentinel_path}' already present — skipping download.")
        return True

    if not repo_id:
        print("[download_models] HF_REPO_ID not set and no local models — "
              "API will run in demo mode.")
        return False

    try:
        from huggingface_hub import snapshot_download

        print(f"[download_models] Downloading artifacts from HF Hub repo '{repo_id}' "
              f"into '{model_dir}/' ...")
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=model_dir,
            token=token,
        )
        ok = sentinel_path.is_file()
        print("[download_models] Done." if ok
              else f"[download_models] WARNING: '{SENTINEL}' not found after download.")
        return ok
    except Exception as exc:  # noqa: BLE001 - best-effort; demo mode is the fallback
        print(f"[download_models] Download failed ({exc!r}) — API will run in demo mode.")
        return False


if __name__ == "__main__":
    ensure_models()
