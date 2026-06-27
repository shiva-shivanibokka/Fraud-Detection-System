"""
One-time / on-retrain upload of model artifacts to the Hugging Face Hub.

Run this LOCALLY (not on Render). It pushes everything in the models/
directory to your HF Hub model repo, which the deployed API then pulls
from at build/startup time (see src/download_models.py).

Prerequisites
-------------
1. A free Hugging Face account: https://huggingface.co/join
2. An access token with WRITE scope:
   https://huggingface.co/settings/tokens  ->  "New token" -> Write
3. Set environment variables before running:
       HF_REPO_ID   e.g. your-username/fraud-detection-model
       HF_TOKEN     the write token from step 2
       MODEL_DIR    optional, defaults to "models"

Usage (PowerShell)
------------------
    $env:HF_REPO_ID = "your-username/fraud-detection-model"
    $env:HF_TOKEN   = "hf_xxxxxxxxxxxxxxxxxxxx"
    conda run -n fraud-detection --no-capture-output python scripts/upload_models_to_hf.py
"""

import os
import socket
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def _prefer_ipv4() -> None:
    """Force IPv4 for outbound connections.

    On networks with a broken IPv6 path to the HF CDN, Python's httpx/httpcore
    hangs because (unlike curl) it does not do Happy Eyeballs — it tries the
    IPv6 address first and never falls back. Pinning getaddrinfo to AF_INET
    avoids the hang.
    """
    _orig = socket.getaddrinfo

    def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _orig(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _ipv4_only

REPO_ID = os.environ.get("HF_REPO_ID", "").strip()
MODEL_DIR = os.environ.get("MODEL_DIR", "models").strip()
TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
# Make the model repo public by default (portfolio); set HF_PRIVATE=1 to keep it private.
PRIVATE = os.environ.get("HF_PRIVATE", "").strip() in {"1", "true", "True"}


def main() -> int:
    _prefer_ipv4()
    if not REPO_ID:
        print("ERROR: HF_REPO_ID is not set. Example: your-username/fraud-detection-model")
        return 1
    if not TOKEN:
        print("ERROR: HF_TOKEN is not set. Create a WRITE token at "
              "https://huggingface.co/settings/tokens")
        return 1

    model_path = Path(MODEL_DIR)
    if not model_path.is_dir():
        print(f"ERROR: model directory '{MODEL_DIR}' not found (run from repo root).")
        return 1

    files = sorted(p.name for p in model_path.iterdir() if p.is_file())
    if not files:
        print(f"ERROR: no files found in '{MODEL_DIR}'.")
        return 1

    print(f"Uploading {len(files)} artifact(s) to HF Hub repo '{REPO_ID}' "
          f"({'private' if PRIVATE else 'public'}):")
    for f in files:
        print(f"  - {f}")

    create_repo(REPO_ID, repo_type="model", exist_ok=True, private=PRIVATE, token=TOKEN)

    api = HfApi()
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        token=TOKEN,
        commit_message="Upload fraud detection model artifacts",
        ignore_patterns=["*.pyc", "__pycache__/*"],
    )

    print(f"\nDone. View at: https://huggingface.co/{REPO_ID}/tree/main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
