"""Unit tests for the HF Hub model-download helper (no network)."""

from src.download_models import ensure_models


def test_skip_when_sentinel_present(tmp_path):
    # If the sentinel artifact already exists, ensure_models is a no-op -> True.
    (tmp_path / "fraud_model.pkl").write_text("dummy")
    assert ensure_models(repo_id="anything", model_dir=str(tmp_path)) is True


def test_no_repo_and_no_files_returns_false(tmp_path):
    # No local artifacts and no HF_REPO_ID -> cannot obtain models -> False
    # (the API then falls back to demo mode rather than crashing).
    assert ensure_models(repo_id="", model_dir=str(tmp_path)) is False
