"""Shared pytest fixtures.

The TestClient context manager triggers the FastAPI lifespan (model load,
velocity store init), so these are true integration tests against the app.
They are written to pass whether the real model is loaded or the app is in
demo mode, so CI does not need to download artifacts from HF Hub.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
