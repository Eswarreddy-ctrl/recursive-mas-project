"""API tests — FastAPI endpoints via TestClient.

OWNER: P3 (QA/DevOps)
"""
import os

import pytest

os.environ.pop("GEMINI_API_KEY", None)

try:
    from fastapi.testclient import TestClient
    from backend.main import app  # load_dotenv() inside may re-set GEMINI_API_KEY from .env
    import backend.api.routes as _routes
    from backend.benchmarks.benchmark_service import BenchmarkService
    from backend.services.llm_service import LLMService
    # pop again after load_dotenv ran, then reset the module-level singleton
    os.environ.pop("GEMINI_API_KEY", None)
    _routes._service = BenchmarkService(LLMService(simulate_latency=False))
    _client = TestClient(app)
    _HAS_FASTAPI = True
except Exception:  # pragma: no cover
    _HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


def test_root_ok():
    r = _client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_health_reports_mock_mode():
    r = _client.get("/health")
    assert r.status_code == 200
    assert r.json()["llm_mode"] == "mock"


def test_config_reports_backend():
    r = _client.get("/config")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "mock"
    assert "model" in body


def test_benchmark_endpoint_returns_full_payload():
    r = _client.post("/benchmark", json={"query": "My card was charged twice."})
    assert r.status_code == 200
    body = r.json()
    assert len(body["recursive_rounds"]) == 3
    assert len(body["speedups"]) == 3
    assert "summary" in body


def test_benchmark_rejects_empty_query():
    r = _client.post("/benchmark", json={"query": "   "})
    assert r.status_code == 400


def test_benchmark_rejects_missing_field():
    r = _client.post("/benchmark", json={})
    assert r.status_code == 422  # pydantic validation error
