"""
tests/test_api_routes.py — FastAPI route tests.
Day: Tuesday (Week 1)  |  ~8 assertions
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def payload():
    return {"ticket_id": "T001", "text": "My invoice is wrong.", "customer_id": "C1"}


class TestHealthRoute:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_returns_ok(self):
        assert client.get("/health").json() == {"status": "ok"}


class TestAnalyzeRoute:
    def test_missing_text_422(self):
        assert client.post("/analyze", json={"ticket_id": "T1"}).status_code == 422

    def test_empty_text_422(self):
        assert client.post("/analyze", json={"ticket_id": "T1", "text": ""}).status_code == 422

    def test_missing_ticket_id_422(self):
        assert client.post("/analyze", json={"text": "Help"}).status_code == 422

    def test_valid_payload_200(self, payload, sample_comparison):
        with patch("api.routes.run_traditional", return_value=sample_comparison.traditional), \
             patch("api.routes.run_recursive",   return_value=sample_comparison.recursive):
            r = client.post("/analyze", json=payload)
        assert r.status_code == 200

    def test_response_has_both_workflows(self, payload, sample_comparison):
        with patch("api.routes.run_traditional", return_value=sample_comparison.traditional), \
             patch("api.routes.run_recursive",   return_value=sample_comparison.recursive):
            data = client.post("/analyze", json=payload).json()
        assert "traditional" in data and "recursive" in data

    def test_workflow_error_returns_500(self, payload):
        with patch("api.routes.run_traditional", side_effect=RuntimeError("boom")):
            assert client.post("/analyze", json=payload).status_code == 500