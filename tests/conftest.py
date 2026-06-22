"""Shared pytest fixtures.

OWNER: P3 (QA/DevOps)

Provides a deterministic, offline LLMService (mock backend, no latency sleep)
and reusable service instances so the whole suite runs without a GEMINI_API_KEY.
"""
import os
import sys

import pytest

# Make the project root importable (so `backend.*` resolves) when pytest is run
# from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.llm_service import LLMService          # noqa: E402
from backend.benchmarks.benchmark_service import BenchmarkService  # noqa: E402


@pytest.fixture
def llm():
    """Deterministic mock LLM (no API key, no real latency)."""
    # Ensure no key leaks in from the environment during tests.
    os.environ.pop("GEMINI_API_KEY", None)
    return LLMService(simulate_latency=False)


@pytest.fixture
def benchmark_service(llm):
    return BenchmarkService(llm)


@pytest.fixture
def sample_query():
    return "Payment was deducted twice but my order is not confirmed. Please fix urgently."


@pytest.fixture
def sample_queries():
    return [
        "How do I reset my password?",
        "The app keeps crashing on Android 14.",
        "I want a refund for a damaged item.",
    ]
