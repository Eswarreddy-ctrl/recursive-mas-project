"""FastAPI routes for the benchmark API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.benchmarks.benchmark_service import BenchmarkService
from backend.models.schemas import BenchmarkResponse, QueryInput
from backend.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter()

# Single shared service instance (LLM client is reusable).
_service = BenchmarkService(LLMService())

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_benchmarks.json"


@router.post("/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
async def run_benchmark(payload: QueryInput) -> BenchmarkResponse:
    """Run Traditional MAS + 3 Recursive rounds for a query and return metrics."""
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        return _service.run(query)
    except Exception as e:  # pragma: no cover
        logger.exception("Benchmark failed")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e}")


@router.get("/samples", tags=["Benchmark"])
async def get_samples() -> dict:
    """Return precomputed sample benchmark data (for offline dashboard demos)."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample data not found.")
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/config", tags=["Health"])
async def config() -> dict:
    """Report which LLM backend is active."""
    return {
        "backend": _service.llm.backend,
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
    }
