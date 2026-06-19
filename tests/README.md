# Test Suite (P3 — QA/DevOps)

Runs fully offline in **mock mode** (no `GEMINI_API_KEY` needed).

## Run

```bash
pip install -r requirements.txt
pytest tests/ -v
# with coverage:
pytest tests/ -v --cov=backend --cov-report=term-missing
```

## Files

| File | Covers |
|------|--------|
| `conftest.py` | Shared fixtures: mock `LLMService`, `BenchmarkService`, sample queries |
| `test_schemas.py` | Pydantic models, `QualityScore.average`, `WorkflowRun` properties, `BenchmarkResult` fields |
| `test_llm_service.py` | Backend selection, mock determinism, token estimation, cost math |
| `test_agents.py` | Leaf agents run; `EvaluationAgent` scoring + parse fallback |
| `test_metrics_collector.py` | Accumulation, totals, max context, cost, estimated flag |
| `test_workflows.py` | Traditional (3 calls) + 3 Recursive rounds (2 calls each), context size |
| `test_benchmark_service.py` | Orchestration, speedup formula, summary generation |
| `test_api.py` | FastAPI endpoints via `TestClient` (`/`, `/health`, `/config`, `/benchmark`) |
