# RecursiveMAS Benchmark Dashboard

A research-benchmark dashboard comparing a **Traditional (flat) Multi-Agent System**
against a **RecursiveMAS** across **three recursion rounds**, measuring inference
time, token consumption, cost, and response quality.

The headline question: *how much does recursive, minimal-context delegation speed
up inference versus a flat pipeline that re-sends full context to every agent?*

> **On the numbers:** every figure in the dashboard (speedup, % token/cost
> reduction, quality delta) is **computed from measured run data** — nothing is
> hardcoded. With no API key, the system runs in a deterministic **mock mode**
> where latency and token counts are *simulated from the real context sizes* each
> workflow builds, so architectural differences still show up honestly. Add a
> live key for real measurements.

---

## Architecture

```
Traditional MAS (flat):   Classifier → Responder → QA      (full context each stage)
RecursiveMAS (3 rounds):  R1 Classify→Respond
                          R2 Review→Improve   (sees only R1 output)
                          R3 Refine→Quality   (sees only R2 output)
```

See [`docs/architecture.md`](docs/architecture.md) for full diagrams.

```
backend/
  agents/        Classifier, Responder, QA, recursive round agents, Evaluation
  services/      llm_service (Gemini 2.0 Flash / mock), traditional_mas, recursive_mas
  benchmarks/    metrics_collector, benchmark_service
  models/        Pydantic schemas incl. BenchmarkResult
  api/           FastAPI routes
  main.py        FastAPI app
frontend/
  charts/        Plotly figures (7 charts)
  components/    speedup KPI cards, summary metrics
  app.py         Streamlit dashboard
data/
  sample_benchmarks.json   precomputed sample results (offline demo)
docs/
  architecture.md
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # optional: add your GEMINI_API_KEY for live mode
```

### Run the API

```bash
uvicorn backend.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

### Run the dashboard

```bash
streamlit run frontend/app.py
# http://localhost:8501
```

The dashboard talks to the API at `API_URL` (default `http://localhost:8000`).
If the backend is offline, it falls back to bundled sample data so it is always
demonstrable.

---

## API

### `POST /benchmark`

```json
{ "query": "Payment deducted twice but order not confirmed. Please fix urgently." }
```

Returns a `BenchmarkResponse` with the traditional run, the three recursive
rounds, a flat `BenchmarkResult`, speedup KPI cards, and an auto-generated
research summary.

Other endpoints: `GET /samples` (precomputed data), `GET /config` (active
backend), `GET /health`.

---

## Metrics collected

| Metric | Per stage | Per workflow | In summary |
|---|---|---|---|
| Duration (ms / s) | ✓ | ✓ | ✓ |
| Prompt / completion / total tokens | ✓ | ✓ | ✓ |
| LLM calls | ✓ | ✓ | |
| Estimated cost (USD) | ✓ | ✓ | ✓ |
| Context characters processed | ✓ | ✓ | |
| Quality (correctness, completeness, relevance, tone → avg) | | ✓ | ✓ |

## Dashboard charts

1. Inference Time Comparison (bar)
2. Token Consumption (bar)
3. Quality Score Improvement (line)
4. Cost Comparison (bar)
5. **Inference Time Speedup across 3 rounds** (large KPI cards)
6. RecursiveMAS Scalability Curve — time saved % vs recursion depth
7. Quality vs Latency Tradeoff (scatter)

Plus an auto-generated research summary strip.

---

## Notes on honest benchmarking

- **Mock mode is not faked-to-win.** Token counts derive from the actual prompt
  and context each agent builds. Because the recursive rounds pass smaller
  context, they genuinely register fewer tokens and lower simulated latency.
- **Quality is noisy in mock mode** (randomized per round) — by design, so you
  don't mistake mock output for a real quality finding.
- For a defensible result, run several queries with a **live API key** and
  average across them. A single run is anecdote, not evidence.

## Testing (P3 — QA/DevOps)

Runs fully offline in mock mode (no API key needed).

```bash
pytest tests/ -v
pytest tests/ -v --cov=backend --cov-report=term-missing   # with coverage
```

47 test functions / 81 assertions across 8 files (schemas, llm service, agents,
metrics collector, workflows, benchmark service, API routes). See
[`tests/README.md`](tests/README.md). CI runs them on every push via
`.github/workflows/ci.yml`.

## Deployment (P3 — QA/DevOps)

`render.yaml` defines two Render.com web services — the FastAPI API and the
Streamlit dashboard. Set `GEMINI_API_KEY` (API service) and `API_URL`
(dashboard service) in the Render dashboard; leave the key blank to deploy in
mock mode.

## License

MIT
