"""Benchmark service tests — orchestration, speedups, summary.

OWNER: P3 (QA/DevOps)
"""
from backend.models.schemas import BenchmarkResponse


def test_run_returns_full_response(benchmark_service, sample_query):
    resp = benchmark_service.run(sample_query)
    assert isinstance(resp, BenchmarkResponse)
    assert resp.query == sample_query
    assert len(resp.recursive_rounds) == 3
    assert len(resp.speedups) == 3


def test_benchmark_result_mirrors_runs(benchmark_service, sample_query):
    resp = benchmark_service.run(sample_query)
    b = resp.benchmark
    assert b.traditional_tokens == resp.traditional.total_tokens
    assert b.recursive_r1_tokens == resp.recursive_rounds[0].total_tokens
    assert b.recursive_r3_quality == resp.recursive_rounds[2].quality_avg


def test_speedup_formula(benchmark_service, sample_query):
    resp = benchmark_service.run(sample_query)
    trad_t = resp.traditional.duration_s
    for card, rnd in zip(resp.speedups, resp.recursive_rounds):
        expected = round(trad_t / max(rnd.duration_s, 1e-6), 2)
        assert abs(card.speedup - expected) < 0.02


def test_time_saved_pct_consistent(benchmark_service, sample_query):
    resp = benchmark_service.run(sample_query)
    for card in resp.speedups:
        assert -200 <= card.time_saved_pct <= 100


def test_summary_fields_computed(benchmark_service, sample_query):
    s = benchmark_service.run(sample_query).summary
    assert s.best_speedup > 0
    assert isinstance(s.narrative, str) and s.narrative
    assert "RecursiveMAS" in s.narrative


def test_backend_mode_is_mock_without_key(benchmark_service, sample_query):
    resp = benchmark_service.run(sample_query)
    assert resp.benchmark.backend_mode == "mock"
