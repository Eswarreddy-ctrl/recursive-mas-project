"""Schema tests — Pydantic models and computed properties.

OWNER: P3 (QA/DevOps)
"""
import pytest

from backend.models.schemas import (
    BenchmarkResult,
    QualityScore,
    QueryInput,
    StageMetrics,
    WorkflowRun,
)


def test_query_input_requires_text():
    qi = QueryInput(query="hello")
    assert qi.query == "hello"


def test_quality_score_average():
    q = QualityScore(correctness=8, completeness=6, relevance=10, tone=8)
    # (8 + 6 + 10 + 8) / 4 = 8.0
    assert q.average == 8.0


def test_quality_score_defaults_to_zero():
    q = QualityScore()
    assert q.average == 0.0


def test_quality_score_bounds_enforced():
    with pytest.raises(Exception):
        QualityScore(correctness=11)  # > 10 should fail validation
    with pytest.raises(Exception):
        QualityScore(tone=-1)  # < 0 should fail validation


def test_quality_dump_with_average_includes_average():
    q = QualityScore(correctness=7, completeness=7, relevance=7, tone=7)
    d = q.model_dump_with_average()
    assert d["average"] == 7.0
    assert set(d) >= {"correctness", "completeness", "relevance", "tone", "average"}


def test_workflow_run_duration_seconds_property():
    wr = WorkflowRun(label="X", duration_ms=2500.0)
    assert wr.duration_s == 2.5


def test_workflow_run_quality_avg_property():
    wr = WorkflowRun(label="X", quality=QualityScore(correctness=9, completeness=9,
                                                     relevance=9, tone=9))
    assert wr.quality_avg == 9.0


def test_stage_metrics_defaults():
    s = StageMetrics(name="Classify")
    assert s.total_tokens == 0
    assert s.llm_calls == 0


def test_benchmark_result_all_fields_present():
    b = BenchmarkResult(query="q")
    # 4 buckets × (time, tokens, cost, quality, calls)
    for prefix in ("traditional", "recursive_r1", "recursive_r2", "recursive_r3"):
        for suffix in ("time", "tokens", "cost", "quality", "calls"):
            assert hasattr(b, f"{prefix}_{suffix}")
    assert b.backend_mode == "mock"
