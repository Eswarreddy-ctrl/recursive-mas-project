"""Metrics collector tests — accumulation, totals, max context.

OWNER: P3 (QA/DevOps)
"""
from backend.benchmarks.metrics_collector import MetricsCollector
from backend.models.schemas import QualityScore


def _fake_result(tokens=100, prompt=60, completion=40, dur=200.0, estimated=True):
    return {
        "text": "x",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": tokens,
        "duration_ms": dur,
        "estimated": estimated,
    }


def test_record_accumulates_totals(llm):
    mc = MetricsCollector("Test", llm)
    mc.record("StageA", _fake_result(tokens=100), context_chars=300)
    mc.record("StageB", _fake_result(tokens=150), context_chars=900)
    run = mc.build()
    assert run.total_tokens == 250
    assert run.llm_calls == 2


def test_max_context_size_is_largest_stage(llm):
    mc = MetricsCollector("Test", llm)
    mc.record("A", _fake_result(), context_chars=300)
    mc.record("B", _fake_result(), context_chars=1200)
    mc.record("C", _fake_result(), context_chars=500)
    run = mc.build()
    # context_chars rolls up into per-stage; the workflow sums them, but the
    # largest single stage is preserved in stages.
    assert max(s.context_chars for s in run.stages) == 1200


def test_duration_sums_across_stages(llm):
    mc = MetricsCollector("Test", llm)
    mc.record("A", _fake_result(dur=100.0), context_chars=10)
    mc.record("B", _fake_result(dur=250.0), context_chars=10)
    run = mc.build()
    assert run.duration_ms == 350.0
    assert run.duration_s == 0.35


def test_estimated_flag_propagates(llm):
    mc = MetricsCollector("Test", llm)
    mc.record("A", _fake_result(estimated=True), context_chars=10)
    assert mc.build().estimated is True


def test_set_response_and_quality(llm):
    mc = MetricsCollector("Test", llm)
    mc.record("A", _fake_result(), context_chars=10)
    mc.set_response("final answer")
    run = mc.build(QualityScore(correctness=8, completeness=8, relevance=8, tone=8))
    assert run.response == "final answer"
    assert run.quality_avg == 8.0


def test_cost_is_computed(llm):
    mc = MetricsCollector("Test", llm)
    mc.record("A", _fake_result(prompt=1000, completion=1000), context_chars=10)
    assert mc.build().estimated_cost > 0
