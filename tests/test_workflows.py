"""Workflow tests — Traditional pipeline and 3 Recursive rounds.

OWNER: P3 (QA/DevOps)
"""
from backend.models.schemas import WorkflowRun
from backend.services.recursive_mas import RecursiveMAS
from backend.services.traditional_mas import TraditionalMAS


def test_traditional_runs_three_agents(llm, sample_query):
    run = TraditionalMAS(llm).run(sample_query)
    assert isinstance(run, WorkflowRun)
    # Classifier + Responder + QA = 3 LLM calls (evaluation excluded from totals).
    assert run.llm_calls == 3
    assert run.total_tokens > 0
    assert run.response  # non-empty


def test_traditional_has_quality_score(llm, sample_query):
    run = TraditionalMAS(llm).run(sample_query)
    assert 0 <= run.quality_avg <= 10


def test_recursive_returns_three_rounds(llm, sample_query):
    rounds = RecursiveMAS(llm).run(sample_query)
    assert len(rounds) == 3
    assert [r.label for r in rounds] == ["Recursive R1", "Recursive R2", "Recursive R3"]


def test_recursive_each_round_has_two_calls(llm, sample_query):
    rounds = RecursiveMAS(llm).run(sample_query)
    for r in rounds:
        assert r.llm_calls == 2
        assert r.total_tokens > 0


def test_round_one_context_smaller_than_traditional(llm, sample_query):
    """RecursiveMAS R1 passes less context than the full traditional pipeline."""
    trad = TraditionalMAS(llm).run(sample_query)
    r1 = RecursiveMAS(llm).run(sample_query)[0]
    # Largest single stage context in R1 should not exceed traditional's largest.
    trad_max = max(s.context_chars for s in trad.stages)
    r1_max = max(s.context_chars for s in r1.stages)
    assert r1_max <= trad_max


def test_workflows_are_deterministic(llm, sample_query):
    a = TraditionalMAS(llm).run(sample_query)
    b = TraditionalMAS(llm).run(sample_query)
    assert a.total_tokens == b.total_tokens
