"""Agent tests — leaf agents run, evaluation agent scores.

OWNER: P3 (QA/DevOps)
"""
from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.recursive_agents import ClassifyAgent, RespondAgent
from backend.agents.traditional_agents import (
    ClassifierAgent,
    QAAgent,
    ResponderAgent,
)
from backend.models.schemas import QualityScore


def test_base_agent_run_returns_meta(llm):
    agent = ResponderAgent(llm)
    out = agent.run("Help me reset my password.")
    assert "text" in out
    assert "_context_chars" in out
    assert out["_context_chars"] > 0


def test_context_chars_tracks_input_size(llm):
    agent = ResponderAgent(llm)
    small = agent.run("hi")
    big = agent.run("word " * 100)
    assert big["_context_chars"] > small["_context_chars"]


def test_traditional_agents_instantiate(llm):
    for cls in (ClassifierAgent, ResponderAgent, QAAgent):
        agent = cls(llm)
        assert agent.system_prompt  # non-empty prompt
        assert agent.run("test").get("text") is not None


def test_recursive_agents_instantiate(llm):
    for cls in (ClassifyAgent, RespondAgent):
        agent = cls(llm)
        assert agent.run("test").get("text") is not None


def test_evaluation_agent_returns_quality_score(llm):
    ev = EvaluationAgent(llm)
    score, meta = ev.evaluate("My payment failed.", "We have issued a refund.")
    assert isinstance(score, QualityScore)
    assert 0 <= score.average <= 10
    assert "total_tokens" in meta


def test_evaluation_parse_handles_garbage():
    # Direct parser test: invalid text falls back to a neutral score, no crash.
    score = EvaluationAgent._parse("not json at all")
    assert isinstance(score, QualityScore)
    assert 0 <= score.average <= 10
