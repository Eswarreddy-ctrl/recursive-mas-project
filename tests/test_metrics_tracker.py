"""
tests/test_metrics_tracker.py — All 5 metrics correctly captured.
Day: Wednesday (Week 1)  |  ~18 assertions
"""

import pytest
from metrics.metrics_tracker import MetricsTracker
from metrics.timer import Timer
from models.schemas import WorkflowMetrics


def tracker(elapsed: float = 1.0) -> MetricsTracker:
    return MetricsTracker(timer=Timer(mock_elapsed=elapsed))


class TestTokens:
    def test_single_call(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=150, context_size=500)
        assert t.finish_run().total_tokens == 150

    def test_summed_across_calls(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=100, context_size=400)
        t.record_llm_call(tokens=200, context_size=600)
        assert t.finish_run().total_tokens == 300

    def test_zero_tokens_ok(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=0, context_size=100)
        assert t.finish_run().total_tokens == 0


class TestLLMCallCount:
    def test_five_calls(self):
        t = tracker(); t.start_run()
        for _ in range(5):
            t.record_llm_call(tokens=100, context_size=500)
        assert t.finish_run().total_llm_calls == 5

    def test_no_calls_zero(self):
        t = tracker(); t.start_run()
        assert t.finish_run().total_llm_calls == 0

    def test_increments_per_call(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=10, context_size=50)
        assert t.total_llm_calls == 1
        t.record_llm_call(tokens=10, context_size=50)
        assert t.total_llm_calls == 2


class TestMaxContextSize:
    def test_largest_captured(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=100, context_size=500)
        t.record_llm_call(tokens=100, context_size=3200)
        t.record_llm_call(tokens=100, context_size=800)
        assert t.finish_run().max_context_size == 3200

    def test_no_calls_zero(self):
        t = tracker(); t.start_run()
        assert t.finish_run().max_context_size == 0

    def test_recursive_smaller_than_traditional(self):
        """⭐ Core metric: RecursiveMAS must have smaller max context than Traditional."""
        trad = tracker(4.5); trad.start_run()
        trad.record_llm_call(tokens=300, context_size=3200)
        trad.record_llm_call(tokens=200, context_size=2800)
        trad_m = trad.finish_run()

        rec = tracker(2.1); rec.start_run()
        rec.record_llm_call(tokens=150, context_size=980)
        rec.record_llm_call(tokens=100, context_size=750)
        rec_m = rec.finish_run()

        assert rec_m.max_context_size < trad_m.max_context_size


class TestWallClockTime:
    def test_mock_elapsed(self):
        t = tracker(3.75); t.start_run()
        assert t.finish_run().wall_clock_time_seconds == pytest.approx(3.75)


class TestCommunicationFlow:
    def test_recorded_in_order(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=100, context_size=500, agent_from="Root",          agent_to="CustomerCoord")
        t.record_llm_call(tokens=80,  context_size=400, agent_from="CustomerCoord", agent_to="Sentiment")
        assert t.finish_run().communication_flow == ["Root->CustomerCoord", "CustomerCoord->Sentiment"]

    def test_no_agents_empty_flow(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=100, context_size=500)
        assert t.finish_run().communication_flow == []

    def test_partial_agent_info_not_logged(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=100, context_size=500, agent_from="Root")  # no agent_to
        assert t.finish_run().communication_flow == []


class TestFullRun:
    def test_returns_workflow_metrics(self):
        t = tracker(); t.start_run()
        t.record_llm_call(tokens=100, context_size=500, agent_from="A", agent_to="B")
        assert isinstance(t.finish_run(), WorkflowMetrics)

    def test_traditional_five_agent_pipeline(self):
        t = tracker(4.5); t.start_run()
        t.record_llm_call(250, 800,  "Ticket",    "Sentiment")
        t.record_llm_call(280, 1600, "Sentiment", "Priority")
        t.record_llm_call(300, 2400, "Priority",  "Category")
        t.record_llm_call(320, 3200, "Category",  "Resolution")
        t.record_llm_call(50,  3200, "Resolution","Output")
        m = t.finish_run()
        assert m.total_llm_calls == 5
        assert m.total_tokens == 1200
        assert m.max_context_size == 3200

    def test_recursive_five_agent_pipeline(self):
        t = tracker(2.1); t.start_run()
        t.record_llm_call(120, 400, "Root",          "CustomerCoord")
        t.record_llm_call(110, 350, "CustomerCoord", "Sentiment")
        t.record_llm_call(100, 380, "CustomerCoord", "Category")
        t.record_llm_call(130, 420, "Root",          "BusinessCoord")
        t.record_llm_call(160, 980, "BusinessCoord", "Resolution")
        m = t.finish_run()
        assert m.total_llm_calls == 5
        assert m.max_context_size == 980   # much lower than traditional's 3200
        assert m.total_tokens == 620