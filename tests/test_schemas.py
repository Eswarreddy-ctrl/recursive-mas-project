"""
tests/test_schemas.py — Pydantic schema validation tests.
Day: Monday (Week 1)  |  ~16 assertions
"""

import pytest
from pydantic import ValidationError
from models.schemas import (
    Ticket, SentimentResult, PriorityResult, CategoryResult, ResolutionResult,
    WorkflowMetrics, WorkflowResult, ComparisonResponse,
    Sentiment, Priority, Category, Department,
)


class TestTicket:
    def test_valid_minimal(self):
        t = Ticket(ticket_id="T001", text="Need billing help.")
        assert t.customer_id is None

    def test_valid_full(self):
        t = Ticket(ticket_id="T002", text="Login issue.", customer_id="C99")
        assert t.customer_id == "C99"

    def test_empty_text_rejected(self, empty_text_ticket_data):
        with pytest.raises(ValidationError) as exc:
            Ticket(**empty_text_ticket_data)
        assert "text" in str(exc.value)

    def test_missing_ticket_id_rejected(self):
        with pytest.raises(ValidationError):
            Ticket(text="Some text")  # type: ignore


class TestSentimentResult:
    def test_valid(self):
        s = SentimentResult(sentiment=Sentiment.NEGATIVE, confidence=0.95, raw_text="x")
        assert s.sentiment == Sentiment.NEGATIVE

    def test_confidence_above_1_rejected(self):
        with pytest.raises(ValidationError):
            SentimentResult(sentiment=Sentiment.POSITIVE, confidence=1.1, raw_text="x")

    def test_confidence_below_0_rejected(self):
        with pytest.raises(ValidationError):
            SentimentResult(sentiment=Sentiment.POSITIVE, confidence=-0.1, raw_text="x")

    def test_invalid_sentiment_rejected(self):
        with pytest.raises(ValidationError):
            SentimentResult(sentiment="furious", confidence=0.5, raw_text="x")  # type: ignore


class TestWorkflowMetrics:
    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            WorkflowMetrics(total_tokens=-1, total_llm_calls=5,
                            wall_clock_time_seconds=1.0, max_context_size=100, communication_flow=[])

    def test_negative_time_rejected(self):
        with pytest.raises(ValidationError):
            WorkflowMetrics(total_tokens=100, total_llm_calls=5,
                            wall_clock_time_seconds=-0.1, max_context_size=100, communication_flow=[])


class TestComparisonResponse:
    def test_recursive_fewer_tokens(self, sample_comparison):
        assert sample_comparison.recursive.metrics.total_tokens < sample_comparison.traditional.metrics.total_tokens

    def test_recursive_lower_max_context(self, sample_comparison):
        """⭐ Core metric: RecursiveMAS must have smaller max context size."""
        assert sample_comparison.recursive.metrics.max_context_size < sample_comparison.traditional.metrics.max_context_size

    def test_speedup_greater_than_1(self, sample_comparison):
        assert sample_comparison.speedup_factor > 1.0

    def test_token_reduction_positive(self, sample_comparison):
        assert sample_comparison.token_reduction_pct > 0