"""
tests/conftest.py — Shared pytest fixtures.
"""

import pytest
from models.schemas import (
    Ticket, SentimentResult, PriorityResult, CategoryResult, ResolutionResult,
    WorkflowMetrics, WorkflowResult, ComparisonResponse,
    Sentiment, Priority, Category, Department,
)


@pytest.fixture
def simple_ticket():
    return Ticket(ticket_id="T001", text="My invoice is wrong, please help.", customer_id="C42")

@pytest.fixture
def angry_ticket():
    return Ticket(ticket_id="T002", text="This is terrible! My account is broken and nobody helps!")

@pytest.fixture
def technical_ticket():
    return Ticket(ticket_id="T003", text="The app crashes every time I login. Urgent fix needed.")

@pytest.fixture
def empty_text_ticket_data():
    return {"ticket_id": "T_BAD", "text": ""}

@pytest.fixture
def sample_traditional_metrics():
    return WorkflowMetrics(
        total_tokens=1200, total_llm_calls=5, wall_clock_time_seconds=4.5,
        max_context_size=3200,
        communication_flow=["Ticket->Sentiment", "Sentiment->Priority", "Priority->Category", "Category->Resolution", "Resolution->Output"],
    )

@pytest.fixture
def sample_recursive_metrics():
    return WorkflowMetrics(
        total_tokens=620, total_llm_calls=5, wall_clock_time_seconds=2.1,
        max_context_size=980,
        communication_flow=["Root->CustomerCoord", "CustomerCoord->Sentiment", "CustomerCoord->Category", "Root->BusinessCoord", "BusinessCoord->Resolution"],
    )

@pytest.fixture
def sample_sentiment():
    return SentimentResult(sentiment=Sentiment.NEGATIVE, confidence=0.91, raw_text="sentiment:negative")

@pytest.fixture
def sample_priority():
    return PriorityResult(priority=Priority.HIGH, reasoning="Urgent language detected", raw_text="priority:high")

@pytest.fixture
def sample_category():
    return CategoryResult(category=Category.BILLING, raw_text="category:billing")

@pytest.fixture
def sample_resolution():
    return ResolutionResult(routing=Department.BILLING, suggested_action="Review charge", raw_text="routing:billing")

@pytest.fixture
def sample_traditional_result(simple_ticket, sample_traditional_metrics,
                               sample_sentiment, sample_priority, sample_category, sample_resolution):
    return WorkflowResult(
        workflow_name="traditional", ticket_id=simple_ticket.ticket_id,
        sentiment=sample_sentiment, priority=sample_priority,
        category=sample_category, resolution=sample_resolution,
        metrics=sample_traditional_metrics,
    )

@pytest.fixture
def sample_recursive_result(simple_ticket, sample_recursive_metrics,
                             sample_sentiment, sample_priority, sample_category, sample_resolution):
    return WorkflowResult(
        workflow_name="recursive", ticket_id=simple_ticket.ticket_id,
        sentiment=sample_sentiment, priority=sample_priority,
        category=sample_category, resolution=sample_resolution,
        metrics=sample_recursive_metrics,
    )

@pytest.fixture
def sample_comparison(sample_traditional_result, sample_recursive_result):
    trad, rec = sample_traditional_result, sample_recursive_result
    return ComparisonResponse(
        ticket_id=trad.ticket_id,
        traditional=trad,
        recursive=rec,
        token_reduction_pct=(1 - rec.metrics.total_tokens / trad.metrics.total_tokens) * 100,
        speedup_factor=trad.metrics.wall_clock_time_seconds / rec.metrics.wall_clock_time_seconds,
        context_size_reduction_pct=(1 - rec.metrics.max_context_size / trad.metrics.max_context_size) * 100,
    )