"""
tests/test_agents.py — Unit tests for all four agents using MagicMock service.
Day: Tuesday (Week 1)  |  ~18 assertions
"""

import pytest
from unittest.mock import MagicMock
from agents.sentiment_agent import SentimentAgent
from agents.priority_agent import PriorityAgent
from agents.category_agent import CategoryAgent
from agents.resolution_agent import ResolutionAgent
from models.schemas import Sentiment, Priority, Category, Department


def mock_svc(text: str, tokens: int = 50):
    svc = MagicMock()
    svc.generate.return_value = (text, tokens)
    return svc


class TestSentimentAgent:
    def test_negative(self):
        r, _ = SentimentAgent(mock_svc("sentiment:negative|confidence:0.92")).run("Terrible!")
        assert r.sentiment == Sentiment.NEGATIVE
        assert r.confidence == pytest.approx(0.92)

    def test_positive(self):
        r, _ = SentimentAgent(mock_svc("sentiment:positive|confidence:0.88")).run("Great!")
        assert r.sentiment == Sentiment.POSITIVE

    def test_unknown_defaults_to_neutral(self):
        r, _ = SentimentAgent(mock_svc("sentiment:confused|confidence:0.5")).run("?")
        assert r.sentiment == Sentiment.NEUTRAL

    def test_confidence_clamped(self):
        r, _ = SentimentAgent(mock_svc("sentiment:positive|confidence:1.9")).run("x")
        assert r.confidence <= 1.0

    def test_context_size_tracked(self):
        agent = SentimentAgent(mock_svc("sentiment:neutral|confidence:0.6"))
        agent.run("Short", context="Some context here")
        assert agent.last_context_size > 0

    def test_service_called_once(self):
        svc = mock_svc("sentiment:neutral|confidence:0.6")
        SentimentAgent(svc).run("Some text")
        assert svc.generate.call_count == 1


class TestPriorityAgent:
    def test_critical(self):
        r, _ = PriorityAgent(mock_svc("priority:critical|reasoning:System down")).run("Outage!")
        assert r.priority == Priority.CRITICAL

    def test_low(self):
        r, _ = PriorityAgent(mock_svc("priority:low|reasoning:Non-urgent")).run("Question")
        assert r.priority == Priority.LOW

    def test_unknown_defaults_to_medium(self):
        r, _ = PriorityAgent(mock_svc("priority:sorta_urgent|reasoning:Unclear")).run("x")
        assert r.priority == Priority.MEDIUM

    def test_reasoning_extracted(self):
        r, _ = PriorityAgent(mock_svc("priority:high|reasoning:Customer is angry")).run("x")
        assert "angry" in r.reasoning.lower()


class TestCategoryAgent:
    def test_billing_with_subcategory(self):
        r, _ = CategoryAgent(mock_svc("category:billing|subcategory:overcharge")).run("Wrong charge")
        assert r.category == Category.BILLING
        assert r.subcategory == "overcharge"

    def test_empty_subcategory_is_none(self):
        r, _ = CategoryAgent(mock_svc("category:general|subcategory:")).run("x")
        assert r.subcategory is None

    def test_unknown_defaults_to_general(self):
        r, _ = CategoryAgent(mock_svc("category:unknown|subcategory:")).run("x")
        assert r.category == Category.GENERAL


class TestResolutionAgent:
    def test_engineering_routing(self):
        r, _ = ResolutionAgent(mock_svc("routing:engineering|suggested_action:Escalate to dev")).run("Bug")
        assert r.routing == Department.ENGINEERING
        assert "dev" in r.suggested_action.lower()

    def test_logistics_routing(self):
        r, _ = ResolutionAgent(mock_svc("routing:logistics|suggested_action:Check tracking")).run("Package")
        assert r.routing == Department.LOGISTICS

    def test_unknown_defaults_to_support(self):
        r, _ = ResolutionAgent(mock_svc("routing:unknown|suggested_action:Review")).run("x")
        assert r.routing == Department.SUPPORT