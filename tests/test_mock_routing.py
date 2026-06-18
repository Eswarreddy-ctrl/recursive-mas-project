"""
tests/test_mock_routing.py — Mock fallback routing and GeminiService mock mode.
Day: Tuesday (Week 1)  |  ~12 assertions
"""

import pytest
from llm.gemini_service import GeminiService, _mock_response


class TestMockResponse:
    def test_billing_routes_to_billing(self):
        text, _ = _mock_response("My invoice shows wrong charge")
        assert "category:billing" in text and "routing:billing" in text

    def test_technical_routes_to_engineering(self):
        text, _ = _mock_response("The app crashes with an error")
        assert "category:technical" in text and "routing:engineering" in text

    def test_account_routes_to_support(self):
        text, _ = _mock_response("Cannot login, password reset broken")
        assert "category:account" in text and "routing:support" in text

    def test_shipping_routes_to_logistics(self):
        text, _ = _mock_response("My package delivery tracking is lost")
        assert "category:shipping" in text and "routing:logistics" in text

    def test_negative_sentiment(self):
        text, _ = _mock_response("This is terrible, worst service, I hate it")
        assert "sentiment:negative" in text

    def test_positive_sentiment(self):
        text, _ = _mock_response("Amazing service, I love this, it's wonderful")
        assert "sentiment:positive" in text

    def test_neutral_sentiment_default(self):
        text, _ = _mock_response("I have a question please")
        assert "sentiment:neutral" in text

    def test_critical_priority(self):
        text, _ = _mock_response("Urgent critical outage system is down")
        assert "priority:critical" in text

    def test_token_estimate_positive(self):
        _, tokens = _mock_response("Some prompt")
        assert tokens > 0

    def test_longer_prompt_more_tokens(self):
        _, short = _mock_response("Hi")
        _, long = _mock_response("I need urgent help with billing invoice charge " * 10)
        assert long > short


class TestGeminiServiceMockMode:
    def test_no_api_key_is_mock(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert GeminiService().is_mock is True

    def test_mock_under_150ms(self, monkeypatch):
        import time
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        svc = GeminiService()
        start = time.perf_counter()
        svc.generate("My invoice has a wrong charge, I need urgent help")
        assert (time.perf_counter() - start) * 1000 < 150