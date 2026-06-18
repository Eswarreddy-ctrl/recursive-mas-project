"""
llm/gemini_service.py — Gemini API wrapper with mock fallback and exponential backoff.
Falls back to keyword-based mock parser when GEMINI_API_KEY is absent or on rate limit.
"""

from __future__ import annotations

import os
import re
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock fallback — keyword-based, deterministic, <150ms
# ---------------------------------------------------------------------------

_NEGATIVE_KEYWORDS = {"angry", "frustrated", "terrible", "broken", "worst", "awful", "useless", "hate"}
_POSITIVE_KEYWORDS = {"great", "love", "excellent", "amazing", "perfect", "wonderful", "fantastic"}
_CRITICAL_KEYWORDS = {"urgent", "asap", "immediately", "critical", "outage", "down", "broken"}
_HIGH_KEYWORDS     = {"important", "priority", "soon", "quickly", "need", "must"}
_BILLING_KEYWORDS  = {"invoice", "charge", "payment", "refund", "billing", "money", "cost", "price"}
_TECHNICAL_KEYWORDS= {"error", "bug", "crash", "broken", "not working", "issue", "fail"}
_ACCOUNT_KEYWORDS  = {"login", "password", "account", "access", "locked", "reset"}
_SHIPPING_KEYWORDS = {"delivery", "ship", "package", "tracking", "order", "arrived"}


def _mock_response(prompt: str) -> tuple[str, int]:
    """
    Returns (mock_text, token_estimate).
    Token estimate ≈ len(prompt) // 4 + 20 (rough approximation).
    """
    text_lower = prompt.lower()

    if any(k in text_lower for k in _NEGATIVE_KEYWORDS):
        sentiment = "negative"
    elif any(k in text_lower for k in _POSITIVE_KEYWORDS):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    if any(k in text_lower for k in _CRITICAL_KEYWORDS):
        priority = "critical"
    elif any(k in text_lower for k in _HIGH_KEYWORDS):
        priority = "high"
    else:
        priority = "medium"

    if any(k in text_lower for k in _BILLING_KEYWORDS):
        category = "billing"
    elif any(k in text_lower for k in _TECHNICAL_KEYWORDS):
        category = "technical"
    elif any(k in text_lower for k in _ACCOUNT_KEYWORDS):
        category = "account"
    elif any(k in text_lower for k in _SHIPPING_KEYWORDS):
        category = "shipping"
    else:
        category = "general"

    routing_map = {
        "billing": "billing", "technical": "engineering",
        "account": "support", "shipping": "logistics", "general": "support",
    }
    routing = routing_map[category]

    mock_text = (
        f"sentiment:{sentiment}|priority:{priority}|"
        f"category:{category}|routing:{routing}|"
        "suggested_action:Review ticket and respond within SLA"
    )
    token_estimate = len(prompt) // 4 + 20
    return mock_text, token_estimate


# ---------------------------------------------------------------------------
# GeminiService
# ---------------------------------------------------------------------------

class GeminiService:
    """
    Wraps the Gemini generate API.
    - If GEMINI_API_KEY is unset, runs in mock mode automatically.
    - Retries up to max_retries times with exponential backoff on failure.
    - Falls back to mock after all retries are exhausted.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._mock_mode = False

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — running in mock mode")
            self._mock_mode = True
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel("gemini-1.5-flash")
        except ImportError:
            logger.warning("google-generativeai not installed — running in mock mode")
            self._mock_mode = True

    @property
    def is_mock(self) -> bool:
        return self._mock_mode

    def generate(self, prompt: str, context: str = "") -> tuple[str, int]:
        """
        Returns (response_text, tokens_used).
        Raises RuntimeError after max_retries exhausted (then falls back to mock).
        """
        full_prompt = f"{context}\n\n{prompt}".strip() if context else prompt

        if self._mock_mode:
            return _mock_response(full_prompt)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._model.generate_content(full_prompt)
                text = response.text
                tokens = getattr(response, "usage_metadata", None)
                if tokens:
                    total = (tokens.prompt_token_count or 0) + (tokens.candidates_token_count or 0)
                else:
                    total = len(full_prompt) // 4 + len(text) // 4
                return text, total

            except Exception as exc:
                last_exc = exc
                delay = self.base_delay * (2 ** attempt)
                logger.warning("Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                               attempt + 1, self.max_retries, exc, delay)
                if attempt < self.max_retries - 1:
                    time.sleep(delay)

        logger.error("All %d retries failed (%s). Falling back to mock.", self.max_retries, last_exc)
        self._mock_mode = True
        return _mock_response(full_prompt)