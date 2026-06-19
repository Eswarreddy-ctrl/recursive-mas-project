"""LLM service tests — backend selection, mock determinism, token + cost math.

OWNER: P3 (QA/DevOps)
"""
from backend.services.llm_service import (
    LLMService,
    estimate_cost,
    estimate_tokens,
)


def test_no_key_selects_mock_backend(llm):
    assert llm.backend == "mock"


def test_generate_returns_uniform_fields(llm):
    res = llm.generate("You are a responder.", "Help me reset my password.")
    d = res.as_dict()
    assert set(d) == {"text", "prompt_tokens", "completion_tokens",
                      "total_tokens", "duration_ms", "estimated"}
    assert d["total_tokens"] == d["prompt_tokens"] + d["completion_tokens"]
    assert d["estimated"] is True  # mock always estimates


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("a" * 4)
    long = estimate_tokens("a" * 400)
    assert long > short
    assert estimate_tokens("") >= 1  # never zero


def test_mock_is_deterministic(llm):
    a = llm.generate("You are a classifier.", "Charge appeared twice on my card.")
    b = llm.generate("You are a classifier.", "Charge appeared twice on my card.")
    assert a.text == b.text
    assert a.total_tokens == b.total_tokens


def test_larger_context_costs_more_prompt_tokens(llm):
    small = llm.generate("sys", "short")
    big = llm.generate("sys", "word " * 200)
    assert big.prompt_tokens > small.prompt_tokens


def test_cost_increases_with_tokens(llm):
    low = llm.cost(100, 100)
    high = llm.cost(1000, 1000)
    assert high > low > 0


def test_gemini_pricing_used_for_mock():
    # Mock mirrors gemini pricing in the PRICE_TABLE.
    c = estimate_cost(1000, 0, "mock")
    assert c == estimate_cost(1000, 0, "gemini")


def test_classifier_prompt_yields_json(llm):
    res = llm.generate("You are a classifier agent.", "My bill is wrong.")
    assert res.text.strip().startswith("{")
