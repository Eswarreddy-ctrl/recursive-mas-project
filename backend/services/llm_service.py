"""Unified LLM service.

Backend selection (automatic):
  1. Google Gemini 2.0 Flash  (if GEMINI_API_KEY is set)
  2. Deterministic mock       (offline fallback — always available)

The mock is not a toy: it simulates token usage from the *real* size of the
prompt/context it is handed, and simulates latency proportional to the tokens
generated. This makes the benchmark architecturally honest offline — a workflow
that passes smaller context genuinely shows fewer tokens and lower simulated
latency, because those numbers are derived from the actual context it built.

Every call returns a uniform dict:
    {
      "text": str,             # raw model text
      "prompt_tokens": int,
      "completion_tokens": int,
      "total_tokens": int,
      "duration_ms": float,    # measured (live) or simulated (mock)
      "estimated": bool,       # True if tokens were estimated, not from API
    }
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # standard rough estimate for English text


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


# Approximate prices (USD per 1K tokens). Adjust to your contract.
# Gemini 2.0 Flash list pricing is ~$0.10 / 1M input, ~$0.40 / 1M output.
PRICE_TABLE = {
    "gemini": {"prompt": 0.0001, "completion": 0.0004},
    "mock": {"prompt": 0.0001, "completion": 0.0004},  # mirror gemini
}


def estimate_cost(prompt_tokens: int, completion_tokens: int, backend: str) -> float:
    p = PRICE_TABLE.get(backend, PRICE_TABLE["mock"])
    cost = (prompt_tokens / 1000.0) * p["prompt"] + (
        completion_tokens / 1000.0
    ) * p["completion"]
    return round(cost, 6)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_ms: float
    estimated: bool

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "estimated": self.estimated,
        }


class LLMService:
    """Backend-agnostic LLM wrapper (Gemini 2.0 Flash + mock fallback)."""

    def __init__(
        self,
        temperature: float = 0.0,
        simulate_latency: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.temperature = temperature
        self.simulate_latency = simulate_latency
        self.max_retries = max_retries
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        self._model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self._model = None
        self.backend = "mock"
        self._init()

    # ------------------------------------------------------------------ #
    def _init(self) -> None:
        if self._api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                    ),
                )
                self.backend = "gemini"
                logger.info("LLMService: using Gemini backend (%s)", self._model_name)
                return
            except Exception as e:  # pragma: no cover
                logger.warning("Gemini init failed (%s); falling back to mock", e)

        self.backend = "mock"
        logger.warning(
            "LLMService: GEMINI_API_KEY not set — using deterministic MOCK backend. "
            "Set GEMINI_API_KEY for live Gemini 2.0 Flash responses."
        )

    # ------------------------------------------------------------------ #
    def generate(self, system_prompt: str, user_content: str, max_tokens: int = 512) -> LLMResult:
        """Generate a completion. Returns a uniform LLMResult."""
        if self.backend == "gemini":
            return self._gemini(system_prompt, user_content, max_tokens)
        return self._mock(system_prompt, user_content, max_tokens)

    # ------------------------------------------------------------------ #
    def _gemini(self, system_prompt: str, user_content: str, max_tokens: int) -> LLMResult:
        import google.generativeai as genai

        # Gemini has a single prompt channel; prepend the system instruction.
        full_prompt = f"{system_prompt}\n\n{user_content}".strip()

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                dur = (time.perf_counter() - start) * 1000.0

                text = (getattr(resp, "text", "") or "").strip()

                # Token usage from API metadata when available.
                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    pt = int(getattr(usage, "prompt_token_count", 0) or 0)
                    ct = int(getattr(usage, "candidates_token_count", 0) or 0)
                    total = int(getattr(usage, "total_token_count", 0) or (pt + ct))
                    estimated = False
                else:
                    pt = estimate_tokens(full_prompt)
                    ct = estimate_tokens(text)
                    total = pt + ct
                    estimated = True

                return LLMResult(text, pt, ct, total, dur, estimated=estimated)

            except Exception as e:  # pragma: no cover
                last_err = e
                logger.warning("Gemini call attempt %d/%d failed: %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(2.0 * attempt)  # linear backoff: 2s, 4s

        logger.error("Gemini failed after %d attempts (%s); using mock", self.max_retries, last_err)
        return self._mock(system_prompt, user_content, max_tokens)

    # ------------------------------------------------------------------ #
    def _mock(self, system_prompt: str, user_content: str, max_tokens: int) -> LLMResult:
        """Deterministic mock.

        - prompt_tokens derived from the REAL prompt + context size
        - completion_tokens derived deterministically from prompt hash
        - latency simulated as a function of total tokens
        """
        full_prompt = f"{system_prompt}\n\n{user_content}"
        prompt_tokens = estimate_tokens(full_prompt)

        # Deterministic pseudo-random completion length seeded by content.
        seed = int(hashlib.sha256(full_prompt.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        completion_tokens = min(max_tokens, rng.randint(40, 160))

        text = self._mock_text(system_prompt, user_content, rng)

        total = prompt_tokens + completion_tokens
        # Simulated latency: base + per-token processing cost. Processing the
        # prompt scales with prompt size; generation scales with completion size.
        sim_ms = 120.0 + prompt_tokens * 0.9 + completion_tokens * 4.5
        if self.simulate_latency:
            # tiny real sleep so wall-clock timers register something, scaled down
            time.sleep(min(0.05, sim_ms / 8000.0))

        return LLMResult(text, prompt_tokens, completion_tokens, total, sim_ms, estimated=True)

    @staticmethod
    def _mock_text(system_prompt: str, user_content: str, rng: random.Random) -> str:
        sp = system_prompt.lower()
        if "classifier" in sp or "classify" in sp:
            cat = rng.choice(["Billing", "Technical", "Account", "Shipping", "General"])
            return f'{{"category": "{cat}", "intent": "support_request"}}'
        if "evaluation" in sp or "evaluate" in sp:
            base = rng.uniform(6.5, 9.5)
            return (
                f'{{"correctness": {round(base + rng.uniform(-0.5, 0.5), 1)}, '
                f'"completeness": {round(base + rng.uniform(-1.0, 0.5), 1)}, '
                f'"relevance": {round(base + rng.uniform(-0.5, 0.8), 1)}, '
                f'"tone": {round(base + rng.uniform(-0.3, 1.0), 1)}}}'
            )
        if "qa" in sp or "quality check" in sp or "verify" in sp:
            return "Verified: the response is accurate, complete, and on-topic."
        if "refine" in sp or "improve" in sp:
            return (
                "Refined response: I've reviewed the prior draft and tightened the "
                "explanation, added the missing next step, and confirmed the resolution path."
            )
        # default responder
        snippet = user_content.strip().split("\n")[-1][:80]
        return (
            "Thank you for reaching out. Based on your request "
            f'("{snippet}"), here is the recommended resolution with clear next steps.'
        )

    # ------------------------------------------------------------------ #
    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return estimate_cost(prompt_tokens, completion_tokens, self.backend)
