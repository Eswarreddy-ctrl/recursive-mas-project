"""Metrics collector.

Accumulates StageMetrics across a workflow run and folds them into a
WorkflowRun. Keeps live (API) vs estimated (mock) token sources honest.
"""

from __future__ import annotations

import logging
from typing import List

from backend.models.schemas import StageMetrics, WorkflowRun, QualityScore
from backend.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics for a single workflow run."""

    def __init__(self, label: str, llm: LLMService) -> None:
        self.label = label
        self.llm = llm
        self.stages: List[StageMetrics] = []
        self._response_text: str = ""

    def record(self, name: str, llm_result_dict: dict, context_chars: int) -> None:
        """Record one LLM stage from an LLMService result dict."""
        pt = llm_result_dict["prompt_tokens"]
        ct = llm_result_dict["completion_tokens"]
        cost = self.llm.cost(pt, ct)
        stage = StageMetrics(
            name=name,
            duration_ms=round(llm_result_dict["duration_ms"], 3),
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=llm_result_dict["total_tokens"],
            llm_calls=1,
            estimated_cost=cost,
            context_chars=context_chars,
            estimated=llm_result_dict["estimated"],
        )
        self.stages.append(stage)

    def set_response(self, text: str) -> None:
        self._response_text = text

    def build(self, quality: QualityScore | None = None) -> WorkflowRun:
        total_tokens = sum(s.total_tokens for s in self.stages)
        prompt_tokens = sum(s.prompt_tokens for s in self.stages)
        completion_tokens = sum(s.completion_tokens for s in self.stages)
        duration_ms = sum(s.duration_ms for s in self.stages)
        cost = round(sum(s.estimated_cost for s in self.stages), 6)
        calls = sum(s.llm_calls for s in self.stages)
        context_chars = sum(s.context_chars for s in self.stages)
        estimated = any(s.estimated for s in self.stages)

        return WorkflowRun(
            label=self.label,
            response=self._response_text,
            duration_ms=round(duration_ms, 3),
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_calls=calls,
            estimated_cost=cost,
            context_chars=context_chars,
            quality=quality or QualityScore(),
            stages=self.stages,
            estimated=estimated,
        )
