"""
metrics/metrics_tracker.py — Collects all 5 benchmark metrics per workflow run.

Metrics:
  1. total_tokens         — sum of prompt + completion tokens across all LLM calls
  2. total_llm_calls      — count of generate() invocations
  3. wall_clock_time      — end-to-end latency via injected Timer
  4. max_context_size     — char length of largest (prompt + context) passed to any agent  ⭐
  5. communication_flow   — ordered agent-to-agent handoff log

Usage:
    tracker = MetricsTracker()
    tracker.start_run()
    tracker.record_llm_call(tokens=150, context_size=800, agent_from="Root", agent_to="Sentiment")
    metrics = tracker.finish_run()   # → WorkflowMetrics
"""

from __future__ import annotations
from typing import Optional
from models.schemas import WorkflowMetrics
from metrics.timer import Timer


class MetricsTracker:
    def __init__(self, timer: Optional[Timer] = None):
        self._timer = timer or Timer()
        self._reset()

    def start_run(self) -> None:
        self._reset()
        self._timer.start()

    def finish_run(self) -> WorkflowMetrics:
        wall_clock = self._timer.stop()
        return WorkflowMetrics(
            total_tokens=self._total_tokens,
            total_llm_calls=self._total_llm_calls,
            wall_clock_time_seconds=wall_clock,
            max_context_size=self._max_context_size,
            communication_flow=list(self._communication_flow),
        )

    def record_llm_call(
        self,
        tokens: int,
        context_size: int,
        agent_from: Optional[str] = None,
        agent_to: Optional[str] = None,
    ) -> None:
        self._total_tokens += tokens
        self._total_llm_calls += 1
        if context_size > self._max_context_size:
            self._max_context_size = context_size
        if agent_from and agent_to:
            self._communication_flow.append(f"{agent_from}->{agent_to}")

    # ------------------------------------------------------------------
    # Introspection (readable mid-run)
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_llm_calls(self) -> int:
        return self._total_llm_calls

    @property
    def max_context_size(self) -> int:
        return self._max_context_size

    @property
    def communication_flow(self) -> list[str]:
        return list(self._communication_flow)

    @property
    def elapsed_seconds(self) -> float:
        return self._timer.elapsed()

    def _reset(self) -> None:
        self._total_tokens: int = 0
        self._total_llm_calls: int = 0
        self._max_context_size: int = 0
        self._communication_flow: list[str] = []