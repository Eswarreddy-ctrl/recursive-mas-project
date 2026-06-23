"""Benchmark service.

Runs Traditional MAS and the 3 Recursive rounds for a query, then assembles
the BenchmarkResult, speedup KPI cards, and an auto-generated research summary.

All headline figures (speedup, % reductions, quality delta) are COMPUTED from
measured/simulated data — never hardcoded.
"""

from __future__ import annotations

import logging
from typing import List

from backend.models.schemas import (
    BenchmarkResponse,
    BenchmarkResult,
    ResearchSummary,
    SpeedupCard,
    WorkflowRun,
)
from backend.services.llm_service import LLMService
from backend.services.recursive_mas import RecursiveMAS
from backend.services.traditional_mas import TraditionalMAS

logger = logging.getLogger(__name__)


class BenchmarkService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self.llm = llm or LLMService()
        self.traditional = TraditionalMAS(self.llm)
        self.recursive = RecursiveMAS(self.llm)

    # ------------------------------------------------------------------ #
    def run(self, query: str) -> BenchmarkResponse:
        trad = self.traditional.run(query)
        rounds = self.recursive.run(query)  # [R1, R2, R3]

        benchmark = self._to_benchmark_result(query, trad, rounds)
        speedups = self._speedups(trad, rounds)
        summary = self._summary(trad, rounds)

        return BenchmarkResponse(
            query=query,
            traditional=trad,
            recursive_rounds=rounds,
            benchmark=benchmark,
            speedups=speedups,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    def _to_benchmark_result(
        self, query: str, trad: WorkflowRun, rounds: List[WorkflowRun]
    ) -> BenchmarkResult:
        r1, r2, r3 = rounds
        return BenchmarkResult(
            query=query,
            traditional_time=trad.duration_s,
            recursive_r1_time=r1.duration_s,
            recursive_r2_time=r2.duration_s,
            recursive_r3_time=r3.duration_s,
            traditional_tokens=trad.total_tokens,
            recursive_r1_tokens=r1.total_tokens,
            recursive_r2_tokens=r2.total_tokens,
            recursive_r3_tokens=r3.total_tokens,
            traditional_cost=trad.estimated_cost,
            recursive_r1_cost=r1.estimated_cost,
            recursive_r2_cost=r2.estimated_cost,
            recursive_r3_cost=r3.estimated_cost,
            traditional_quality=trad.quality_avg,
            recursive_r1_quality=r1.quality_avg,
            recursive_r2_quality=r2.quality_avg,
            recursive_r3_quality=r3.quality_avg,
            traditional_calls=trad.llm_calls,
            recursive_r1_calls=r1.llm_calls,
            recursive_r2_calls=r2.llm_calls,
            recursive_r3_calls=r3.llm_calls,
            backend_mode=self.llm.backend,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _speedups(trad: WorkflowRun, rounds: List[WorkflowRun]) -> List[SpeedupCard]:
        cards: List[SpeedupCard] = []
        t = max(trad.duration_s, 1e-6)
        for i, run in enumerate(rounds, start=1):
            rt = max(run.duration_s, 1e-6)
            cards.append(
                SpeedupCard(
                    round_label=f"Round {i}",
                    recursive_time=run.duration_s,
                    speedup=round(t / rt, 2),
                    time_saved_pct=round((1 - rt / t) * 100, 1),
                )
            )
        return cards

    # ------------------------------------------------------------------ #
    @staticmethod
    def _summary(trad: WorkflowRun, rounds: List[WorkflowRun]) -> ResearchSummary:
        # Use the best (fastest) recursive round as the headline comparator.
        best = min(rounds, key=lambda r: r.duration_s)
        t_time = max(trad.duration_s, 1e-6)
        t_tok = max(trad.total_tokens, 1)
        t_cost = max(trad.estimated_cost, 1e-9)
        t_qual = max(trad.quality_avg, 1e-6)

        best_speedup = round(t_time / max(best.duration_s, 1e-6), 2)
        token_red = round((1 - best.total_tokens / t_tok) * 100, 1)
        cost_red = round((1 - best.estimated_cost / t_cost) * 100, 1)
        # Quality: compare best round's quality to traditional
        qual_delta = round((best.quality_avg / t_qual - 1) * 100, 1)

        narrative = (
            f"RecursiveMAS (best round: {best.label}) achieved "
            f"{best_speedup}x faster inference, {token_red}% "
            f"{'lower' if token_red >= 0 else 'higher'} token usage, "
            f"{cost_red}% {'lower' if cost_red >= 0 else 'higher'} cost, and "
            f"{abs(qual_delta)}% {'higher' if qual_delta >= 0 else 'lower'} response "
            f"quality compared to Traditional MAS."
        )

        return ResearchSummary(
            best_speedup=best_speedup,
            token_reduction_pct=token_red,
            cost_reduction_pct=cost_red,
            quality_delta_pct=qual_delta,
            narrative=narrative,
        )
