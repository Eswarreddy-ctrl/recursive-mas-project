"""Pydantic data models for the RecursiveMAS vs Traditional MAS benchmark.

All data contracts shared between the agents, services, benchmark layer,
API, and frontend live here.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class QueryInput(BaseModel):
    """A single user query to benchmark."""

    query: str = Field(..., description="The user query / support request to analyze")


# --------------------------------------------------------------------------- #
# Quality scoring
# --------------------------------------------------------------------------- #
class QualityScore(BaseModel):
    """Output of the Evaluation Agent for one response."""

    correctness: float = Field(0.0, ge=0, le=10)
    completeness: float = Field(0.0, ge=0, le=10)
    relevance: float = Field(0.0, ge=0, le=10)
    tone: float = Field(0.0, ge=0, le=10)

    @property
    def average(self) -> float:
        return round(
            (self.correctness + self.completeness + self.relevance + self.tone) / 4.0, 3
        )

    def model_dump_with_average(self) -> dict:
        d = self.model_dump()
        d["average"] = self.average
        return d


# --------------------------------------------------------------------------- #
# Per-stage / per-round metrics
# --------------------------------------------------------------------------- #
class StageMetrics(BaseModel):
    """Metrics for a single agent invocation (a pipeline stage or a round)."""

    name: str = Field(..., description="Stage or round label")
    duration_ms: float = Field(0.0, description="Wall-clock duration in milliseconds")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    estimated_cost: float = Field(0.0, description="Estimated USD cost")
    context_chars: int = Field(0, description="Characters of context processed")
    estimated: bool = Field(False, description="Whether token counts were estimated")


class RoundMetrics(BaseModel):
    """Metrics for one recursive round (matches the spec contract)."""

    round_number: int
    duration_ms: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    estimated_cost: float = 0.0
    quality_score: float = 0.0


# --------------------------------------------------------------------------- #
# Workflow-level results
# --------------------------------------------------------------------------- #
class WorkflowRun(BaseModel):
    """Result of running one workflow (traditional, or one recursive round)."""

    label: str
    response: str = Field("", description="Final response text produced")
    duration_ms: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    estimated_cost: float = 0.0
    context_chars: int = 0
    quality: QualityScore = Field(default_factory=QualityScore)
    stages: List[StageMetrics] = Field(default_factory=list)
    estimated: bool = False

    @property
    def duration_s(self) -> float:
        return round(self.duration_ms / 1000.0, 4)

    @property
    def quality_avg(self) -> float:
        return self.quality.average


# --------------------------------------------------------------------------- #
# Benchmark storage model (as specified)
# --------------------------------------------------------------------------- #
class BenchmarkResult(BaseModel):
    """Flat benchmark record comparing Traditional MAS to 3 recursive rounds."""

    query: str

    # Times (seconds)
    traditional_time: float = 0.0
    recursive_r1_time: float = 0.0
    recursive_r2_time: float = 0.0
    recursive_r3_time: float = 0.0

    # Tokens
    traditional_tokens: int = 0
    recursive_r1_tokens: int = 0
    recursive_r2_tokens: int = 0
    recursive_r3_tokens: int = 0

    # Cost (USD)
    traditional_cost: float = 0.0
    recursive_r1_cost: float = 0.0
    recursive_r2_cost: float = 0.0
    recursive_r3_cost: float = 0.0

    # Quality (0-10)
    traditional_quality: float = 0.0
    recursive_r1_quality: float = 0.0
    recursive_r2_quality: float = 0.0
    recursive_r3_quality: float = 0.0

    # Extra: LLM calls (handy for the dashboard)
    traditional_calls: int = 0
    recursive_r1_calls: int = 0
    recursive_r2_calls: int = 0
    recursive_r3_calls: int = 0

    backend_mode: str = Field("mock", description="live | mock — token source")


# --------------------------------------------------------------------------- #
# Speedup + summary
# --------------------------------------------------------------------------- #
class SpeedupCard(BaseModel):
    """A single speedup KPI card."""

    round_label: str
    recursive_time: float
    speedup: float  # traditional_time / recursive_time
    time_saved_pct: float  # (1 - recursive/traditional) * 100


class ResearchSummary(BaseModel):
    """Auto-generated headline comparison vs Traditional MAS."""

    best_speedup: float = 0.0
    token_reduction_pct: float = 0.0
    cost_reduction_pct: float = 0.0
    quality_delta_pct: float = 0.0
    narrative: str = ""


# --------------------------------------------------------------------------- #
# Full API response
# --------------------------------------------------------------------------- #
class BenchmarkResponse(BaseModel):
    """Complete benchmark payload returned by the API."""

    query: str
    traditional: WorkflowRun
    recursive_rounds: List[WorkflowRun]  # R1, R2, R3
    benchmark: BenchmarkResult
    speedups: List[SpeedupCard]
    summary: ResearchSummary
