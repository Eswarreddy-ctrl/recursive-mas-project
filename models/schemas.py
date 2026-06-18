"""
models/schemas.py — Shared Pydantic schemas for RecursiveMAS benchmarking project.
All agent inputs/outputs validated at runtime; invalid data raises clear errors.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    ACCOUNT = "account"
    SHIPPING = "shipping"


class Department(str, Enum):
    SUPPORT = "support"
    BILLING = "billing"
    ENGINEERING = "engineering"
    LOGISTICS = "logistics"
    ESCALATIONS = "escalations"


# ---------------------------------------------------------------------------
# Ticket (shared input)
# ---------------------------------------------------------------------------

class Ticket(BaseModel):
    ticket_id: str = Field(..., description="Unique ticket identifier")
    text: str = Field(..., min_length=1, description="Raw customer message")
    customer_id: Optional[str] = Field(None, description="Optional customer reference")


# ---------------------------------------------------------------------------
# Per-agent output schemas
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_text: str = Field(..., description="Agent's raw LLM output")


class PriorityResult(BaseModel):
    priority: Priority
    reasoning: str
    raw_text: str


class CategoryResult(BaseModel):
    category: Category
    subcategory: Optional[str] = None
    raw_text: str


class ResolutionResult(BaseModel):
    routing: Department
    suggested_action: str
    raw_text: str


# ---------------------------------------------------------------------------
# Metrics (captured per workflow run)
# ---------------------------------------------------------------------------

class WorkflowMetrics(BaseModel):
    total_tokens: int = Field(..., ge=0, description="Sum of prompt + completion tokens across all LLM calls")
    total_llm_calls: int = Field(..., ge=0, description="Count of generate() invocations")
    wall_clock_time_seconds: float = Field(..., ge=0.0, description="Elapsed seconds from ticket input to final output")
    max_context_size: int = Field(..., ge=0, description="Character count of largest (prompt + context) passed to any single agent")
    communication_flow: list[str] = Field(..., description="Ordered list of agent-to-agent handoff steps")


# ---------------------------------------------------------------------------
# Full workflow output
# ---------------------------------------------------------------------------

class WorkflowResult(BaseModel):
    workflow_name: str
    ticket_id: str
    sentiment: SentimentResult
    priority: PriorityResult
    category: CategoryResult
    resolution: ResolutionResult
    metrics: WorkflowMetrics


# ---------------------------------------------------------------------------
# ComparisonResponse — top-level API response (P2 renders this)
# ---------------------------------------------------------------------------

class ComparisonResponse(BaseModel):
    ticket_id: str
    traditional: WorkflowResult
    recursive: WorkflowResult
    token_reduction_pct: float = Field(..., description="% fewer tokens in recursive vs traditional")
    speedup_factor: float = Field(..., description="traditional_time / recursive_time")
    context_size_reduction_pct: float = Field(..., description="% reduction in max_context_size")