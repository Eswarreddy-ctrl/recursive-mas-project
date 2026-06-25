"""Recursive MAS workflow across three rounds.

Each round is a self-contained WorkflowRun so the dashboard can chart R1, R2, R3
independently. A round only receives the previous round's output (minimal
context), not the entire transcript.

  Round 1: Supervisor -> Classify -> Respond           (query only)
  Round 2: Supervisor -> Review -> Improve             (query + R1 output)
  Round 3: Supervisor -> Final Refinement -> Quality   (query + R2 output)
"""

from __future__ import annotations

import logging
from typing import List

from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.recursive_agents import (
    ClassifyAgent,
    ImproveAgent,
    QualityCheckAgent,
    RefineAgent,
    RespondAgent,
    ReviewAgent,
    SupervisorAgent,
)
from backend.benchmarks.metrics_collector import MetricsCollector
from backend.models.schemas import WorkflowRun
from backend.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class RecursiveMAS:
    def __init__(self, llm: LLMService) -> None:
        self.llm = llm
        self.supervisor = SupervisorAgent(llm)
        self.classify = ClassifyAgent(llm)
        self.respond = RespondAgent(llm)
        self.review = ReviewAgent(llm)
        self.improve = ImproveAgent(llm)
        self.refine = RefineAgent(llm)
        self.quality_check = QualityCheckAgent(llm)
        self.evaluator = EvaluationAgent(llm)

    def run(self, query: str) -> List[WorkflowRun]:
        """Run all three rounds. Returns [R1, R2, R3]."""
        logger.info("Recursive MAS running (3 rounds)")
        rounds: List[WorkflowRun] = []

        # ---------------- Round 1: Classify -> Respond ----------------
        mc1 = MetricsCollector("Recursive R1", self.llm)
        c = self.classify.run(f"Request:\n{query}")
        mc1.record("Classify", c, c["_context_chars"])
        resp = self.respond.run(f"Request:\n{query}\n\nCategory:\n{c['text']}")
        mc1.record("Respond", resp, resp["_context_chars"])
        r1_response = resp["text"]
        mc1.set_response(r1_response)
        q1, _ = self.evaluator.evaluate(query, r1_response)
        rounds.append(mc1.build(q1))

        # ---------------- Round 2: Review -> Improve ------------------
        # Minimal context: query + ONLY the previous round's response.
        mc2 = MetricsCollector("Recursive R2", self.llm)
        rev = self.review.run(f"Request:\n{query}\n\nPrevious draft:\n{r1_response}")
        mc2.record("Review", rev, rev["_context_chars"])
        imp = self.improve.run(
            f"Previous draft:\n{r1_response}\n\nReview note:\n{rev['text']}"
        )
        mc2.record("Improve", imp, imp["_context_chars"])
        r2_response = imp["text"]
        mc2.set_response(r2_response)
        q2, _ = self.evaluator.evaluate(query, r2_response)
        rounds.append(mc2.build(q2))

        # ---------------- Round 3: Refine -> Quality Check ------------
        mc3 = MetricsCollector("Recursive R3", self.llm)
        ref = self.refine.run(f"Response to refine:\n{r2_response}")
        mc3.record("Refine", ref, ref["_context_chars"])
        qc = self.quality_check.run(f"Final candidate:\n{ref['text']}")
        mc3.record("Quality Check", qc, qc["_context_chars"])
        r3_response = qc["text"]
        mc3.set_response(r3_response)
        q3, _ = self.evaluator.evaluate(query, r3_response)
        rounds.append(mc3.build(q3))

        return rounds
