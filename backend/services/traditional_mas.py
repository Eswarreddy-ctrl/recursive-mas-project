"""Traditional MAS workflow (flat pipeline).

Classifier -> Responder -> QA, where each agent receives the full accumulated
context. Context grows at every stage.
"""

from __future__ import annotations

import logging

from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.traditional_agents import ClassifierAgent, QAAgent, ResponderAgent
from backend.benchmarks.metrics_collector import MetricsCollector
from backend.models.schemas import WorkflowRun
from backend.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class TraditionalMAS:
    def __init__(self, llm: LLMService) -> None:
        self.llm = llm
        self.classifier = ClassifierAgent(llm)
        self.responder = ResponderAgent(llm)
        self.qa = QAAgent(llm)
        self.evaluator = EvaluationAgent(llm)

    def run(self, query: str) -> WorkflowRun:
        logger.info("Traditional MAS running")
        mc = MetricsCollector("Traditional", self.llm)

        # Stage 1: Classifier — sees the query
        ctx1 = f"Request:\n{query}"
        r1 = self.classifier.run(ctx1)
        mc.record("Classifier", r1, r1["_context_chars"])
        classification = r1["text"]

        # Stage 2: Responder — sees query + classification (FULL context)
        ctx2 = f"Request:\n{query}\n\nClassification:\n{classification}"
        r2 = self.responder.run(ctx2)
        mc.record("Responder", r2, r2["_context_chars"])
        draft = r2["text"]

        # Stage 3: QA — sees query + classification + draft (FULL accumulated context)
        ctx3 = (
            f"Request:\n{query}\n\nClassification:\n{classification}\n\n"
            f"Drafted response:\n{draft}"
        )
        r3 = self.qa.run(ctx3)
        mc.record("QA", r3, r3["_context_chars"])
        final = r3["text"]

        mc.set_response(final)

        # Quality scoring
        quality, eval_meta = self.evaluator.evaluate(query, final)
        # Evaluation cost counts toward observability but not the pipeline's own
        # production cost; we keep it out of the workflow totals to compare
        # production workflows fairly. (Toggle by recording it instead.)

        return mc.build(quality)
