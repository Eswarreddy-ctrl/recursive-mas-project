"""Evaluation Agent.

Scores a response on four criteria (0-10) and returns a QualityScore.
Works with live LLMs (parses JSON) and with the deterministic mock.
"""

from __future__ import annotations

import json
import logging
import re

from backend.agents.base import BaseAgent
from backend.models.schemas import QualityScore

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    name = "Evaluation Agent"
    system_prompt = (
        "You are an evaluation agent. Score the candidate response against the "
        "original query on four criteria, each 0-10:\n"
        "- correctness\n- completeness\n- relevance\n- tone\n"
        'Respond with ONLY JSON: {"correctness": x, "completeness": x, '
        '"relevance": x, "tone": x}.'
    )

    def evaluate(self, query: str, response: str) -> tuple[QualityScore, dict]:
        content = f"Query:\n{query}\n\nCandidate response:\n{response}"
        result = self.llm.generate(self.system_prompt, content, max_tokens=128)
        score = self._parse(result.text)
        meta = result.as_dict() | {"_context_chars": len(self.system_prompt) + len(content)}
        return score, meta

    @staticmethod
    def _parse(text: str) -> QualityScore:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            return QualityScore(
                correctness=float(data.get("correctness", 7.0)),
                completeness=float(data.get("completeness", 7.0)),
                relevance=float(data.get("relevance", 7.0)),
                tone=float(data.get("tone", 7.0)),
            )
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to parse evaluation (%s); using neutral score", e)
            return QualityScore(correctness=7, completeness=7, relevance=7, tone=7)
