"""Traditional (flat pipeline) MAS agents.

Classifier -> Responder -> QA. Every agent receives the FULL accumulated
context (query + all prior outputs), which is the defining inefficiency of
the traditional architecture.
"""

from __future__ import annotations

from backend.agents.base import BaseAgent


class ClassifierAgent(BaseAgent):
    name = "Classifier Agent"
    system_prompt = (
        "You are a classifier agent for customer support. "
        "Classify the user's request. Respond with a JSON object: "
        '{"category": "...", "intent": "..."}.'
    )


class ResponderAgent(BaseAgent):
    name = "Responder Agent"
    system_prompt = (
        "You are a responder agent. Given the user request and its classification, "
        "write a clear, complete, helpful response to the customer."
    )


class QAAgent(BaseAgent):
    name = "QA Agent"
    system_prompt = (
        "You are a QA agent. Review the request, classification, and drafted response. "
        "Verify the response is accurate, complete, relevant, and well-toned. "
        "Return the final verified response."
    )
