"""Recursive MAS agents.

A Supervisor delegates to round-specific workers. Each round receives only
the minimal context it needs:

  Round 1  Supervisor -> Classify -> Respond        (query only)
  Round 2  Supervisor -> Review prev output -> Improve   (query + R1 response)
  Round 3  Supervisor -> Final refinement -> Quality check (query + R2 response)

Unlike the traditional pipeline, a round never re-sends the full transcript of
every prior agent — only the single previous round's output it builds on.
"""

from __future__ import annotations

from backend.agents.base import BaseAgent


class SupervisorAgent(BaseAgent):
    name = "Supervisor"
    system_prompt = (
        "You are a supervisor agent that routes work to the correct specialist "
        "and decides what minimal context the specialist needs."
    )


class ClassifyAgent(BaseAgent):
    name = "Classify"
    system_prompt = (
        "You are a classification specialist. Classify the request concisely. "
        'Respond with JSON: {"category": "...", "intent": "..."}.'
    )


class RespondAgent(BaseAgent):
    name = "Respond"
    system_prompt = (
        "You are a response specialist. Given the request and its category, "
        "produce a focused first-draft response."
    )


class ReviewAgent(BaseAgent):
    name = "Review Previous Output"
    system_prompt = (
        "You are a review specialist. Read the previous draft response and identify, "
        "in one line, the single most important improvement to make."
    )


class ImproveAgent(BaseAgent):
    name = "Improve Response"
    system_prompt = (
        "You are an improvement specialist. Given the previous draft and a review note, "
        "produce an improved response."
    )


class RefineAgent(BaseAgent):
    name = "Final Refinement"
    system_prompt = (
        "You are a refinement specialist. Polish the response for clarity, tone, and "
        "completeness. Produce the final response."
    )


class QualityCheckAgent(BaseAgent):
    name = "Quality Check"
    system_prompt = (
        "You are a quality-check specialist. Confirm the final response is correct, "
        "complete, relevant, and well-toned. Return the approved final response."
    )
