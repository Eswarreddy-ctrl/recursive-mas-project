"""Traditional Multi-Agent System workflow with sequential agent execution."""

import json
import logging

from agents.sentiment_agent import SentimentAgent
from agents.priority_agent import PriorityAgent
from agents.category_agent import CategoryAgent
from agents.resolution_agent import ResolutionAgent
from llm.gemini_service import GeminiService
from metrics.metrics_tracker import MetricsTracker
from metrics.timer import Timer
from models.schemas import AnalysisResult, WorkflowResult

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """You are a customer support analyst. Given the following ticket and analysis results, write a brief one-sentence summary.

Respond with ONLY a JSON object in this exact format:
{"summary": "<brief summary>"}

"""


class TraditionalWorkflow:
    """Traditional MAS: sequential pipeline where each agent receives
    the original ticket plus all previous agent outputs.

    Flow: Ticket → Sentiment → Priority → Category → Resolution
    """

    def __init__(self) -> None:
        self._llm = GeminiService()
        self._sentiment = SentimentAgent(self._llm)
        self._priority = PriorityAgent(self._llm)
        self._category = CategoryAgent(self._llm)
        self._resolution = ResolutionAgent(self._llm)

    def run(self, ticket: str) -> WorkflowResult:
        """Execute the traditional sequential workflow.

        Args:
            ticket: The customer support ticket text.

        Returns:
            WorkflowResult containing the analysis and metrics.
        """
        logger.info("Starting Traditional MAS workflow")
        tracker = MetricsTracker()
        timer = Timer()

        with timer:
            accumulated_context = ""

            # Step 1: Sentiment Analysis
            sentiment_result = self._sentiment.analyze(ticket, previous_outputs=accumulated_context)
            tracker.record_call(
                tokens=sentiment_result["tokens_used"],
                input_length=sentiment_result["input_length"],
                estimated=sentiment_result["estimated"],
                source="Ticket",
                target="Sentiment Agent",
            )
            accumulated_context += json.dumps(sentiment_result["output"]) + "\n"

            # Step 2: Priority Classification
            priority_result = self._priority.analyze(ticket, previous_outputs=accumulated_context)
            tracker.record_call(
                tokens=priority_result["tokens_used"],
                input_length=priority_result["input_length"],
                estimated=priority_result["estimated"],
                source="Sentiment Agent",
                target="Priority Agent",
            )
            accumulated_context += json.dumps(priority_result["output"]) + "\n"

            # Step 3: Category Classification
            category_result = self._category.analyze(ticket, previous_outputs=accumulated_context)
            tracker.record_call(
                tokens=category_result["tokens_used"],
                input_length=category_result["input_length"],
                estimated=category_result["estimated"],
                source="Priority Agent",
                target="Category Agent",
            )
            accumulated_context += json.dumps(category_result["output"]) + "\n"

            # Step 4: Resolution
            resolution_result = self._resolution.analyze(ticket, previous_outputs=accumulated_context)
            tracker.record_call(
                tokens=resolution_result["tokens_used"],
                input_length=resolution_result["input_length"],
                estimated=resolution_result["estimated"],
                source="Category Agent",
                target="Resolution Agent",
            )

            # Generate summary
            all_outputs = {
                **sentiment_result["output"],
                **priority_result["output"],
                **category_result["output"],
                **resolution_result["output"],
            }
            summary_context = f"Ticket:\n{ticket}\n\nAnalysis:\n{json.dumps(all_outputs, indent=2)}"
            summary_response = self._llm.generate(
                prompt=SUMMARY_PROMPT,
                context=summary_context,
            )
            tracker.record_call(
                tokens=summary_response["tokens_used"],
                input_length=summary_response["input_length"],
                estimated=summary_response["estimated"],
                source="Resolution Agent",
                target="Summary",
            )
            summary_text = summary_response["response"].get("summary", "Ticket analyzed and routed.")

        tracker.set_time(timer.elapsed)

        result = AnalysisResult(
            category=category_result["output"]["category"],
            priority=priority_result["output"]["priority"],
            sentiment=sentiment_result["output"]["sentiment"],
            recommended_team=resolution_result["output"]["recommended_team"],
            summary=summary_text,
        )

        logger.info("Traditional MAS workflow completed in %.4fs", timer.elapsed)
        return WorkflowResult(result=result, metrics=tracker.get_metrics())
