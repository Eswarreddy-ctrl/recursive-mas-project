"""Base agent definition."""

from __future__ import annotations

from backend.services.llm_service import LLMService


class BaseAgent:
    """Common base for all agents."""

    name: str = "Agent"
    system_prompt: str = "You are a helpful assistant."

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    def run(self, content: str, max_tokens: int = 512) -> dict:
        """Run the agent on the given content. Returns the LLMService dict."""
        result = self.llm.generate(self.system_prompt, content, max_tokens=max_tokens)
        return result.as_dict() | {"_context_chars": len(self.system_prompt) + len(content)}
