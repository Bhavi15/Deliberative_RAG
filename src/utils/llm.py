"""LangChain-Anthropic LLM client and factory.

Centralises model configuration so all pipeline stages share a single,
consistently configured ChatAnthropic instance.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage


class LLMClient:
    """Thin wrapper around ChatAnthropic with project-wide defaults."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        """Initialise the LLM client.

        Args:
            model: Anthropic model ID (e.g. ``"claude-sonnet-4-6"``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
        """
        self._llm = ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def invoke(self, prompt: str) -> str:
        """Send a plain-text prompt and return the response text.

        Args:
            prompt: Fully formatted prompt string.

        Returns:
            Model response as a plain string.
        """
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def invoke_messages(self, messages: list[BaseMessage]) -> str:
        """Send a list of LangChain messages and return the response text.

        Args:
            messages: Ordered list of HumanMessage / SystemMessage objects.

        Returns:
            Model response as a plain string.
        """
        response = self._llm.invoke(messages)
        return response.content

    def invoke_structured(self, prompt: str, output_schema: type) -> object:
        """Invoke with structured output parsing via with_structured_output.

        Args:
            prompt: Fully formatted prompt string.
            output_schema: Pydantic model class to parse the response into.

        Returns:
            Parsed instance of output_schema.
        """
        structured_llm = self._llm.with_structured_output(output_schema)
        return structured_llm.invoke([HumanMessage(content=prompt)])


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """Return a cached LLMClient built from application settings.

    Returns:
        Singleton LLMClient instance.
    """
    from config.settings import get_settings
    settings = get_settings()
    return LLMClient(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
