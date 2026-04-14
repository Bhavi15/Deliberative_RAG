"""Ollama LLM client with light/heavy model support.

Provides two model tiers so compute-heavy stages (synthesis, claim extraction)
use a larger model while frequent/simple stages (query analysis, conflict
classification) use a smaller, faster one.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage


class LLMClient:
    """Thin wrapper around ChatOllama with project-wide defaults."""

    def __init__(self, model: str, temperature: float, max_tokens: int, base_url: str = "http://localhost:11434") -> None:
        """Initialise the LLM client.

        Args:
            model: Ollama model name (e.g. ``"llama3.1:8b"``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            base_url: Ollama server URL.
        """
        self._llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
            base_url=base_url,
        )
        self.model = model

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


@lru_cache(maxsize=4)
def get_llm(weight: str = "heavy") -> LLMClient:
    """Return a cached LLMClient built from application settings.

    Args:
        weight: ``"heavy"`` for complex tasks (synthesis, claim extraction)
            or ``"light"`` for simple tasks (query analysis, conflict
            classification).

    Returns:
        Cached LLMClient instance for the requested tier.
    """
    from config.settings import get_settings
    settings = get_settings()

    if weight == "light":
        model = settings.llm_model_light
    else:
        model = settings.llm_model_heavy

    return LLMClient(
        model=model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        base_url=settings.ollama_base_url,
    )
