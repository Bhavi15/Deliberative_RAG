"""Stage 1 — Query Analysis.

Decomposes the raw user query into focused sub-queries using an LLM call.
Falls back to returning the original query as the sole sub-query when
the LLM output cannot be parsed.
"""

from __future__ import annotations

import json
import re

import structlog

from src.utils.llm import LLMClient
from src.utils.prompts import load_prompt

log = structlog.get_logger()


def decompose_query(query_text: str, llm: LLMClient) -> list[str]:
    """Call the LLM to decompose a query string into sub-queries.

    Args:
        query_text: The raw user query.
        llm: Configured LLMClient instance.

    Returns:
        List of sub-query strings (at least one).
    """
    prompt = load_prompt("query_analysis", query=query_text)
    raw = llm.invoke(prompt)

    sub_queries = parse_sub_queries(raw)
    if sub_queries:
        return sub_queries

    # Retry once
    log.warning("query_decomposition_retry", query=query_text[:60])
    raw_retry = llm.invoke(prompt)
    sub_queries_retry = parse_sub_queries(raw_retry)

    if sub_queries_retry:
        return sub_queries_retry

    # Fallback: use original query as-is
    log.warning("query_decomposition_fallback", query=query_text[:60])
    return [query_text]


def parse_sub_queries(llm_output: str) -> list[str]:
    """Parse structured LLM output into a list of sub-query strings.

    Args:
        llm_output: Raw text returned by the LLM.

    Returns:
        List of sub-query strings, or empty list on failure.
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", llm_output)
        cleaned = cleaned.replace("```", "")

        start = cleaned.find("[")
        if start == -1:
            return []

        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "[":
                depth += 1
            elif cleaned[i] == "]":
                depth -= 1
                if depth == 0:
                    arr = json.loads(cleaned[start : i + 1])
                    if isinstance(arr, list):
                        result = [str(s).strip() for s in arr if str(s).strip()]
                        return result
                    return []

        return []
    except (json.JSONDecodeError, ValueError):
        return []
