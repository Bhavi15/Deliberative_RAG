"""Baseline RAG implementation for comparison.

A standard retrieve-then-read RAG pipeline with no conflict detection,
scoring, or deliberative synthesis. Used as the comparison baseline
when evaluating the deliberative system.
"""

from src.schemas import DeliberationResult


def run_baseline(query_text: str) -> DeliberationResult:
    """Execute a naive RAG pipeline — retrieve top-k, concatenate, generate.

    Args:
        query_text: Raw user query string.

    Returns:
        DeliberationResult with empty reasoning trace and high confidence.
    """
    pass


def retrieve_top_k(query_text: str, k: int) -> list[str]:
    """Retrieve the top-k passage texts for a query via vector search.

    Args:
        query_text: User query.
        k: Number of passages to retrieve.

    Returns:
        List of passage text strings.
    """
    pass


def generate_answer(query_text: str, context_passages: list[str]) -> str:
    """Call the LLM with a naive context-stuffing prompt.

    Args:
        query_text: User query.
        context_passages: Retrieved passages to stuff into the prompt.

    Returns:
        Generated answer string.
    """
    pass
