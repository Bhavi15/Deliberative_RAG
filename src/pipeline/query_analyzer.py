"""Stage 1 — Query Analysis pipeline node.

Decomposes the raw user query into focused sub-queries using an LLM call.
"""

from src.models import Query, SubQuery
from src.pipeline.graph import PipelineState


def query_analysis_node(state: PipelineState) -> PipelineState:
    """LangGraph node: decompose the user query into sub-queries.

    Args:
        state: Current pipeline state containing the raw query text.

    Returns:
        Updated state with ``query.sub_queries`` populated.
    """
    pass


def decompose_query(query_text: str) -> Query:
    """Call the LLM to decompose a query string into sub-queries.

    Args:
        query_text: The raw user query.

    Returns:
        A Query instance with sub_queries filled in.
    """
    pass


def parse_sub_queries(llm_output: str, query_id: str) -> list[SubQuery]:
    """Parse structured LLM output into a list of SubQuery objects.

    Args:
        llm_output: Raw text returned by the LLM.
        query_id: Parent query ID to associate with each sub-query.

    Returns:
        List of SubQuery instances.
    """
    pass
