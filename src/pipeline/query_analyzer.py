"""Stage 1 — Query Analysis pipeline node.

Decomposes the raw user query into focused sub-queries using an LLM call.
"""


def query_analysis_node(state: dict) -> dict:
    """LangGraph node: decompose the user query into sub-queries.

    Args:
        state: Current pipeline state containing the raw query text.

    Returns:
        Partial state update with ``sub_queries`` populated.
    """
    pass


def decompose_query(query_text: str) -> list[str]:
    """Call the LLM to decompose a query string into sub-queries.

    Args:
        query_text: The raw user query.

    Returns:
        List of sub-query strings.
    """
    pass


def parse_sub_queries(llm_output: str) -> list[str]:
    """Parse structured LLM output into a list of sub-query strings.

    Args:
        llm_output: Raw text returned by the LLM.

    Returns:
        List of sub-query strings.
    """
    pass
