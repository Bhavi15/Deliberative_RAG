"""Stage 2 — Multi-Strategy Retrieval pipeline node.

Executes vector search and metadata-filtered search against Qdrant for each
sub-query, then merges and deduplicates the resulting passages.
"""


def retrieval_node(state: dict) -> dict:
    """LangGraph node: retrieve passages for all sub-queries.

    Args:
        state: Current pipeline state with populated sub-queries.

    Returns:
        Partial state update with ``passages`` populated.
    """
    pass


def retrieve_for_sub_query(sub_query: str) -> list[dict]:
    """Run all retrieval strategies for a single sub-query.

    Args:
        sub_query: The sub-query text to retrieve passages for.

    Returns:
        List of passage dicts with ``"text"``, ``"score"``, and metadata.
    """
    pass


def merge_and_deduplicate(passage_groups: list[list[dict]]) -> list[dict]:
    """Merge passage lists from multiple sub-queries and remove duplicates.

    Args:
        passage_groups: Per-sub-query passage lists.

    Returns:
        Flat, deduplicated list of passage dicts sorted by score descending.
    """
    pass
