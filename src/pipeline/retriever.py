"""Stage 2 — Multi-Strategy Retrieval pipeline node.

Executes vector search and metadata-filtered search against Qdrant for each
sub-query, then merges and deduplicates the resulting passages.
"""

from src.models import Passage, SubQuery
from src.pipeline.graph import PipelineState


def retrieval_node(state: PipelineState) -> PipelineState:
    """LangGraph node: retrieve passages for all sub-queries.

    Args:
        state: Current pipeline state with populated sub-queries.

    Returns:
        Updated state with ``passages`` populated.
    """
    pass


def retrieve_for_sub_query(sub_query: SubQuery) -> list[Passage]:
    """Run all retrieval strategies for a single sub-query.

    Args:
        sub_query: The sub-query to retrieve passages for.

    Returns:
        List of retrieved Passage objects.
    """
    pass


def merge_and_deduplicate(passages: list[list[Passage]]) -> list[Passage]:
    """Merge passage lists from multiple sub-queries and remove duplicates.

    Args:
        passages: Per-sub-query passage lists.

    Returns:
        Flat, deduplicated list of passages sorted by retrieval score.
    """
    pass
