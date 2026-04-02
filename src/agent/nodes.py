"""LangGraph node functions — one per pipeline stage.

Each node receives the current AgentState, delegates to the corresponding
pipeline module, and returns a partial state update dict.
"""

from src.agent.state import AgentState


def query_analysis_node(state: AgentState) -> dict:
    """Node 1: Decompose the user query into sub-queries.

    Args:
        state: Current agent state with ``query.text`` set.

    Returns:
        Partial state update with ``query.sub_queries`` populated.
    """
    pass


def retrieval_node(state: AgentState) -> dict:
    """Node 2: Retrieve relevant passages for all sub-queries.

    Args:
        state: Current agent state with sub-queries populated.

    Returns:
        Partial state update with ``passages`` populated.
    """
    pass


def claim_extraction_node(state: AgentState) -> dict:
    """Node 3: Extract atomic claims from retrieved passages.

    Args:
        state: Current agent state with passages populated.

    Returns:
        Partial state update with ``claims`` populated.
    """
    pass


def conflict_graph_node(state: AgentState) -> dict:
    """Node 4: Classify claim pairs and build the conflict graph.

    Args:
        state: Current agent state with claims populated.

    Returns:
        Partial state update with ``conflict_edges`` populated.
    """
    pass


def scoring_node(state: AgentState) -> dict:
    """Node 5: Compute composite trustworthiness scores for all claims.

    Args:
        state: Current agent state with claims and conflict edges.

    Returns:
        Partial state update with ``scored_claims`` populated.
    """
    pass


def synthesis_node(state: AgentState) -> dict:
    """Node 6: Generate the final answer with reasoning trace.

    Args:
        state: Current agent state with scored claims and conflict edges.

    Returns:
        Partial state update with ``answer`` populated.
    """
    pass
