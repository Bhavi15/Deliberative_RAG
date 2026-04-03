"""LangGraph workflow graph construction.

Wires all six pipeline stage nodes into a compiled LangGraph StateGraph
ready for synchronous or async invocation.
"""

from langgraph.graph import StateGraph

from src.agent.state import AgentState
from src.schemas import DeliberationResult


def build_graph() -> StateGraph:
    """Construct and compile the deliberative RAG LangGraph workflow.

    Adds one node per pipeline stage and connects them in sequence:
    query_analysis → retrieval → claim_extraction → conflict_graph
    → scoring → synthesis.

    Returns:
        Compiled LangGraph StateGraph.
    """
    pass


def run_graph(query_text: str) -> DeliberationResult:
    """Execute the full deliberative RAG pipeline for a single query.

    Args:
        query_text: Raw user query string.

    Returns:
        A fully populated DeliberationResult with reasoning trace and confidence.
    """
    pass
