"""LangGraph workflow graph construction.

Wires all six pipeline stage nodes into a compiled LangGraph StateGraph
ready for synchronous invocation.  Includes a conditional retrieval loop
that re-fetches when fewer than 3 documents are found (up to 3 rounds).

Two LLM tiers are supported so heavy stages use a larger model while
frequent/simple stages use a smaller, faster one:

- **heavy** (synthesis, claim extraction): ``llm_heavy``
- **light** (query analysis, conflict classification): ``llm_light``
"""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    analyze_query_node,
    build_conflict_graph_node,
    extract_claims_node,
    retrieve_documents_node,
    score_claims_node,
    should_retrieve_more,
    synthesize_answer_node,
)
from src.agent.state import DeliberativeRAGState
from src.schemas import DeliberationResult
from src.utils.llm import LLMClient
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager


def build_graph(
    llm_heavy: LLMClient,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 10,
    llm_light: LLMClient | None = None,
) -> Any:
    """Construct and compile the deliberative RAG LangGraph workflow.

    Topology::

        START → analyze → retrieve ─┐
                              ↑      │
                              └──────┤ (conditional: needs_more_retrieval?)
                                     ↓
                                  extract → conflict_graph → score → synthesize → END

    Args:
        llm_heavy: LLM client for heavy stages (claim extraction, synthesis).
        qdrant: Qdrant manager for retrieval.
        embedder: Embedding model for retrieval and graph building.
        top_k: Max results per sub-query retrieval.
        llm_light: Optional lighter LLM for simple stages (query analysis,
            conflict classification).  Falls back to *llm_heavy* if not given.

    Returns:
        Compiled LangGraph runnable.
    """
    if llm_light is None:
        llm_light = llm_heavy

    # Bind dependencies into node functions via partial
    analyze = partial(analyze_query_node, llm=llm_light)
    retrieve = partial(retrieve_documents_node, qdrant=qdrant, embedder=embedder, top_k=top_k)
    extract = partial(extract_claims_node, llm=llm_light)
    build_graph_node = partial(build_conflict_graph_node, llm=llm_light, embedder=embedder)
    synthesize = partial(synthesize_answer_node, llm=llm_heavy)

    workflow = StateGraph(DeliberativeRAGState)

    # Add nodes
    workflow.add_node("analyze", analyze)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("extract", extract)
    workflow.add_node("conflict_graph", build_graph_node)
    workflow.add_node("score", score_claims_node)
    workflow.add_node("synthesize", synthesize)

    # Wire edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "retrieve")

    # Conditional loop: after retrieval, check if more docs are needed
    workflow.add_conditional_edges(
        "retrieve",
        should_retrieve_more,
        {"retrieve": "retrieve", "extract": "extract"},
    )

    workflow.add_edge("extract", "conflict_graph")
    workflow.add_edge("conflict_graph", "score")
    workflow.add_edge("score", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def run_query(
    query_text: str,
    llm_heavy: LLMClient,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 10,
    llm_light: LLMClient | None = None,
) -> DeliberationResult:
    """Execute the full deliberative RAG pipeline for a single query.

    Args:
        query_text: Raw user query string.
        llm_heavy: LLM client for heavy stages.
        qdrant: Qdrant manager.
        embedder: Embedding model.
        top_k: Max results per sub-query.
        llm_light: Optional lighter LLM for simple stages.

    Returns:
        A fully populated DeliberationResult with reasoning trace and confidence.
    """
    app = build_graph(llm_heavy, qdrant, embedder, top_k=top_k, llm_light=llm_light)

    initial_state: DeliberativeRAGState = {
        "raw_query": query_text,
        "structured_query": [],
        "retrieved_documents": [],
        "extracted_claims": [],
        "conflict_graph": {"nodes": [], "edges": []},
        "scored_claims": [],
        "final_answer": None,
        "needs_more_retrieval": False,
        "retrieval_rounds": 0,
        "error_log": [],
    }

    final_state = app.invoke(initial_state)
    return final_state["final_answer"]


def run_query_full(
    query_text: str,
    llm_heavy: LLMClient,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 10,
    llm_light: LLMClient | None = None,
) -> dict:
    """Execute the full pipeline and return the complete final state.

    Like :func:`run_query` but exposes the entire state dict so callers
    can inspect intermediate outputs (e.g. ``conflict_graph`` for
    evaluation).

    Args:
        query_text: Raw user query string.
        llm_heavy: LLM client for heavy stages.
        qdrant: Qdrant manager.
        embedder: Embedding model.
        top_k: Max results per sub-query.
        llm_light: Optional lighter LLM for simple stages.

    Returns:
        The full final ``DeliberativeRAGState`` dict.
    """
    app = build_graph(llm_heavy, qdrant, embedder, top_k=top_k, llm_light=llm_light)

    initial_state: DeliberativeRAGState = {
        "raw_query": query_text,
        "structured_query": [],
        "retrieved_documents": [],
        "extracted_claims": [],
        "conflict_graph": {"nodes": [], "edges": []},
        "scored_claims": [],
        "final_answer": None,
        "needs_more_retrieval": False,
        "retrieval_rounds": 0,
        "error_log": [],
    }

    return app.invoke(initial_state)
