"""LangGraph node functions — one per pipeline stage.

Each node receives the current DeliberativeRAGState, delegates to the
corresponding pipeline module, and returns a partial state update dict.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import structlog

from src.agent.state import DeliberativeRAGState
from src.pipeline.claim_extractor import ClaimExtractor
from src.pipeline.conflict_graph import ConflictGraphBuilder
from src.pipeline.query_analyzer import decompose_query
from src.pipeline.retriever import merge_and_deduplicate, retrieve_for_sub_query
from src.pipeline.scorer import ClaimScorer
from src.pipeline.synthesizer import DeliberationSynthesizer
from src.schemas import Claim, ConflictEdge, ScoredClaim
from src.utils.llm import LLMClient
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Node 1 — Query Analysis
# ---------------------------------------------------------------------------

def analyze_query_node(
    state: DeliberativeRAGState,
    *,
    llm: LLMClient,
) -> dict[str, Any]:
    """Decompose the user query into focused sub-queries.

    Args:
        state: Current agent state with ``raw_query`` set.
        llm: LLM client for query decomposition.

    Returns:
        Partial state update with ``structured_query`` populated.
    """
    try:
        sub_queries = decompose_query(state["raw_query"], llm)
        log.info("query_analyzed", sub_queries=len(sub_queries))
        return {"structured_query": sub_queries}
    except Exception as exc:
        log.error("query_analysis_failed", error=str(exc))
        return {
            "structured_query": [state["raw_query"]],
            "error_log": state.get("error_log", []) + [f"query_analysis: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 2 — Retrieval
# ---------------------------------------------------------------------------

def retrieve_documents_node(
    state: DeliberativeRAGState,
    *,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 10,
) -> dict[str, Any]:
    """Retrieve relevant passages for all sub-queries.

    Args:
        state: Current agent state with ``structured_query`` populated.
        qdrant: Qdrant manager for vector search.
        embedder: Embedding model for encoding queries.
        top_k: Max results per sub-query.

    Returns:
        Partial state update with ``retrieved_documents`` and retrieval
        control fields.
    """
    try:
        passage_groups: list[list[dict]] = []
        for sq in state["structured_query"]:
            hits = retrieve_for_sub_query(sq, qdrant, embedder, top_k=top_k)
            passage_groups.append(hits)

        existing = state.get("retrieved_documents", [])
        new_docs = merge_and_deduplicate(passage_groups)

        # Merge with any docs from previous rounds
        all_docs = merge_and_deduplicate([existing, new_docs])
        rounds = state.get("retrieval_rounds", 0) + 1

        needs_more = rounds < 3 and len(all_docs) < 3

        log.info("retrieval_done", total_docs=len(all_docs), round=rounds, needs_more=needs_more)
        return {
            "retrieved_documents": all_docs,
            "retrieval_rounds": rounds,
            "needs_more_retrieval": needs_more,
        }
    except Exception as exc:
        log.error("retrieval_failed", error=str(exc))
        return {
            "retrieved_documents": state.get("retrieved_documents", []),
            "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
            "needs_more_retrieval": False,
            "error_log": state.get("error_log", []) + [f"retrieval: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 3 — Claim Extraction
# ---------------------------------------------------------------------------

def extract_claims_node(
    state: DeliberativeRAGState,
    *,
    llm: LLMClient,
) -> dict[str, Any]:
    """Extract atomic claims from retrieved passages.

    Args:
        state: Current agent state with ``retrieved_documents`` populated.
        llm: LLM client for claim extraction.

    Returns:
        Partial state update with ``extracted_claims`` populated.
    """
    try:
        extractor = ClaimExtractor(llm)
        passages = [
            {
                "text": doc.get("text", ""),
                "source_doc_id": str(doc.get("id", doc.get("record_id", "unknown"))),
            }
            for doc in state["retrieved_documents"]
            if doc.get("text", "").strip()
        ]

        # Single batched call: combine all passages into one prompt
        all_claims = extractor.extract_claims_combined(
            passages, query=state["raw_query"],
        )

        log.info("claims_extracted", count=len(all_claims))
        return {"extracted_claims": all_claims}
    except Exception as exc:
        log.error("claim_extraction_failed", error=str(exc))
        return {
            "extracted_claims": [],
            "error_log": state.get("error_log", []) + [f"claim_extraction: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 4 — Conflict Graph Construction
# ---------------------------------------------------------------------------

def build_conflict_graph_node(
    state: DeliberativeRAGState,
    *,
    llm: LLMClient,
    embedder: EmbeddingModel,
) -> dict[str, Any]:
    """Build the conflict graph from extracted claims.

    Args:
        state: Current agent state with ``extracted_claims`` populated.
        llm: LLM client for pairwise classification.
        embedder: Embedding model for claim clustering.

    Returns:
        Partial state update with ``conflict_graph`` (serialized).
    """
    try:
        builder = ConflictGraphBuilder(llm, embedder)
        graph = builder.build_graph(state["extracted_claims"])

        # Serialize graph for state transport
        serialized = _serialize_graph(graph)
        log.info("conflict_graph_built", nodes=graph.number_of_nodes(), edges=graph.number_of_edges())
        return {"conflict_graph": serialized}
    except Exception as exc:
        log.error("conflict_graph_failed", error=str(exc))
        return {
            "conflict_graph": {"nodes": [], "edges": []},
            "error_log": state.get("error_log", []) + [f"conflict_graph: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 5 — Scoring
# ---------------------------------------------------------------------------

def score_claims_node(
    state: DeliberativeRAGState,
) -> dict[str, Any]:
    """Compute composite trustworthiness scores for all claims.

    Args:
        state: Current agent state with ``extracted_claims`` and
            ``conflict_graph``.

    Returns:
        Partial state update with ``scored_claims`` populated.
    """
    try:
        graph = _deserialize_graph(state["conflict_graph"], state["extracted_claims"])

        # Build metadata from retrieved documents
        doc_metadata: dict[str, dict] = {}
        for doc in state.get("retrieved_documents", []):
            doc_id = str(doc.get("id", ""))
            doc_metadata[doc_id] = {
                "source_type": doc.get("source_type", ""),
                "publication_date": doc.get("publication_date"),
            }

        scorer = ClaimScorer()
        scored = scorer.score_all_claims(
            state["extracted_claims"], graph, metadata=doc_metadata,
        )

        log.info("claims_scored", count=len(scored))
        return {"scored_claims": scored}
    except Exception as exc:
        log.error("scoring_failed", error=str(exc))
        return {
            "scored_claims": [],
            "error_log": state.get("error_log", []) + [f"scoring: {exc}"],
        }


# ---------------------------------------------------------------------------
# Node 6 — Synthesis
# ---------------------------------------------------------------------------

def synthesize_answer_node(
    state: DeliberativeRAGState,
    *,
    llm: LLMClient,
) -> dict[str, Any]:
    """Generate the final answer with reasoning trace.

    Args:
        state: Current agent state with ``scored_claims`` and ``conflict_graph``.
        llm: LLM client for answer synthesis.

    Returns:
        Partial state update with ``final_answer`` populated.
    """
    try:
        graph = _deserialize_graph(state["conflict_graph"], state["extracted_claims"])
        synthesizer = DeliberationSynthesizer(llm)
        result = synthesizer.synthesize(
            state["raw_query"], state["scored_claims"], graph,
        )
        log.info("synthesis_done", confidence=result.confidence.value)
        return {"final_answer": result}
    except Exception as exc:
        log.error("synthesis_failed", error=str(exc))
        from src.schemas import ConfidenceLevel, DeliberationResult
        return {
            "final_answer": DeliberationResult(
                query=state["raw_query"],
                answer="Pipeline encountered an error during synthesis.",
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.0,
                reasoning_trace=[f"Error: {exc}"],
                source_attribution=[],
                conflict_summary="",
            ),
            "error_log": state.get("error_log", []) + [f"synthesis: {exc}"],
        }


# ---------------------------------------------------------------------------
# Conditional edge: should we retrieve more?
# ---------------------------------------------------------------------------

def should_retrieve_more(state: DeliberativeRAGState) -> str:
    """Decide whether the retrieval loop should continue.

    Returns ``"retrieve"`` if more docs are needed, ``"extract"`` otherwise.

    Criteria:
    - retrieval_rounds < 3
    - fewer than 3 documents retrieved so far

    Args:
        state: Current agent state.

    Returns:
        Next node name: ``"retrieve"`` or ``"extract"``.
    """
    if state.get("needs_more_retrieval", False):
        return "retrieve"
    return "extract"


# ---------------------------------------------------------------------------
# Graph serialization helpers
# ---------------------------------------------------------------------------

def _serialize_graph(graph: nx.DiGraph) -> dict[str, Any]:
    """Serialize a NetworkX DiGraph into a JSON-safe dict.

    Args:
        graph: The conflict graph.

    Returns:
        Dict with ``nodes`` (list of claim_id) and ``edges``
        (list of serialized ConflictEdge dicts).
    """
    nodes = list(graph.nodes)
    edges = []
    for u, v, data in graph.edges(data=True):
        edge_obj = data.get("edge")
        if edge_obj is not None:
            edges.append(edge_obj.model_dump())
    return {"nodes": nodes, "edges": edges}


def _deserialize_graph(
    serialized: dict[str, Any],
    claims: list[Claim],
) -> nx.DiGraph:
    """Reconstruct a NetworkX DiGraph from serialized state.

    Args:
        serialized: Dict with ``nodes`` and ``edges`` keys.
        claims: Claim objects to attach as node data.

    Returns:
        Reconstructed DiGraph.
    """
    claim_map = {c.claim_id: c for c in claims}
    g = nx.DiGraph()

    for node_id in serialized.get("nodes", []):
        claim = claim_map.get(node_id)
        if claim is not None:
            g.add_node(node_id, claim=claim)
        else:
            g.add_node(node_id)

    for edge_data in serialized.get("edges", []):
        edge = ConflictEdge(**edge_data)
        g.add_edge(
            edge.source_claim_id,
            edge.target_claim_id,
            edge=edge,
            relation=edge.relation.value,
        )

    return g
