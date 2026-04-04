"""LangGraph pipeline state schema.

Defines the shared mutable state TypedDict that is threaded through every
node in the deliberative RAG workflow graph.
"""

from __future__ import annotations

from typing import Any, TypedDict

from src.schemas import Claim, ConflictEdge, DeliberationResult, ScoredClaim


class DeliberativeRAGState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    # Stage 1 — Query Analysis
    raw_query: str
    structured_query: list[str]  # sub-queries from decomposition

    # Stage 2 — Retrieval
    retrieved_documents: list[dict[str, Any]]

    # Stage 3 — Claim Extraction
    extracted_claims: list[Claim]

    # Stage 4 — Conflict Graph (serialized as edge list for state transport)
    conflict_graph: dict[str, Any]

    # Stage 5 — Scoring
    scored_claims: list[ScoredClaim]

    # Stage 6 — Synthesis
    final_answer: DeliberationResult | None

    # Control flow
    needs_more_retrieval: bool
    retrieval_rounds: int
    error_log: list[str]
