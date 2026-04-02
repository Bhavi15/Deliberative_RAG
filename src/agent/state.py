"""LangGraph pipeline state schema.

Defines the shared mutable state TypedDict that is threaded through every
node in the deliberative RAG workflow graph.
"""

from typing import TypedDict

from src.schemas import Answer, Claim, ConflictEdge, Passage, Query, ScoredClaim


class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    query: Query
    passages: list[Passage]
    claims: list[Claim]
    conflict_edges: list[ConflictEdge]
    scored_claims: list[ScoredClaim]
    answer: Answer | None
