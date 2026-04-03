"""LangGraph pipeline state schema.

Defines the shared mutable state TypedDict that is threaded through every
node in the deliberative RAG workflow graph.
"""

from typing import TypedDict

from src.schemas import Claim, ConflictEdge, DeliberationResult, ScoredClaim


class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    query: str
    sub_queries: list[str]
    passages: list[dict]
    claims: list[Claim]
    conflict_edges: list[ConflictEdge]
    scored_claims: list[ScoredClaim]
    result: DeliberationResult | None
