"""Stage 4 — Conflict Graph Construction pipeline node.

Compares claim pairs with an LLM classifier and builds a typed NetworkX
graph with supports / contradicts / supersedes edges.
"""

import networkx as nx

from src.models import Claim, ConflictEdge
from src.pipeline.graph import PipelineState


def conflict_graph_node(state: PipelineState) -> PipelineState:
    """LangGraph node: classify claim relationships and build the conflict graph.

    Args:
        state: Current pipeline state with populated claims.

    Returns:
        Updated state with ``conflict_edges`` populated.
    """
    pass


def classify_claim_pair(claim_a: Claim, claim_b: Claim) -> ConflictEdge:
    """Call the LLM to classify the relationship between two claims.

    Args:
        claim_a: The source claim.
        claim_b: The target claim.

    Returns:
        A ConflictEdge describing the typed relationship.
    """
    pass


def build_conflict_graph(claims: list[Claim], edges: list[ConflictEdge]) -> nx.DiGraph:
    """Construct a NetworkX directed graph from claims and conflict edges.

    Args:
        claims: All extracted claims (become graph nodes).
        edges: Classified relationships (become graph edges).

    Returns:
        A directed NetworkX graph with claim nodes and typed edges.
    """
    pass


def select_candidate_pairs(claims: list[Claim]) -> list[tuple[Claim, Claim]]:
    """Heuristically select claim pairs worth classifying (avoids O(n²) LLM calls).

    Args:
        claims: Full list of extracted claims.

    Returns:
        Filtered list of (claim_a, claim_b) pairs to classify.
    """
    pass
