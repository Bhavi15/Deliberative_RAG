"""Tests for Stage 4 — Conflict Graph Construction."""

import pytest

from src.graph.builder import ConflictGraphBuilder
from src.models import Claim, ConflictEdge, EdgeType


def test_builder_adds_claim_nodes(sample_claim: Claim) -> None:
    """Adding a claim should create a corresponding node in the graph."""
    pass


def test_builder_adds_typed_edges(sample_claim: Claim) -> None:
    """Adding a ConflictEdge should create a directed edge with edge_type attribute."""
    pass


def test_builder_get_contradictions_filters_correctly() -> None:
    """get_contradictions should return only CONTRADICTS edges."""
    pass


def test_builder_get_supporting_claims() -> None:
    """get_supporting_claims should return only IDs connected via SUPPORTS edges."""
    pass


def test_select_candidate_pairs_limits_combinations() -> None:
    """select_candidate_pairs should not return all O(n²) pairs for large claim sets."""
    pass
