"""Tests for Stage 5 — Temporal-Authority Scoring."""

import pytest

from src.models import Claim, ScoredClaim
from src.pipeline.scoring import compute_authority_score, compute_graph_score, compute_temporal_score, score_claim


def test_score_claim_composite_within_bounds(sample_claim: Claim) -> None:
    """Composite score should always be in [0, 1]."""
    pass


def test_compute_temporal_score_recent_is_higher() -> None:
    """A more recent document should yield a higher temporal score."""
    pass


def test_compute_temporal_score_unknown_date_returns_midpoint() -> None:
    """A claim with no publication date should return 0.5."""
    pass


def test_compute_authority_score_reflects_document_value(sample_claim: Claim) -> None:
    """Authority score should match the parent document's authority_score."""
    pass


def test_compute_graph_score_no_edges_returns_midpoint(sample_claim: Claim) -> None:
    """A claim with no graph edges should return 0.5."""
    pass
