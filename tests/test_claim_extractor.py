"""Tests for Stage 3 — Claim Extraction."""

import pytest

from src.models import Claim, Passage
from src.pipeline.claim_extraction import parse_claims


def test_parse_claims_returns_claim_objects() -> None:
    """parse_claims should return a non-empty list of Claim objects."""
    pass


def test_parse_claims_preserves_passage_provenance(sample_passage: Passage) -> None:
    """Each returned Claim should reference the source passage and document."""
    pass


def test_parse_claims_handles_empty_llm_output(sample_passage: Passage) -> None:
    """parse_claims should return an empty list when the LLM returns nothing."""
    pass
