"""Tests for Stage 3 — Claim Extraction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.pipeline.claim_extractor import ClaimExtractor, _extract_json_array
from src.schemas import Claim, ClaimType


# -- Fixtures ----------------------------------------------------------------

SAMPLE_PASSAGE = (
    "3M reported total revenue of $32.8 billion in FY2023, "
    "a 4% decline from the prior year. Operating income fell to "
    "$5.6 billion, driven by restructuring charges. The company "
    "expects revenue growth of 2-3% in FY2024. Analysts consider "
    "the restructuring plan aggressive but necessary."
)

EXPECTED_LLM_RESPONSE = json.dumps([
    {
        "text": "3M reported total revenue of $32.8 billion in FY2023.",
        "claim_type": "data_point",
        "temporal_marker": "FY2023",
        "confidence_in_extraction": 0.95,
    },
    {
        "text": "3M's total revenue declined 4% from the prior year.",
        "claim_type": "data_point",
        "temporal_marker": "FY2023",
        "confidence_in_extraction": 0.93,
    },
    {
        "text": "3M's operating income fell to $5.6 billion due to restructuring charges.",
        "claim_type": "data_point",
        "temporal_marker": None,
        "confidence_in_extraction": 0.92,
    },
    {
        "text": "3M expects revenue growth of 2-3% in FY2024.",
        "claim_type": "forecast",
        "temporal_marker": "FY2024",
        "confidence_in_extraction": 0.90,
    },
    {
        "text": "Analysts consider 3M's restructuring plan aggressive but necessary.",
        "claim_type": "opinion",
        "temporal_marker": None,
        "confidence_in_extraction": 0.85,
    },
])


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a mock LLMClient that returns the expected JSON."""
    llm = MagicMock()
    llm.invoke.return_value = EXPECTED_LLM_RESPONSE
    return llm


@pytest.fixture
def extractor(mock_llm: MagicMock) -> ClaimExtractor:
    """Return a ClaimExtractor wired to the mock LLM."""
    return ClaimExtractor(mock_llm)


# -- Tests: core extraction --------------------------------------------------


def test_extract_claims_returns_at_least_3(extractor: ClaimExtractor) -> None:
    """Extractor should return at least 3 claims from a passage with 4-5 facts."""
    claims = extractor.extract_claims(
        SAMPLE_PASSAGE,
        {"source_doc_id": "doc_3m_2023"},
    )
    assert len(claims) >= 3


def test_extract_claims_returns_claim_objects(extractor: ClaimExtractor) -> None:
    """Every item returned should be a Claim instance."""
    claims = extractor.extract_claims(
        SAMPLE_PASSAGE,
        {"source_doc_id": "doc_3m_2023"},
    )
    for c in claims:
        assert isinstance(c, Claim)


def test_claim_types_are_valid_enums(extractor: ClaimExtractor) -> None:
    """Every claim_type must be a valid ClaimType enum value."""
    claims = extractor.extract_claims(
        SAMPLE_PASSAGE,
        {"source_doc_id": "doc_3m_2023"},
    )
    valid = {t.value for t in ClaimType}
    for c in claims:
        assert c.claim_type in valid, f"Invalid claim_type: {c.claim_type}"


def test_claims_have_source_doc_id(extractor: ClaimExtractor) -> None:
    """Each claim should carry the source_doc_id it was extracted from."""
    claims = extractor.extract_claims(
        SAMPLE_PASSAGE,
        {"source_doc_id": "doc_3m_2023"},
    )
    for c in claims:
        assert c.source_doc_id == "doc_3m_2023"


def test_claims_have_unique_ids(extractor: ClaimExtractor) -> None:
    """Each claim should have a unique claim_id."""
    claims = extractor.extract_claims(
        SAMPLE_PASSAGE,
        {"source_doc_id": "doc_3m_2023"},
    )
    ids = [c.claim_id for c in claims]
    assert len(ids) == len(set(ids))


def test_temporal_markers_extracted(extractor: ClaimExtractor) -> None:
    """Claims with temporal references should have temporal_marker set."""
    claims = extractor.extract_claims(
        SAMPLE_PASSAGE,
        {"source_doc_id": "doc_3m_2023"},
    )
    markers = [c.temporal_marker for c in claims if c.temporal_marker is not None]
    assert len(markers) >= 1, "Expected at least one temporal marker"


def test_confidence_within_bounds(extractor: ClaimExtractor) -> None:
    """confidence_in_extraction must be in [0, 1]."""
    claims = extractor.extract_claims(
        SAMPLE_PASSAGE,
        {"source_doc_id": "doc_3m_2023"},
    )
    for c in claims:
        assert 0.0 <= c.confidence_in_extraction <= 1.0


# -- Tests: malformed JSON handling ------------------------------------------


def test_retry_on_malformed_json_then_success() -> None:
    """If the first LLM call returns garbage, the extractor should retry once."""
    llm = MagicMock()
    llm.invoke.side_effect = [
        "This is not JSON at all",  # first call fails
        EXPECTED_LLM_RESPONSE,       # retry succeeds
    ]
    ext = ClaimExtractor(llm)
    claims = ext.extract_claims("Some passage.", {"source_doc_id": "doc1"})
    assert len(claims) >= 3
    assert llm.invoke.call_count == 2


def test_retry_on_malformed_json_both_fail() -> None:
    """If both calls return garbage, return an empty list (no crash)."""
    llm = MagicMock()
    llm.invoke.return_value = "totally invalid {{{}}}"
    ext = ClaimExtractor(llm)
    claims = ext.extract_claims("Some passage.", {"source_doc_id": "doc1"})
    assert claims == []
    assert llm.invoke.call_count == 2


# -- Tests: batch extraction -------------------------------------------------


def test_extract_claims_batch(extractor: ClaimExtractor) -> None:
    """extract_claims_batch should return one list per input passage."""
    passages = [
        {"text": SAMPLE_PASSAGE, "source_doc_id": "doc1"},
        {"text": "", "source_doc_id": "doc2"},
        {"text": "Another passage.", "source_doc_id": "doc3"},
    ]
    results = extractor.extract_claims_batch(passages)
    assert len(results) == 3
    assert len(results[0]) >= 3    # real passage
    assert results[1] == []         # empty text
    assert isinstance(results[2], list)


# -- Tests: JSON parsing helper ----------------------------------------------


def test_extract_json_array_plain() -> None:
    """Should parse a plain JSON array string."""
    result = _extract_json_array('[{"text": "hello"}]')
    assert result == [{"text": "hello"}]


def test_extract_json_array_with_markdown_fences() -> None:
    """Should strip markdown code fences around JSON."""
    raw = '```json\n[{"text": "hello"}]\n```'
    result = _extract_json_array(raw)
    assert result == [{"text": "hello"}]


def test_extract_json_array_with_leading_prose() -> None:
    """Should ignore leading prose and extract the array."""
    raw = 'Here are the claims:\n[{"text": "hello"}]'
    result = _extract_json_array(raw)
    assert result == [{"text": "hello"}]


def test_extract_json_array_no_array_raises() -> None:
    """Should raise ValueError when no JSON array is present."""
    with pytest.raises(ValueError, match="No JSON array"):
        _extract_json_array("no array here")
