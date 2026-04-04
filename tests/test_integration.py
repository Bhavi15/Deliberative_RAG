"""End-to-end integration tests for the full deliberative RAG pipeline.

These tests exercise the complete 6-stage pipeline with a mocked LLM and
a mock Qdrant/embedder to avoid external dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import build_graph, run_query
from src.schemas import ConfidenceLevel, DeliberationResult


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

# Stage 1: query decomposition → returns a JSON array of sub-queries
QUERY_DECOMPOSITION_RESPONSE = json.dumps(["What is Kathy Saltzman's occupation?"])

# Stage 3: claim extraction → returns a JSON array of claims
CLAIM_EXTRACTION_RESPONSE = json.dumps([
    {
        "text": "Kathy Saltzman is a professional musician and flutist.",
        "claim_type": "fact",
        "temporal_marker": None,
        "confidence_in_extraction": 0.85,
    },
    {
        "text": "Kathy Saltzman served in the Minnesota Senate representing District 56.",
        "claim_type": "fact",
        "temporal_marker": None,
        "confidence_in_extraction": 0.92,
    },
])

# Stage 4: conflict classification → one CONTRADICTS pair
CONFLICT_CLASSIFICATION_RESPONSE = json.dumps({
    "relation": "CONTRADICTS",
    "confidence": 0.90,
    "reasoning": "One claims she is a musician; the other claims she is a politician — mutually exclusive.",
})

# Stage 6: synthesis → full JSON answer
SYNTHESIS_RESPONSE = json.dumps({
    "answer": "Kathy Saltzman is a politician who served in the Minnesota Senate.",
    "confidence": "high",
    "confidence_score": 0.88,
    "reasoning_trace": [
        "Two sources provide conflicting claims about her occupation.",
        "The politician claim has higher authority and graph support.",
        "Resolved in favour of 'politician'.",
    ],
    "source_attribution": [
        {"claim_id": "c2", "source_doc_id": "doc_counter", "relevance": 0.95},
    ],
    "conflict_summary": "Musician vs politician — resolved in favour of politician.",
})

# For the "no passages" scenario
SYNTHESIS_NO_DOCS_RESPONSE = json.dumps({
    "answer": "Unable to determine with available information.",
    "confidence": "low",
    "confidence_score": 0.15,
    "reasoning_trace": ["No relevant passages were retrieved."],
    "source_attribution": [],
    "conflict_summary": "",
})


def _make_mock_llm(responses: list[str] | None = None) -> MagicMock:
    """Create a mock LLM that cycles through staged responses.

    The default response sequence handles the standard ConflictQA flow:
    1. query decomposition (1 call)
    2. claim extraction — one call per passage (2 calls)
    3. conflict classification — one call per pair; 4 claims → C(4,2)=6 pairs (6 calls)
    4. synthesis (1 call)
    """
    if responses is None:
        responses = [
            QUERY_DECOMPOSITION_RESPONSE,       # analyze_query_node
            CLAIM_EXTRACTION_RESPONSE,           # extract_claims_node (passage 1)
            CLAIM_EXTRACTION_RESPONSE,           # extract_claims_node (passage 2)
            # conflict classification — 6 pairs from 4 claims
            CONFLICT_CLASSIFICATION_RESPONSE,
            CONFLICT_CLASSIFICATION_RESPONSE,
            CONFLICT_CLASSIFICATION_RESPONSE,
            CONFLICT_CLASSIFICATION_RESPONSE,
            CONFLICT_CLASSIFICATION_RESPONSE,
            CONFLICT_CLASSIFICATION_RESPONSE,
            SYNTHESIS_RESPONSE,                  # synthesize_answer_node
        ]
    llm = MagicMock()
    llm.invoke.side_effect = list(responses)
    return llm


def _make_mock_qdrant(passages: list[dict] | None = None) -> MagicMock:
    """Create a mock QdrantManager returning fixed passages."""
    if passages is None:
        passages = [
            {
                "id": "doc_param",
                "score": 0.75,
                "text": "Kathy Saltzman is a professional musician and a flutist.",
                "source_type": "blog",
                "record_id": "conflict_qa_001",
            },
            {
                "id": "doc_counter",
                "score": 0.85,
                "text": "Kathy Saltzman served in the Minnesota Senate representing District 56.",
                "source_type": "encyclopedic",
                "record_id": "conflict_qa_001",
            },
        ]
    qdrant = MagicMock()
    qdrant.search.return_value = passages
    return qdrant


def _make_mock_embedder() -> MagicMock:
    """Create a mock embedding model returning fixed 384-dim vectors."""
    embedder = MagicMock()
    # embed_text: return a 384-dim vector
    embedder.embed_text.return_value = [0.1] * 384
    # embed_batch: return one vector per input, with high similarity
    # so claims get clustered together
    embedder.embed_batch.side_effect = lambda texts, **kw: [[0.1 + i * 0.01] * 384 for i in range(len(texts))]
    return embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:

    def test_returns_deliberation_result(self) -> None:
        """run_query should return a well-formed DeliberationResult."""
        llm = _make_mock_llm()
        qdrant = _make_mock_qdrant()
        embedder = _make_mock_embedder()

        result = run_query(
            "What is Kathy Saltzman's occupation?",
            llm=llm, qdrant=qdrant, embedder=embedder,
        )

        assert isinstance(result, DeliberationResult)

    def test_answer_not_empty(self) -> None:
        """Pipeline should produce a non-empty answer."""
        llm = _make_mock_llm()
        qdrant = _make_mock_qdrant()
        embedder = _make_mock_embedder()

        result = run_query(
            "What is Kathy Saltzman's occupation?",
            llm=llm, qdrant=qdrant, embedder=embedder,
        )

        assert len(result.answer) > 0

    def test_confidence_within_bounds(self) -> None:
        """Answer confidence score must be in [0, 1]."""
        llm = _make_mock_llm()
        qdrant = _make_mock_qdrant()
        embedder = _make_mock_embedder()

        result = run_query(
            "What is Kathy Saltzman's occupation?",
            llm=llm, qdrant=qdrant, embedder=embedder,
        )

        assert 0.0 <= result.confidence_score <= 1.0
        assert result.confidence in ConfidenceLevel

    def test_reasoning_trace_not_empty(self) -> None:
        """Reasoning trace should contain at least one step."""
        llm = _make_mock_llm()
        qdrant = _make_mock_qdrant()
        embedder = _make_mock_embedder()

        result = run_query(
            "What is Kathy Saltzman's occupation?",
            llm=llm, qdrant=qdrant, embedder=embedder,
        )

        assert len(result.reasoning_trace) >= 1

    def test_conflict_summary_present(self) -> None:
        """Pipeline should surface conflict info when sources disagree."""
        llm = _make_mock_llm()
        qdrant = _make_mock_qdrant()
        embedder = _make_mock_embedder()

        result = run_query(
            "What is Kathy Saltzman's occupation?",
            llm=llm, qdrant=qdrant, embedder=embedder,
        )

        assert len(result.conflict_summary) > 0

    def test_query_preserved_in_result(self) -> None:
        """The original query should appear in the result."""
        llm = _make_mock_llm()
        qdrant = _make_mock_qdrant()
        embedder = _make_mock_embedder()

        query = "What is Kathy Saltzman's occupation?"
        result = run_query(
            query, llm=llm, qdrant=qdrant, embedder=embedder,
        )

        assert result.query == query


class TestPipelineEdgeCases:

    def test_handles_no_retrieved_passages(self) -> None:
        """Pipeline should return a low-confidence answer when retrieval yields nothing."""
        responses = [
            QUERY_DECOMPOSITION_RESPONSE,    # analyze
            SYNTHESIS_NO_DOCS_RESPONSE,      # synthesis (fallback path)
        ]
        llm = _make_mock_llm(responses)
        qdrant = _make_mock_qdrant(passages=[])  # No docs
        embedder = _make_mock_embedder()

        result = run_query(
            "What is an obscure topic?",
            llm=llm, qdrant=qdrant, embedder=embedder,
        )

        assert isinstance(result, DeliberationResult)
        # Should still produce an answer, even if low confidence
        assert len(result.answer) > 0

    def test_graph_compiles(self) -> None:
        """build_graph should return a compiled runnable without error."""
        llm = MagicMock()
        qdrant = MagicMock()
        embedder = MagicMock()

        app = build_graph(llm, qdrant, embedder)
        # LangGraph compiled graphs have an invoke method
        assert hasattr(app, "invoke")
