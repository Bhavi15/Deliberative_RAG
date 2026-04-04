"""Tests for evaluation/ablations.py — ablation runner, configs, and helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import networkx as nx
import pytest

from evaluation.ablations import (
    ABLATION_CONFIGS,
    ABLATION_VARIANTS,
    AblationConfig,
    AblationRunner,
    _empty_graph,
    _override_scores,
    _passages_to_pseudo_claims,
    format_ablation_table,
)
from src.pipeline.scorer import ClaimScorer
from src.schemas import (
    Claim,
    ClaimType,
    ConfidenceLevel,
    DeliberationResult,
    ScoredClaim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(claim_id: str = "c1", text: str = "Test claim.") -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=ClaimType.FACT,
        source_doc_id="doc_test",
        temporal_marker=None,
        confidence_in_extraction=0.9,
    )


def _make_scored(
    claim_id: str = "c1",
    temporal: float = 0.5,
    authority: float = 0.5,
    graph: float = 0.5,
) -> ScoredClaim:
    scorer = ClaimScorer()
    final = scorer.w_temporal * temporal + scorer.w_authority * authority + scorer.w_graph * graph
    return ScoredClaim(
        claim_id=claim_id,
        text="Test",
        claim_type=ClaimType.FACT,
        source_doc_id="doc_test",
        temporal_marker=None,
        confidence_in_extraction=0.9,
        temporal_score=temporal,
        authority_score=authority,
        graph_evidence_score=graph,
        final_score=round(final, 4),
    )


# LLM mock responses for a full ablation run
QUERY_DECOMPOSITION = json.dumps(["What is X's occupation?"])

CLAIM_EXTRACTION = json.dumps([
    {"text": "X is a musician.", "claim_type": "fact", "confidence_in_extraction": 0.85},
    {"text": "X is a politician.", "claim_type": "fact", "confidence_in_extraction": 0.90},
])

CONFLICT_CLASSIFICATION = json.dumps({
    "relation": "CONTRADICTS",
    "confidence": 0.9,
    "reasoning": "Mutually exclusive occupations.",
})

SYNTHESIS_RESPONSE = json.dumps({
    "answer": "X is a politician.",
    "confidence": "high",
    "confidence_score": 0.88,
    "reasoning_trace": ["Resolved conflict in favour of politician."],
    "source_attribution": [{"claim_id": "c1", "source_doc_id": "d1", "relevance": 0.9}],
    "conflict_summary": "Musician vs politician resolved.",
})


def _mock_llm_for_full_run(extra_classification_calls: int = 0) -> MagicMock:
    """Create a mock LLM with enough responses for a full ablation pipeline."""
    responses = [
        QUERY_DECOMPOSITION,
        CLAIM_EXTRACTION,   # passage 1
        CLAIM_EXTRACTION,   # passage 2
    ]
    # 4 claims → C(4,2) = 6 pairs for conflict classification
    for _ in range(6 + extra_classification_calls):
        responses.append(CONFLICT_CLASSIFICATION)
    responses.append(SYNTHESIS_RESPONSE)
    llm = MagicMock()
    llm.invoke.side_effect = list(responses)
    return llm


def _mock_qdrant() -> MagicMock:
    qdrant = MagicMock()
    qdrant.search.return_value = [
        {"id": "d1", "score": 0.9, "text": "X is a musician.", "source_type": "blog"},
        {"id": "d2", "score": 0.8, "text": "X is a politician.", "source_type": "encyclopedic"},
    ]
    return qdrant


def _mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 384
    embedder.embed_batch.side_effect = lambda texts, **kw: [
        [0.1 + i * 0.01] * 384 for i in range(len(texts))
    ]
    return embedder


# ===========================================================================
# AblationConfig
# ===========================================================================


class TestAblationConfigs:

    def test_all_four_variants_registered(self) -> None:
        assert len(ABLATION_VARIANTS) == 4
        for v in ABLATION_VARIANTS:
            assert v in ABLATION_CONFIGS

    def test_no_temporal_config(self) -> None:
        c = ABLATION_CONFIGS["no_temporal"]
        assert c.disable_temporal is True
        assert c.disable_authority is False
        assert c.disable_graph is False
        assert c.disable_claims is False

    def test_no_authority_config(self) -> None:
        c = ABLATION_CONFIGS["no_authority"]
        assert c.disable_authority is True
        assert c.disable_temporal is False

    def test_no_graph_config(self) -> None:
        c = ABLATION_CONFIGS["no_graph"]
        assert c.disable_graph is True
        assert c.disable_claims is False

    def test_no_claims_config(self) -> None:
        c = ABLATION_CONFIGS["no_claims"]
        assert c.disable_claims is True
        assert c.disable_graph is False


# ===========================================================================
# Helper functions
# ===========================================================================


class TestPassagesToPseudoClaims:

    def test_creates_one_claim_per_passage(self) -> None:
        docs = [
            {"id": "d1", "text": "Passage one."},
            {"id": "d2", "text": "Passage two."},
        ]
        claims = _passages_to_pseudo_claims(docs)
        assert len(claims) == 2
        assert claims[0].text == "Passage one."
        assert claims[1].source_doc_id == "d2"

    def test_skips_empty_passages(self) -> None:
        docs = [
            {"id": "d1", "text": "Real content."},
            {"id": "d2", "text": "   "},
            {"id": "d3", "text": ""},
        ]
        claims = _passages_to_pseudo_claims(docs)
        assert len(claims) == 1

    def test_pseudo_claim_has_full_confidence(self) -> None:
        claims = _passages_to_pseudo_claims([{"id": "d1", "text": "test"}])
        assert claims[0].confidence_in_extraction == 1.0

    def test_claim_ids_are_unique(self) -> None:
        docs = [{"id": f"d{i}", "text": f"Text {i}"} for i in range(10)]
        claims = _passages_to_pseudo_claims(docs)
        ids = [c.claim_id for c in claims]
        assert len(set(ids)) == len(ids)


class TestEmptyGraph:

    def test_has_nodes_no_edges(self) -> None:
        claims = [_make_claim("c1"), _make_claim("c2")]
        g = _empty_graph(claims)
        assert g.number_of_nodes() == 2
        assert g.number_of_edges() == 0


class TestOverrideScores:

    def test_no_temporal_sets_temporal_to_1(self) -> None:
        sc = _make_scored(temporal=0.3, authority=0.5, graph=0.5)
        config = AblationConfig(name="test", disable_temporal=True)
        scorer = ClaimScorer()
        result = _override_scores([sc], scorer, config)
        assert result[0].temporal_score == 1.0
        assert result[0].authority_score == 0.5  # unchanged

    def test_no_authority_sets_authority_to_1(self) -> None:
        sc = _make_scored(temporal=0.5, authority=0.2, graph=0.5)
        config = AblationConfig(name="test", disable_authority=True)
        scorer = ClaimScorer()
        result = _override_scores([sc], scorer, config)
        assert result[0].authority_score == 1.0
        assert result[0].temporal_score == 0.5  # unchanged

    def test_final_score_recomputed(self) -> None:
        sc = _make_scored(temporal=0.3, authority=0.2, graph=0.5)
        config = AblationConfig(name="test", disable_temporal=True)
        scorer = ClaimScorer()
        result = _override_scores([sc], scorer, config)
        expected = scorer.w_temporal * 1.0 + scorer.w_authority * 0.2 + scorer.w_graph * 0.5
        assert result[0].final_score == pytest.approx(round(expected, 4))

    def test_result_sorted_by_final_score(self) -> None:
        sc1 = _make_scored("c1", temporal=0.1, authority=0.9, graph=0.5)
        sc2 = _make_scored("c2", temporal=0.9, authority=0.1, graph=0.5)
        config = AblationConfig(name="test", disable_temporal=True)
        scorer = ClaimScorer()
        result = _override_scores([sc1, sc2], scorer, config)
        assert result[0].final_score >= result[1].final_score


# ===========================================================================
# AblationRunner end-to-end (with mocks)
# ===========================================================================


class TestAblationRunner:

    def test_no_graph_skips_conflict_classification(self) -> None:
        """With disable_graph, the LLM should NOT be called for conflict classification."""
        # Fewer LLM calls needed: query decomp + 2 claim extractions + synthesis = 4
        responses = [
            QUERY_DECOMPOSITION,
            CLAIM_EXTRACTION,
            CLAIM_EXTRACTION,
            SYNTHESIS_RESPONSE,
        ]
        llm = MagicMock()
        llm.invoke.side_effect = list(responses)

        config = ABLATION_CONFIGS["no_graph"]
        runner = AblationRunner(llm, _mock_qdrant(), _mock_embedder(), config)
        out = runner.run("What is X's occupation?")

        assert isinstance(out["result"], DeliberationResult)
        assert out["conflict_edges"] == []
        assert out["error"] is None

    def test_no_claims_skips_extraction(self) -> None:
        """With disable_claims, claim extraction LLM calls should be skipped."""
        # query decomp + 6 conflict classification (2 pseudo-claims → C(2,2)=1 pair)
        # + synthesis = 3
        responses = [
            QUERY_DECOMPOSITION,
            CONFLICT_CLASSIFICATION,  # 1 pair from 2 pseudo-claims
            SYNTHESIS_RESPONSE,
        ]
        llm = MagicMock()
        llm.invoke.side_effect = list(responses)

        config = ABLATION_CONFIGS["no_claims"]
        runner = AblationRunner(llm, _mock_qdrant(), _mock_embedder(), config)
        out = runner.run("What is X's occupation?")

        assert isinstance(out["result"], DeliberationResult)
        assert out["error"] is None

    def test_no_temporal_returns_result(self) -> None:
        llm = _mock_llm_for_full_run()
        config = ABLATION_CONFIGS["no_temporal"]
        runner = AblationRunner(llm, _mock_qdrant(), _mock_embedder(), config)
        out = runner.run("query?")
        assert isinstance(out["result"], DeliberationResult)
        assert out["error"] is None

    def test_no_authority_returns_result(self) -> None:
        llm = _mock_llm_for_full_run()
        config = ABLATION_CONFIGS["no_authority"]
        runner = AblationRunner(llm, _mock_qdrant(), _mock_embedder(), config)
        out = runner.run("query?")
        assert isinstance(out["result"], DeliberationResult)
        assert out["error"] is None

    def test_catches_exceptions(self) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("boom")
        config = ABLATION_CONFIGS["no_temporal"]
        runner = AblationRunner(llm, _mock_qdrant(), _mock_embedder(), config)
        out = runner.run("query?")
        assert out["error"] is not None
        assert out["result"].confidence == ConfidenceLevel.LOW


# ===========================================================================
# Ablation table formatting
# ===========================================================================


class TestFormatAblationTable:

    def test_contains_all_rows(self) -> None:
        data = {
            "Full System": {"factual_accuracy": 0.85, "conflict_recall": 0.72, "calibration_error": 0.08},
            "- No Temporal": {"factual_accuracy": 0.80, "conflict_recall": 0.72, "calibration_error": 0.12},
            "- No Authority": {"factual_accuracy": 0.78, "conflict_recall": 0.72, "calibration_error": 0.15},
            "- No Graph": {"factual_accuracy": 0.75, "conflict_recall": 0.0, "calibration_error": 0.20},
            "- No Claims": {"factual_accuracy": 0.70, "conflict_recall": 0.50, "calibration_error": 0.25},
            "Baseline RAG": {"factual_accuracy": 0.65, "conflict_recall": 0.0, "calibration_error": 0.35},
        }
        table = format_ablation_table(data)
        assert "Full System" in table
        assert "No Temporal" in table
        assert "No Authority" in table
        assert "No Graph" in table
        assert "No Claims" in table
        assert "Baseline RAG" in table

    def test_contains_column_headers(self) -> None:
        data = {"Full System": {"factual_accuracy": 0.8, "conflict_recall": 0.7, "calibration_error": 0.1}}
        table = format_ablation_table(data)
        assert "Accuracy" in table
        assert "Conflict Det" in table
        assert "Calibration" in table

    def test_handles_missing_metrics_gracefully(self) -> None:
        data = {"System": {"factual_accuracy": 0.5}}
        table = format_ablation_table(data)
        assert "N/A" in table

    def test_returns_string(self) -> None:
        data = {"S": {"factual_accuracy": 0.5, "conflict_recall": 0.5, "calibration_error": 0.5}}
        assert isinstance(format_ablation_table(data), str)
