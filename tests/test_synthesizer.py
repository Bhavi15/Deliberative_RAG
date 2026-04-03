"""Tests for Stage 6 — Deliberation Synthesis."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import networkx as nx
import pytest

from src.pipeline.synthesizer import (
    DeliberationSynthesizer,
    build_claims_context,
    build_conflict_summary,
    parse_synthesis_response,
)
from src.schemas import (
    ClaimType,
    ConfidenceLevel,
    ConflictEdge,
    DeliberationResult,
    RelationType,
    ScoredClaim,
    SourceAttribution,
)


# -- Helpers ------------------------------------------------------------------

def _scored_claim(claim_id: str, text: str, score: float, **kw) -> ScoredClaim:
    return ScoredClaim(
        claim_id=claim_id,
        text=text,
        claim_type=kw.get("claim_type", ClaimType.FACT),
        source_doc_id=kw.get("source_doc_id", "doc_test"),
        temporal_marker=kw.get("temporal_marker"),
        confidence_in_extraction=0.9,
        temporal_score=kw.get("temporal_score", 0.8),
        authority_score=kw.get("authority_score", 0.7),
        graph_evidence_score=kw.get("graph_evidence_score", 0.6),
        final_score=score,
    )


def _build_graph(claims: list[ScoredClaim], edge_triples: list[tuple]) -> nx.DiGraph:
    g = nx.DiGraph()
    for c in claims:
        g.add_node(c.claim_id, claim=c)
    for src, tgt, rel, reasoning in edge_triples:
        edge = ConflictEdge(
            source_claim_id=src,
            target_claim_id=tgt,
            relation=RelationType(rel),
            confidence=0.9,
            reasoning=reasoning,
        )
        g.add_edge(src, tgt, edge=edge, relation=rel)
    return g


GOOD_LLM_RESPONSE = json.dumps({
    "answer": "Kathy Saltzman is a politician who served in the Minnesota Senate.",
    "confidence": "high",
    "confidence_score": 0.88,
    "reasoning_trace": [
        "Two sources provide conflicting claims about Saltzman's occupation.",
        "The counter-memory source (score 0.82) identifies her as a politician with Wikipedia evidence.",
        "The parametric-memory source (score 0.45) claims she is a musician with no corroboration.",
        "Resolved in favour of the higher-scored, Wikipedia-backed claim.",
    ],
    "source_attribution": [
        {"claim_id": "c2", "source_doc_id": "doc_counter", "relevance": 0.95},
    ],
    "conflict_summary": "Two sources disagreed on occupation (musician vs politician). Resolved in favour of 'politician' based on higher authority and graph support.",
})


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def conflicting_claims() -> list[ScoredClaim]:
    return [
        _scored_claim("c1", "Kathy Saltzman is a professional musician.", 0.45,
                       source_doc_id="doc_param", graph_evidence_score=0.3),
        _scored_claim("c2", "Kathy Saltzman is a politician who served in the Minnesota Senate.", 0.82,
                       source_doc_id="doc_counter", graph_evidence_score=0.8),
    ]


@pytest.fixture
def supporting_claims() -> list[ScoredClaim]:
    return [
        _scored_claim("c3", "3M reported revenue of $32.8 billion in FY2023.", 0.90,
                       claim_type=ClaimType.DATA_POINT, temporal_marker="FY2023"),
        _scored_claim("c4", "3M's FY2023 annual revenue was approximately $32.8B.", 0.85,
                       claim_type=ClaimType.DATA_POINT, temporal_marker="FY2023"),
    ]


@pytest.fixture
def all_claims(conflicting_claims, supporting_claims) -> list[ScoredClaim]:
    return sorted(conflicting_claims + supporting_claims, key=lambda c: c.final_score, reverse=True)


@pytest.fixture
def conflict_graph(all_claims) -> nx.DiGraph:
    return _build_graph(all_claims, [
        ("c1", "c2", "CONTRADICTS", "Musician vs politician — mutually exclusive."),
        ("c3", "c4", "SUPPORTS", "Both report the same revenue figure."),
    ])


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = GOOD_LLM_RESPONSE
    return llm


@pytest.fixture
def synthesizer(mock_llm) -> DeliberationSynthesizer:
    return DeliberationSynthesizer(mock_llm)


# -- Tests: full synthesis ----------------------------------------------------


class TestSynthesize:

    def test_returns_deliberation_result(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        assert isinstance(result, DeliberationResult)

    def test_answer_not_empty(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        assert len(result.answer) > 0

    def test_confidence_is_valid_level(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        assert result.confidence in ConfidenceLevel

    def test_confidence_score_between_0_and_1(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        assert 0.0 <= result.confidence_score <= 1.0

    def test_reasoning_trace_has_steps(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        assert len(result.reasoning_trace) >= 2

    def test_source_attribution_present(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        assert len(result.source_attribution) >= 1
        for sa in result.source_attribution:
            assert isinstance(sa, SourceAttribution)
            assert 0.0 <= sa.relevance <= 1.0

    def test_conflict_summary_present_when_conflicts_exist(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        assert len(result.conflict_summary) > 0

    def test_query_preserved(self, synthesizer, all_claims, conflict_graph) -> None:
        query = "What is Kathy Saltzman's occupation?"
        result = synthesizer.synthesize(query, all_claims, conflict_graph)
        assert result.query == query


# -- Tests: fallback on bad LLM output ----------------------------------------


class TestFallback:

    def test_malformed_json_retries_then_fallback(self, all_claims, conflict_graph) -> None:
        llm = MagicMock()
        llm.invoke.return_value = "NOT JSON AT ALL"
        synth = DeliberationSynthesizer(llm)
        result = synth.synthesize("query?", all_claims, conflict_graph)
        assert isinstance(result, DeliberationResult)
        assert result.confidence == ConfidenceLevel.LOW
        assert llm.invoke.call_count == 2  # original + retry

    def test_fallback_uses_top_claim(self, all_claims, conflict_graph) -> None:
        llm = MagicMock()
        llm.invoke.return_value = "garbage"
        synth = DeliberationSynthesizer(llm)
        result = synth.synthesize("query?", all_claims, conflict_graph)
        # Should use the top-scored claim text as the answer
        assert result.answer == all_claims[0].text

    def test_retry_succeeds_on_second_attempt(self, all_claims, conflict_graph) -> None:
        llm = MagicMock()
        llm.invoke.side_effect = ["not json", GOOD_LLM_RESPONSE]
        synth = DeliberationSynthesizer(llm)
        result = synth.synthesize("query?", all_claims, conflict_graph)
        assert result.confidence == ConfidenceLevel.HIGH
        assert llm.invoke.call_count == 2


# -- Tests: context building --------------------------------------------------


class TestBuildClaimsContext:

    def test_all_claims_present(self, all_claims) -> None:
        ctx = build_claims_context(all_claims, [])
        for sc in all_claims:
            assert sc.claim_id in ctx

    def test_scores_in_context(self, all_claims) -> None:
        ctx = build_claims_context(all_claims, [])
        assert "FINAL=" in ctx
        assert "temporal=" in ctx
        assert "authority=" in ctx

    def test_conflict_annotations_added(self, conflicting_claims) -> None:
        edge = ConflictEdge(
            source_claim_id="c1", target_claim_id="c2",
            relation=RelationType.CONTRADICTS, confidence=0.9,
            reasoning="Mutually exclusive occupations.",
        )
        ctx = build_claims_context(conflicting_claims, [edge])
        assert "CONFLICTS" in ctx
        assert "Mutually exclusive" in ctx


class TestBuildConflictSummary:

    def test_empty_when_no_conflicts(self, all_claims) -> None:
        assert build_conflict_summary([], all_claims) == ""

    def test_mentions_relation(self, conflicting_claims) -> None:
        edge = ConflictEdge(
            source_claim_id="c1", target_claim_id="c2",
            relation=RelationType.CONTRADICTS, confidence=0.9,
            reasoning="Occupations differ.",
        )
        summary = build_conflict_summary([edge], conflicting_claims)
        assert "CONTRADICTS" in summary
        assert "c1" in summary
        assert "c2" in summary


# -- Tests: parse_synthesis_response -------------------------------------------


class TestParseSynthesisResponse:

    def test_parses_good_response(self) -> None:
        result = parse_synthesis_response(GOOD_LLM_RESPONSE, "test query")
        assert result is not None
        assert result.answer == "Kathy Saltzman is a politician who served in the Minnesota Senate."
        assert result.confidence == ConfidenceLevel.HIGH

    def test_returns_none_on_garbage(self) -> None:
        assert parse_synthesis_response("not json", "q") is None

    def test_returns_none_on_empty_answer(self) -> None:
        bad = json.dumps({"answer": "", "confidence": "high", "confidence_score": 0.9})
        assert parse_synthesis_response(bad, "q") is None

    def test_clamps_confidence_score(self) -> None:
        raw = json.dumps({
            "answer": "test", "confidence": "high", "confidence_score": 1.5,
            "reasoning_trace": ["step"], "source_attribution": [], "conflict_summary": "",
        })
        result = parse_synthesis_response(raw, "q")
        assert result.confidence_score == 1.0

    def test_handles_markdown_fences(self) -> None:
        raw = f"```json\n{GOOD_LLM_RESPONSE}\n```"
        result = parse_synthesis_response(raw, "q")
        assert result is not None


# -- Tests: format_for_display -------------------------------------------------


class TestFormatForDisplay:

    def test_contains_key_sections(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("What is Kathy Saltzman's occupation?", all_claims, conflict_graph)
        display = DeliberationSynthesizer.format_for_display(result)
        assert "Query:" in display
        assert "Answer:" in display
        assert "Confidence:" in display
        assert "Reasoning:" in display
        assert "Sources:" in display

    def test_returns_string(self, synthesizer, all_claims, conflict_graph) -> None:
        result = synthesizer.synthesize("q?", all_claims, conflict_graph)
        display = DeliberationSynthesizer.format_for_display(result)
        assert isinstance(display, str)
        assert len(display) > 0
