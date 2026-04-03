"""Tests for Stage 5 — Temporal-Authority Scoring."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest

from src.pipeline.scorer import ClaimScorer
from src.schemas import Claim, ClaimType, ConflictEdge, RelationType, ScoredClaim


# -- Helpers ------------------------------------------------------------------

def _make_claim(claim_id: str, text: str = "test claim", **kw) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=kw.get("claim_type", ClaimType.FACT),
        source_doc_id=kw.get("source_doc_id", "doc_test"),
        temporal_marker=kw.get("temporal_marker"),
        confidence_in_extraction=kw.get("confidence_in_extraction", 0.9),
    )


def _build_graph(claims: list[Claim], edges: list[tuple[str, str, str]]) -> nx.DiGraph:
    """Build a quick graph from (source_id, target_id, relation_str) triples."""
    g = nx.DiGraph()
    for c in claims:
        g.add_node(c.claim_id, claim=c)
    for src, tgt, rel in edges:
        g.add_edge(src, tgt, relation=rel, edge=ConflictEdge(
            source_claim_id=src,
            target_claim_id=tgt,
            relation=RelationType(rel),
            confidence=0.9,
            reasoning="test",
        ))
    return g


@pytest.fixture
def scorer() -> ClaimScorer:
    return ClaimScorer()


# =============================================================================
# Temporal scoring
# =============================================================================

class TestTemporalScore:

    def test_none_date_returns_1(self) -> None:
        """No date → assume most recent, score 1.0."""
        assert ClaimScorer.temporal_score(None) == 1.0

    def test_today_returns_near_1(self) -> None:
        """A publication from today should score very close to 1.0."""
        now = datetime.now(timezone.utc)
        score = ClaimScorer.temporal_score(now)
        assert score > 0.99

    def test_one_half_life_ago_returns_near_half(self) -> None:
        """A publication exactly one half-life ago should score ~0.5."""
        half_life = 90
        old = datetime.now(timezone.utc) - timedelta(days=half_life)
        score = ClaimScorer.temporal_score(old, half_life_days=half_life)
        assert 0.45 <= score <= 0.55

    def test_very_old_returns_near_0(self) -> None:
        """A publication 5 half-lives ago should score very low."""
        old = datetime.now(timezone.utc) - timedelta(days=450)
        score = ClaimScorer.temporal_score(old, half_life_days=90)
        assert score < 0.05

    def test_iso_string_date(self) -> None:
        """Should accept ISO-format date strings."""
        date_str = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        score = ClaimScorer.temporal_score(date_str, half_life_days=90)
        assert 0.45 <= score <= 0.55

    def test_recent_scores_higher_than_old(self) -> None:
        """More recent date → higher temporal score."""
        recent = datetime.now(timezone.utc) - timedelta(days=10)
        old = datetime.now(timezone.utc) - timedelta(days=300)
        assert ClaimScorer.temporal_score(recent) > ClaimScorer.temporal_score(old)


# =============================================================================
# Authority scoring
# =============================================================================

class TestAuthorityScore:

    def test_official_is_1(self) -> None:
        assert ClaimScorer.authority_score("official") == 1.0

    def test_blog_is_015(self) -> None:
        assert ClaimScorer.authority_score("blog") == 0.15

    def test_unknown_returns_default(self) -> None:
        """Unknown source type should fall back to 0.3."""
        assert ClaimScorer.authority_score("random_source") == 0.3

    def test_official_higher_than_blog(self) -> None:
        assert ClaimScorer.authority_score("official") > ClaimScorer.authority_score("blog")

    def test_peer_reviewed_higher_than_news(self) -> None:
        assert ClaimScorer.authority_score("peer_reviewed") > ClaimScorer.authority_score("news_major")


# =============================================================================
# Graph evidence scoring
# =============================================================================

class TestGraphEvidenceScore:

    def test_no_edges_returns_0_5(self) -> None:
        """A claim with no graph edges should get the neutral score of 0.5."""
        c = _make_claim("c1")
        g = _build_graph([c], [])
        assert ClaimScorer.graph_evidence_score("c1", g) == pytest.approx(0.5, abs=0.01)

    def test_claim_not_in_graph_returns_0_5(self) -> None:
        """A claim missing from the graph should get 0.5."""
        assert ClaimScorer.graph_evidence_score("missing", nx.DiGraph()) == 0.5

    def test_supports_boost_score(self) -> None:
        """Incoming SUPPORTS edges should push the score above 0.5."""
        c1 = _make_claim("c1")
        c2 = _make_claim("c2")
        g = _build_graph([c1, c2], [("c2", "c1", "SUPPORTS")])
        score = ClaimScorer.graph_evidence_score("c1", g)
        assert score > 0.5

    def test_contradicts_lower_score(self) -> None:
        """Incoming CONTRADICTS edges should push the score below 0.5."""
        c1 = _make_claim("c1")
        c2 = _make_claim("c2")
        g = _build_graph([c1, c2], [("c2", "c1", "CONTRADICTS")])
        score = ClaimScorer.graph_evidence_score("c1", g)
        assert score < 0.5

    def test_superseded_returns_0_1(self) -> None:
        """A superseded claim should get the hard penalty of 0.1."""
        c1 = _make_claim("c1")
        c2 = _make_claim("c2")
        g = _build_graph([c1, c2], [("c2", "c1", "SUPERSEDES")])
        score = ClaimScorer.graph_evidence_score("c1", g)
        assert score == pytest.approx(0.1)

    def test_multiple_supports_stack(self) -> None:
        """Multiple supporters should push the score higher."""
        c1 = _make_claim("c1")
        c2 = _make_claim("c2")
        c3 = _make_claim("c3")
        g = _build_graph(
            [c1, c2, c3],
            [("c2", "c1", "SUPPORTS"), ("c3", "c1", "SUPPORTS")],
        )
        score_2_supporters = ClaimScorer.graph_evidence_score("c1", g)

        g1 = _build_graph([c1, c2], [("c2", "c1", "SUPPORTS")])
        score_1_supporter = ClaimScorer.graph_evidence_score("c1", g1)

        assert score_2_supporters > score_1_supporter


# =============================================================================
# Composite scoring
# =============================================================================

class TestCompositeScore:

    def test_recent_official_beats_old_blog(self, scorer) -> None:
        """A recent claim from an official source should outscore an old blog post."""
        c_official = _make_claim("c_off", "Official claim", source_doc_id="sec_filing")
        c_blog = _make_claim("c_blog", "Blog claim", source_doc_id="random_blog")
        g = _build_graph([c_official, c_blog], [])

        recent = datetime.now(timezone.utc) - timedelta(days=5)
        old = datetime.now(timezone.utc) - timedelta(days=300)

        sc_off = scorer.compute_final_score(c_official, g, publication_date=recent, source_type="official")
        sc_blog = scorer.compute_final_score(c_blog, g, publication_date=old, source_type="blog")

        assert sc_off.final_score > sc_blog.final_score

    def test_superseded_claim_scores_low(self, scorer) -> None:
        """A superseded claim should score much lower than the same claim unsuperseded."""
        c1 = _make_claim("c1", "Old revenue data")
        c2 = _make_claim("c2", "Updated revenue data")

        # Superseded graph
        g_super = _build_graph([c1, c2], [("c2", "c1", "SUPERSEDES")])
        sc_super = scorer.compute_final_score(c1, g_super, source_type="official")
        assert sc_super.graph_evidence_score == pytest.approx(0.1)

        # Same claim, no supersession
        g_clean = _build_graph([c1], [])
        sc_clean = scorer.compute_final_score(c1, g_clean, source_type="official")

        # Superseded version should be significantly lower
        assert sc_super.final_score < sc_clean.final_score
        assert sc_clean.final_score - sc_super.final_score > 0.1

    def test_supporting_claims_boost_each_other(self, scorer) -> None:
        """Claims that SUPPORT each other should both score above neutral."""
        c1 = _make_claim("c1", "Revenue was $32.8B")
        c2 = _make_claim("c2", "Annual revenue ~$32.8 billion")
        g = _build_graph(
            [c1, c2],
            [("c1", "c2", "SUPPORTS"), ("c2", "c1", "SUPPORTS")],
        )

        sc1 = scorer.compute_final_score(c1, g, source_type="institutional")
        sc2 = scorer.compute_final_score(c2, g, source_type="institutional")

        assert sc1.graph_evidence_score > 0.5
        assert sc2.graph_evidence_score > 0.5

    def test_final_score_within_bounds(self, scorer) -> None:
        """final_score must be in [0, 1]."""
        c = _make_claim("c1")
        g = _build_graph([c], [])
        sc = scorer.compute_final_score(c, g)
        assert 0.0 <= sc.final_score <= 1.0

    def test_scored_claim_preserves_claim_fields(self, scorer) -> None:
        """ScoredClaim should carry all original Claim fields."""
        c = _make_claim("cx", "Test text", claim_type=ClaimType.DATA_POINT,
                        source_doc_id="doc_x", temporal_marker="2024")
        g = _build_graph([c], [])
        sc = scorer.compute_final_score(c, g)
        assert sc.claim_id == "cx"
        assert sc.text == "Test text"
        assert sc.claim_type == ClaimType.DATA_POINT
        assert sc.source_doc_id == "doc_x"
        assert sc.temporal_marker == "2024"


# =============================================================================
# Batch scoring
# =============================================================================

class TestScoreAllClaims:

    def test_returns_sorted_descending(self, scorer) -> None:
        """score_all_claims should return claims sorted by final_score desc."""
        c1 = _make_claim("c1", source_doc_id="d1")
        c2 = _make_claim("c2", source_doc_id="d2")
        g = _build_graph([c1, c2], [])

        meta = {
            "d1": {"source_type": "blog"},
            "d2": {"source_type": "official"},
        }
        results = scorer.score_all_claims([c1, c2], g, metadata=meta)
        assert results[0].final_score >= results[1].final_score

    def test_returns_all_claims(self, scorer) -> None:
        """Should return one ScoredClaim per input."""
        claims = [_make_claim(f"c{i}") for i in range(5)]
        g = _build_graph(claims, [])
        results = scorer.score_all_claims(claims, g)
        assert len(results) == 5
