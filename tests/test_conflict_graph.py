"""Tests for Stage 4 — Conflict Graph Construction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.conflict_graph import ConflictGraphBuilder, _extract_json_object
from src.schemas import Claim, ClaimType, ConflictEdge, RelationType


# -- Helpers ------------------------------------------------------------------

def _make_claim(claim_id: str, text: str, **kwargs) -> Claim:
    """Shortcut to build a Claim with defaults."""
    return Claim(
        claim_id=claim_id,
        text=text,
        claim_type=kwargs.get("claim_type", ClaimType.FACT),
        source_doc_id=kwargs.get("source_doc_id", "doc_test"),
        temporal_marker=kwargs.get("temporal_marker"),
        confidence_in_extraction=kwargs.get("confidence_in_extraction", 0.9),
    )


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture
def contradictory_claims() -> list[Claim]:
    """Two claims that directly contradict each other."""
    return [
        _make_claim("c1", "Kathy Saltzman is a professional musician.", claim_type=ClaimType.FACT),
        _make_claim("c2", "Kathy Saltzman is a politician who served in the Minnesota Senate.", claim_type=ClaimType.FACT),
    ]


@pytest.fixture
def supporting_claims() -> list[Claim]:
    """Two claims that support each other."""
    return [
        _make_claim("c3", "3M reported total revenue of $32.8 billion in FY2023.", claim_type=ClaimType.DATA_POINT, temporal_marker="FY2023"),
        _make_claim("c4", "3M's FY2023 annual revenue was approximately $32.8B.", claim_type=ClaimType.DATA_POINT, temporal_marker="FY2023"),
    ]


@pytest.fixture
def all_claims(contradictory_claims, supporting_claims) -> list[Claim]:
    """All four test claims combined."""
    return contradictory_claims + supporting_claims


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock EmbeddingModel that returns vectors placing similar claims close together."""
    embedder = MagicMock()

    def fake_embed_batch(texts, **kwargs):
        """Return vectors where:
        - c1 & c2 (contradictory, both about Saltzman) are similar
        - c3 & c4 (supporting, both about 3M revenue) are similar
        - cross-group pairs are dissimilar
        """
        vecs = []
        for t in texts:
            if "Saltzman" in t:
                # Saltzman cluster: point near [1, 0, 0, ...]
                v = np.array([0.9, 0.1, 0.0, 0.0])
            else:
                # 3M revenue cluster: point near [0, 0, 1, 0, ...]
                v = np.array([0.0, 0.1, 0.9, 0.0])
            # Add small noise to distinguish within cluster
            v = v + np.random.default_rng(hash(t) % 2**31).normal(0, 0.02, size=4)
            v = v / np.linalg.norm(v)
            vecs.append(v.tolist())
        return vecs

    embedder.embed_batch = fake_embed_batch
    return embedder


def _make_llm_response(relation: str, confidence: float, reasoning: str) -> str:
    """Build a JSON string mimicking the LLM classification response."""
    return json.dumps({
        "relation": relation,
        "confidence": confidence,
        "reasoning": reasoning,
    })


@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLMClient that classifies based on claim content."""
    llm = MagicMock()

    def route_classification(prompt: str) -> str:
        if "musician" in prompt and "politician" in prompt:
            return _make_llm_response(
                "CONTRADICTS", 0.95,
                "One claims musician, the other claims politician — mutually exclusive.",
            )
        elif "32.8 billion" in prompt and "32.8B" in prompt:
            return _make_llm_response(
                "SUPPORTS", 0.92,
                "Both claims report the same revenue figure for FY2023.",
            )
        return _make_llm_response("UNRELATED", 0.5, "No clear relationship.")

    llm.invoke.side_effect = route_classification
    return llm


@pytest.fixture
def builder(mock_llm, mock_embedder) -> ConflictGraphBuilder:
    """ConflictGraphBuilder wired to mocks."""
    return ConflictGraphBuilder(mock_llm, mock_embedder, similarity_threshold=0.6)


# -- Tests: graph building ----------------------------------------------------


def test_build_graph_has_all_nodes(builder, all_claims) -> None:
    """Graph should contain one node per claim."""
    graph = builder.build_graph(all_claims)
    assert set(graph.nodes) == {"c1", "c2", "c3", "c4"}


def test_build_graph_contradicts_edge(builder, all_claims) -> None:
    """The two Saltzman claims should have a CONTRADICTS edge."""
    graph = builder.build_graph(all_claims)
    # Edge could be c1->c2 or c2->c1 depending on pair ordering
    found = False
    for u, v, data in graph.edges(data=True):
        edge: ConflictEdge = data["edge"]
        if {u, v} == {"c1", "c2"} and edge.relation == RelationType.CONTRADICTS:
            found = True
            break
    assert found, "Expected a CONTRADICTS edge between c1 and c2"


def test_build_graph_supports_edge(builder, all_claims) -> None:
    """The two 3M revenue claims should have a SUPPORTS edge."""
    graph = builder.build_graph(all_claims)
    found = False
    for u, v, data in graph.edges(data=True):
        edge: ConflictEdge = data["edge"]
        if {u, v} == {"c3", "c4"} and edge.relation == RelationType.SUPPORTS:
            found = True
            break
    assert found, "Expected a SUPPORTS edge between c3 and c4"


def test_build_graph_no_cross_cluster_edges(builder, all_claims) -> None:
    """Claims in different clusters should NOT be connected (UNRELATED filtered)."""
    graph = builder.build_graph(all_claims)
    for u, v, _ in graph.edges(data=True):
        pair = {u, v}
        # Saltzman <-> 3M should not exist
        assert not (pair & {"c1", "c2"} and pair & {"c3", "c4"}), \
            f"Unexpected cross-cluster edge {u} -> {v}"


def test_get_conflicts_returns_only_contradicts_and_supersedes(builder, all_claims) -> None:
    """get_conflicts should not include SUPPORTS edges."""
    graph = builder.build_graph(all_claims)
    conflicts = ConflictGraphBuilder.get_conflicts(graph)
    for edge in conflicts:
        assert edge.relation in (RelationType.CONTRADICTS, RelationType.SUPERSEDES)


def test_get_conflicts_includes_contradicts(builder, all_claims) -> None:
    """get_conflicts should find the Saltzman contradiction."""
    graph = builder.build_graph(all_claims)
    conflicts = ConflictGraphBuilder.get_conflicts(graph)
    contradicts = [e for e in conflicts if e.relation == RelationType.CONTRADICTS]
    assert len(contradicts) >= 1


# -- Tests: edge cases -------------------------------------------------------


def test_build_graph_single_claim(builder) -> None:
    """A single claim should produce a graph with one node and no edges."""
    claim = _make_claim("c_only", "Only claim.")
    graph = builder.build_graph([claim])
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0


def test_build_graph_empty_list(builder) -> None:
    """An empty claim list should produce an empty graph."""
    graph = builder.build_graph([])
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


# -- Tests: edge data --------------------------------------------------------


def test_edge_has_confidence_and_reasoning(builder, all_claims) -> None:
    """Every edge should carry confidence and reasoning from the LLM."""
    graph = builder.build_graph(all_claims)
    for _, _, data in graph.edges(data=True):
        edge: ConflictEdge = data["edge"]
        assert 0.0 <= edge.confidence <= 1.0
        assert len(edge.reasoning) > 0


# -- Tests: JSON helper -------------------------------------------------------


def test_extract_json_object_plain() -> None:
    """Should parse a bare JSON object."""
    result = _extract_json_object('{"relation": "SUPPORTS"}')
    assert result["relation"] == "SUPPORTS"


def test_extract_json_object_with_fences() -> None:
    """Should strip markdown fences."""
    raw = '```json\n{"relation": "CONTRADICTS"}\n```'
    result = _extract_json_object(raw)
    assert result["relation"] == "CONTRADICTS"


def test_extract_json_object_with_prose() -> None:
    """Should ignore leading prose."""
    raw = 'Here is my analysis:\n{"relation": "SUPERSEDES", "confidence": 0.8, "reasoning": "newer"}'
    result = _extract_json_object(raw)
    assert result["relation"] == "SUPERSEDES"


def test_extract_json_object_no_object_raises() -> None:
    """Should raise ValueError when no JSON object found."""
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json_object("no json here")
