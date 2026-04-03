"""Stage 5 — Temporal-Authority Scoring.

Scores each claim by combining temporal recency, source authority, and
graph-derived evidence weight into a single composite score.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import networkx as nx
import structlog

from config.authority_weights import AUTHORITY_WEIGHTS, DEFAULT_AUTHORITY_WEIGHT
from src.schemas import Claim, ClaimType, ConflictEdge, RelationType, ScoredClaim

log = structlog.get_logger()

# Default composite weights: temporal, authority, graph_evidence
_W_TEMPORAL = 0.35
_W_AUTHORITY = 0.30
_W_GRAPH = 0.35


class ClaimScorer:
    """Computes composite trustworthiness scores for extracted claims."""

    def __init__(
        self,
        w_temporal: float = _W_TEMPORAL,
        w_authority: float = _W_AUTHORITY,
        w_graph: float = _W_GRAPH,
    ) -> None:
        """Initialise the scorer with composite weights.

        Args:
            w_temporal: Weight for temporal recency sub-score.
            w_authority: Weight for source authority sub-score.
            w_graph: Weight for graph evidence sub-score.
        """
        self.w_temporal = w_temporal
        self.w_authority = w_authority
        self.w_graph = w_graph

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    @staticmethod
    def temporal_score(
        publication_date: datetime | str | None,
        half_life_days: float = 90,
    ) -> float:
        """Compute recency score via exponential decay.

        Formula: ``exp(-0.693 * days_old / half_life_days)``

        Args:
            publication_date: Publication datetime, ISO-format string, or
                None if unknown.
            half_life_days: Number of days after which the score halves.

        Returns:
            Float in [0, 1]. Returns 1.0 if *publication_date* is None.
        """
        if publication_date is None:
            return 1.0

        if isinstance(publication_date, str):
            try:
                publication_date = datetime.fromisoformat(publication_date)
            except ValueError:
                return 1.0

        # Ensure timezone-aware comparison
        now = datetime.now(timezone.utc)
        if publication_date.tzinfo is None:
            publication_date = publication_date.replace(tzinfo=timezone.utc)

        days_old = max((now - publication_date).days, 0)
        return math.exp(-0.693 * days_old / half_life_days)

    @staticmethod
    def authority_score(source_type: str) -> float:
        """Look up the authority weight for a source type.

        Args:
            source_type: Label such as ``"official"``, ``"blog"``, etc.

        Returns:
            Float in [0, 1]. Falls back to DEFAULT_AUTHORITY_WEIGHT (0.3)
            for unknown types.
        """
        return AUTHORITY_WEIGHTS.get(source_type, DEFAULT_AUTHORITY_WEIGHT)

    @staticmethod
    def graph_evidence_score(claim_id: str, graph: nx.DiGraph) -> float:
        """Derive an evidence score from the claim's position in the conflict graph.

        - Each incoming SUPPORTS edge adds +1
        - Each incoming CONTRADICTS edge adds -1
        - Any incoming SUPERSEDES edge forces the score to 0.1

        The raw tally is normalised to [0, 1] via ``sigmoid(tally)``.

        Args:
            claim_id: The claim to score.
            graph: Conflict graph built by :class:`ConflictGraphBuilder`.

        Returns:
            Float in [0, 1]. 0.5 when no edges exist (neutral). 0.1 if
            superseded.
        """
        if claim_id not in graph:
            return 0.5

        tally = 0.0
        superseded = False

        # Incoming edges (others → this claim)
        for pred in graph.predecessors(claim_id):
            edge_data = graph.edges[pred, claim_id]
            relation = edge_data.get("relation", "")
            if relation == RelationType.SUPPORTS:
                tally += 1.0
            elif relation == RelationType.CONTRADICTS:
                tally -= 1.0
            elif relation == RelationType.SUPERSEDES:
                superseded = True

        # Outgoing edges (this claim → others)
        for succ in graph.successors(claim_id):
            edge_data = graph.edges[claim_id, succ]
            relation = edge_data.get("relation", "")
            if relation == RelationType.SUPPORTS:
                tally += 1.0
            elif relation == RelationType.CONTRADICTS:
                tally -= 1.0

        if superseded:
            return 0.1

        # Sigmoid normalisation: maps (-inf, +inf) → (0, 1), centre at 0.5
        return 1.0 / (1.0 + math.exp(-tally))

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def compute_final_score(
        self,
        claim: Claim,
        graph: nx.DiGraph,
        publication_date: datetime | str | None = None,
        source_type: str = "",
    ) -> ScoredClaim:
        """Score a single claim and return a ScoredClaim.

        Args:
            claim: The claim to score.
            graph: Conflict graph for graph-evidence scoring.
            publication_date: Optional publication date for temporal scoring.
            source_type: Source type label for authority scoring.

        Returns:
            A ScoredClaim with all sub-scores and the weighted final_score.
        """
        t = self.temporal_score(publication_date)
        a = self.authority_score(source_type)
        g = self.graph_evidence_score(claim.claim_id, graph)

        final = (self.w_temporal * t) + (self.w_authority * a) + (self.w_graph * g)

        return ScoredClaim(
            claim_id=claim.claim_id,
            text=claim.text,
            claim_type=claim.claim_type,
            source_doc_id=claim.source_doc_id,
            temporal_marker=claim.temporal_marker,
            confidence_in_extraction=claim.confidence_in_extraction,
            temporal_score=round(t, 4),
            authority_score=round(a, 4),
            graph_evidence_score=round(g, 4),
            final_score=round(final, 4),
        )

    def score_all_claims(
        self,
        claims: list[Claim],
        graph: nx.DiGraph,
        metadata: dict[str, dict] | None = None,
    ) -> list[ScoredClaim]:
        """Score every claim and return a list sorted by final_score descending.

        Args:
            claims: All extracted claims.
            graph: Conflict graph.
            metadata: Optional mapping of ``claim.source_doc_id`` →
                ``{"publication_date": ..., "source_type": ...}``.

        Returns:
            List of ScoredClaim, highest score first.
        """
        if metadata is None:
            metadata = {}

        scored: list[ScoredClaim] = []
        for claim in claims:
            meta = metadata.get(claim.source_doc_id, {})
            sc = self.compute_final_score(
                claim,
                graph,
                publication_date=meta.get("publication_date"),
                source_type=meta.get("source_type", ""),
            )
            scored.append(sc)

        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored
