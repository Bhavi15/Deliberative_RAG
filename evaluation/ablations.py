"""Ablation study runner.

Runs the deliberative RAG pipeline with one component disabled at a time
to measure each component's contribution to overall performance.

Variants
--------
- **no_temporal** — temporal scores fixed to 1.0 (ignores recency)
- **no_authority** — authority scores fixed to 1.0 (ignores source quality)
- **no_graph** — skips conflict graph; all claims scored with neutral graph evidence
- **no_claims** — skips claim extraction; promotes raw passages to pseudo-claims
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import structlog

from src.pipeline.claim_extractor import ClaimExtractor
from src.pipeline.conflict_graph import ConflictGraphBuilder
from src.pipeline.query_analyzer import decompose_query
from src.pipeline.retriever import merge_and_deduplicate, retrieve_for_sub_query
from src.pipeline.scorer import ClaimScorer
from src.pipeline.synthesizer import DeliberationSynthesizer
from src.schemas import (
    Claim,
    ClaimType,
    ConfidenceLevel,
    ConflictEdge,
    DeliberationResult,
    ScoredClaim,
)
from src.utils.llm import LLMClient
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

log = structlog.get_logger()

# All recognised ablation names
ABLATION_VARIANTS: list[str] = [
    "no_temporal",
    "no_authority",
    "no_graph",
    "no_claims",
]


@dataclass
class AblationConfig:
    """Describes which component to disable."""

    name: str
    disable_temporal: bool = False
    disable_authority: bool = False
    disable_graph: bool = False
    disable_claims: bool = False


# Pre-built configs for each variant
ABLATION_CONFIGS: dict[str, AblationConfig] = {
    "no_temporal": AblationConfig(
        name="No Temporal",
        disable_temporal=True,
    ),
    "no_authority": AblationConfig(
        name="No Authority",
        disable_authority=True,
    ),
    "no_graph": AblationConfig(
        name="No Graph",
        disable_graph=True,
    ),
    "no_claims": AblationConfig(
        name="No Claims",
        disable_claims=True,
    ),
}


class AblationRunner:
    """Runs the deliberative pipeline with configurable component ablation.

    Instead of using the compiled LangGraph agent, this runner orchestrates
    the six stages manually so individual components can be surgically
    disabled.
    """

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantManager,
        embedder: EmbeddingModel,
        config: AblationConfig,
        top_k: int = 5,
    ) -> None:
        self._llm = llm
        self._qdrant = qdrant
        self._embedder = embedder
        self._config = config
        self._top_k = top_k

    @property
    def config(self) -> AblationConfig:
        return self._config

    def run(self, query: str) -> dict[str, Any]:
        """Run the ablated pipeline and return result + conflict edges.

        Returns:
            Dict with ``result`` (DeliberationResult), ``conflict_edges``
            (list[ConflictEdge]), and ``error`` (str | None).
        """
        try:
            return self._run_inner(query)
        except Exception as exc:
            log.error("ablation_run_failed", config=self._config.name, error=str(exc))
            return {
                "result": DeliberationResult(
                    query=query,
                    answer="Ablation pipeline error.",
                    confidence=ConfidenceLevel.LOW,
                    confidence_score=0.0,
                    reasoning_trace=[f"Error: {exc}"],
                    source_attribution=[],
                    conflict_summary="",
                ),
                "conflict_edges": [],
                "error": str(exc),
            }

    def _run_inner(self, query: str) -> dict[str, Any]:
        # Stage 1 — Query Analysis
        sub_queries = decompose_query(query, self._llm)
        log.info("ablation_stage1", config=self._config.name, sub_queries=len(sub_queries))

        # Stage 2 — Retrieval
        passage_groups = [
            retrieve_for_sub_query(sq, self._qdrant, self._embedder, top_k=self._top_k)
            for sq in sub_queries
        ]
        documents = merge_and_deduplicate(passage_groups)
        log.info("ablation_stage2", config=self._config.name, docs=len(documents))

        # Stage 3 — Claim Extraction (or passage-level pseudo-claims)
        if self._config.disable_claims:
            claims = _passages_to_pseudo_claims(documents)
            log.info("ablation_stage3_skipped", config=self._config.name, pseudo_claims=len(claims))
        else:
            extractor = ClaimExtractor(self._llm)
            passages = [
                {
                    "text": doc.get("text", ""),
                    "source_doc_id": str(doc.get("id", doc.get("record_id", "unknown"))),
                }
                for doc in documents
                if doc.get("text", "").strip()
            ]
            claims = extractor.extract_claims_combined(passages, query=query)
            log.info("ablation_stage3", config=self._config.name, claims=len(claims))

        # Stage 4 — Conflict Graph (or empty graph)
        if self._config.disable_graph:
            graph = _empty_graph(claims)
            conflict_edges: list[ConflictEdge] = []
            log.info("ablation_stage4_skipped", config=self._config.name)
        else:
            builder = ConflictGraphBuilder(self._llm, self._embedder)
            graph = builder.build_graph(claims)
            conflict_edges = ConflictGraphBuilder.get_conflicts(graph)
            log.info("ablation_stage4", config=self._config.name, edges=graph.number_of_edges())

        # Stage 5 — Scoring (with possible overrides)
        doc_metadata = _build_doc_metadata(documents)
        scorer = ClaimScorer()
        scored = scorer.score_all_claims(claims, graph, metadata=doc_metadata)

        # Apply ablation overrides to sub-scores and recompute final
        if self._config.disable_temporal or self._config.disable_authority:
            scored = _override_scores(scored, scorer, self._config)
        log.info("ablation_stage5", config=self._config.name, scored=len(scored))

        # Stage 6 — Synthesis
        synthesizer = DeliberationSynthesizer(self._llm)
        result = synthesizer.synthesize(query, scored, graph)
        log.info("ablation_stage6", config=self._config.name, confidence=result.confidence.value)

        return {
            "result": result,
            "conflict_edges": conflict_edges,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passages_to_pseudo_claims(documents: list[dict]) -> list[Claim]:
    """Convert raw passages into pseudo-claims (one claim per passage).

    This skips the LLM claim extraction entirely, treating each passage
    as a single atomic claim.
    """
    claims: list[Claim] = []
    for doc in documents:
        text = doc.get("text", "").strip()
        if not text:
            continue
        claims.append(
            Claim(
                claim_id=f"pseudo_{uuid.uuid4().hex[:8]}",
                text=text,
                claim_type=ClaimType.FACT,
                source_doc_id=str(doc.get("id", doc.get("record_id", "unknown"))),
                temporal_marker=None,
                confidence_in_extraction=1.0,
            )
        )
    return claims


def _empty_graph(claims: list[Claim]) -> nx.DiGraph:
    """Build a graph with claim nodes but zero edges."""
    g = nx.DiGraph()
    for c in claims:
        g.add_node(c.claim_id, claim=c)
    return g


def _build_doc_metadata(documents: list[dict]) -> dict[str, dict]:
    """Extract per-document metadata for the scorer."""
    meta: dict[str, dict] = {}
    for doc in documents:
        doc_id = str(doc.get("id", ""))
        meta[doc_id] = {
            "source_type": doc.get("source_type", ""),
            "publication_date": doc.get("publication_date"),
        }
    return meta


def _override_scores(
    scored_claims: list[ScoredClaim],
    scorer: ClaimScorer,
    config: AblationConfig,
) -> list[ScoredClaim]:
    """Override sub-scores per ablation config and recompute final_score.

    Creates new ScoredClaim objects (does not mutate originals).
    """
    result: list[ScoredClaim] = []
    for sc in scored_claims:
        t = 1.0 if config.disable_temporal else sc.temporal_score
        a = 1.0 if config.disable_authority else sc.authority_score
        g = sc.graph_evidence_score

        final = (scorer.w_temporal * t) + (scorer.w_authority * a) + (scorer.w_graph * g)

        result.append(
            ScoredClaim(
                claim_id=sc.claim_id,
                text=sc.text,
                claim_type=sc.claim_type,
                source_doc_id=sc.source_doc_id,
                temporal_marker=sc.temporal_marker,
                confidence_in_extraction=sc.confidence_in_extraction,
                temporal_score=round(t, 4),
                authority_score=round(a, 4),
                graph_evidence_score=round(g, 4),
                final_score=round(final, 4),
            )
        )

    result.sort(key=lambda s: s.final_score, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Ablation table formatting
# ---------------------------------------------------------------------------


def format_ablation_table(
    all_results: dict[str, dict[str, Any]],
) -> str:
    """Format the ablation results as a comparison table.

    Args:
        all_results: ``{config_label: {metric: value}}``.  Expected keys
            include ``"Full System"``, the four ablation labels, and
            ``"Baseline RAG"``.

    Returns:
        Formatted multi-line string.
    """
    rows = list(all_results.keys())

    # Column widths
    name_w = max(20, *(len(r) for r in rows)) + 2
    num_w = 16

    header = (
        f"{'Configuration':<{name_w}}"
        f"{'Accuracy':>{num_w}}"
        f"{'Conflict Det':>{num_w}}"
        f"{'Calibration':>{num_w}}"
    )
    sep = "-" * len(header)

    lines = [
        "",
        "Ablation Study Results",
        "=" * len(header),
        header,
        sep,
    ]

    for label in rows:
        m = all_results[label]
        acc = m.get("factual_accuracy", 0.0)
        cdr = m.get("conflict_recall")
        ece = m.get("calibration_error")

        acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        cdr_s = f"{cdr:.4f}" if isinstance(cdr, (int, float)) else "N/A"
        ece_s = f"{ece:.4f}" if isinstance(ece, (int, float)) else "N/A"

        lines.append(
            f"{label:<{name_w}}"
            f"{acc_s:>{num_w}}"
            f"{cdr_s:>{num_w}}"
            f"{ece_s:>{num_w}}"
        )

    lines.append(sep)
    lines.append("")
    return "\n".join(lines)
