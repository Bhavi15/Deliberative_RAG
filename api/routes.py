"""FastAPI route definitions.

Registers all API endpoints on a single APIRouter that is mounted by api/main.py.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(..., min_length=1, description="User query text.")
    mode: str = Field(
        "deep_research",
        description="Research mode: 'deep_research' or 'classic'.",
    )


class QueryResponse(BaseModel):
    """Full answer payload returned to API callers."""

    query_id: str
    answer: str = Field(..., description="Generated answer text.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    has_unresolved_conflicts: bool
    reasoning_steps: list[str]
    sources: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = Field("deep_research")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field("ok")
    version: str
    capabilities: list[str] = Field(default_factory=list)


class ConflictDetectRequest(BaseModel):
    """Request to detect conflicts between passages."""

    passages: list[str] = Field(..., min_length=2, description="Text passages to check.")


class ConflictDetectResponse(BaseModel):
    """Conflict detection result."""

    num_claims: int
    num_conflicts: int
    claims: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service liveness status and capabilities.

    Returns:
        HealthResponse with status, version, and available features.
    """
    return HealthResponse(
        status="ok",
        version="2.0.0",
        capabilities=[
            "deep_research",
            "classic_pipeline",
            "web_search",
            "conflict_detection",
            "document_upload",
            "mcp_server",
            "a2a_protocol",
        ],
    )


@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Accept a user query and return a deliberative RAG answer.

    Supports two modes:
    - ``deep_research``: Uses the deepagents-powered research agent with
      iterative web + KB search and conflict detection.
    - ``classic``: Original 6-stage LangGraph pipeline.

    Args:
        request: Validated QueryRequest payload.

    Returns:
        QueryResponse containing the answer, confidence, and reasoning steps.
    """
    query_id = str(uuid.uuid4())[:8]

    if request.mode == "deep_research":
        from src.agent.deep_research import run_deep_research

        result = run_deep_research(request.query)
        answer = result.get("answer", "No result produced.")

        return QueryResponse(
            query_id=query_id,
            answer=answer,
            confidence=0.8,
            has_unresolved_conflicts=False,
            reasoning_steps=["Deep research agent completed."],
            sources=[],
            mode="deep_research",
        )
    else:
        from src.agent.graph import run_query_full
        from src.utils.llm import get_llm
        from src.vectorstore.embeddings import EmbeddingModel
        from src.vectorstore.qdrant_client import QdrantManager

        llm_heavy = get_llm("heavy")
        llm_light = get_llm("light")
        embedder = EmbeddingModel()
        qdrant = QdrantManager()

        try:
            full_state = run_query_full(
                request.query,
                llm_heavy=llm_heavy,
                qdrant=qdrant,
                embedder=embedder,
                llm_light=llm_light,
            )

            result = full_state["final_answer"]
            conflict_edges = full_state.get("conflict_graph", {}).get("edges", [])
            has_conflicts = any(
                e.get("relation") in ("CONTRADICTS", "SUPERSEDES")
                for e in conflict_edges
            )

            return QueryResponse(
                query_id=query_id,
                answer=result.answer,
                confidence=result.confidence_score,
                has_unresolved_conflicts=has_conflicts,
                reasoning_steps=result.reasoning_trace,
                sources=[
                    {"claim_id": sa.claim_id, "doc_id": sa.source_doc_id}
                    for sa in result.source_attribution
                ],
                mode="classic",
            )
        finally:
            qdrant.close()


@router.post("/conflict-detect", response_model=ConflictDetectResponse)
async def detect_conflicts(request: ConflictDetectRequest) -> ConflictDetectResponse:
    """Detect contradictions between given text passages.

    Extracts claims, builds a conflict graph, and returns detected
    CONTRADICTS and SUPERSEDES relationships.

    Args:
        request: List of text passages to analyze.

    Returns:
        ConflictDetectResponse with claims and detected conflicts.
    """
    from src.pipeline.claim_extractor import ClaimExtractor
    from src.pipeline.conflict_graph import ConflictGraphBuilder
    from src.utils.llm import get_llm
    from src.vectorstore.embeddings import EmbeddingModel

    llm = get_llm("light")
    embedder = EmbeddingModel()

    extractor = ClaimExtractor(llm)
    passage_dicts = [
        {"text": p, "source_doc_id": f"passage_{i}"}
        for i, p in enumerate(request.passages)
    ]
    claims = extractor.extract_claims_combined(passage_dicts, query="")

    if not claims:
        return ConflictDetectResponse(
            num_claims=0, num_conflicts=0, claims=[], conflicts=[],
        )

    builder = ConflictGraphBuilder(llm, embedder)
    graph = builder.build_graph(claims)

    conflicts = []
    for u, v, data in graph.edges(data=True):
        edge = data.get("edge")
        if edge and edge.relation.value in ("CONTRADICTS", "SUPERSEDES"):
            conflicts.append({
                "source_claim": edge.source_claim_id,
                "target_claim": edge.target_claim_id,
                "relation": edge.relation.value,
                "confidence": edge.confidence,
                "reasoning": edge.reasoning,
            })

    return ConflictDetectResponse(
        num_claims=len(claims),
        num_conflicts=len(conflicts),
        claims=[
            {"id": c.claim_id, "text": c.text, "type": c.claim_type.value}
            for c in claims
        ],
        conflicts=conflicts,
    )
