"""Pydantic v2 data models for the Deliberative RAG pipeline.

All shared data structures — queries, documents, claims, conflict edges,
scored claims, and answers — are defined here as a single source of truth.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class SubQuery(BaseModel):
    """A single focused sub-query decomposed from the original user query."""

    id: str = Field(..., description="Unique identifier for this sub-query.")
    text: str = Field(..., description="The sub-query text.")
    focus: str = Field(..., description="What aspect of the original query this covers.")


class Query(BaseModel):
    """The top-level user query along with its decomposed sub-queries."""

    id: str = Field(..., description="Unique identifier for this query session.")
    text: str = Field(..., description="Original user query text.")
    sub_queries: list[SubQuery] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Documents & Passages
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """A source document ingested into the vector store."""

    id: str = Field(..., description="Unique document identifier.")
    title: str = Field(..., description="Document title or headline.")
    source_url: str = Field("", description="Canonical URL or file path.")
    authority_score: float = Field(
        0.5, ge=0.0, le=1.0, description="Source credibility score (0–1)."
    )
    published_at: datetime | None = Field(None, description="Publication timestamp.")
    metadata: dict = Field(default_factory=dict)


class Passage(BaseModel):
    """A chunk of text retrieved from a document."""

    id: str = Field(..., description="Unique passage identifier.")
    document_id: str = Field(..., description="Parent document identifier.")
    text: str = Field(..., description="Raw passage text.")
    retrieval_score: float = Field(0.0, description="Cosine similarity score.")
    document: Document | None = Field(None, description="Hydrated parent document.")


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """An atomic factual assertion extracted from a passage."""

    id: str = Field(..., description="Unique claim identifier.")
    passage_id: str = Field(..., description="Source passage identifier.")
    document_id: str = Field(..., description="Source document identifier.")
    text: str = Field(..., description="The claim as a standalone assertion.")


class ScoredClaim(BaseModel):
    """A claim enriched with composite trustworthiness scoring."""

    claim: Claim
    temporal_score: float = Field(0.0, ge=0.0, le=1.0)
    authority_score: float = Field(0.0, ge=0.0, le=1.0)
    graph_score: float = Field(0.0, ge=0.0, le=1.0)
    composite_score: float = Field(0.0, ge=0.0, le=1.0, description="Weighted aggregate.")


# ---------------------------------------------------------------------------
# Conflict Graph
# ---------------------------------------------------------------------------


class EdgeType(StrEnum):
    """Typed relationship between two claims in the conflict graph."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    UNRELATED = "unrelated"


class ConflictEdge(BaseModel):
    """A directed edge in the conflict graph connecting two claims."""

    source_claim_id: str
    target_claim_id: str
    edge_type: EdgeType
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    explanation: str = Field("")


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------


class ReasoningTrace(BaseModel):
    """Step-by-step deliberation trace produced during synthesis."""

    steps: list[str] = Field(default_factory=list)
    conflicts_identified: list[ConflictEdge] = Field(default_factory=list)
    claims_used: list[ScoredClaim] = Field(default_factory=list)


class Answer(BaseModel):
    """Final structured answer returned to the user."""

    query_id: str
    text: str = Field(..., description="The generated answer text.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: ReasoningTrace
    has_unresolved_conflicts: bool = Field(False)
