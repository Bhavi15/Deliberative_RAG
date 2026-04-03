"""Pydantic v2 data models for the Deliberative RAG pipeline.

All shared data structures — claims, conflict edges, scored claims,
deliberation results, and evaluation examples — are defined here as a
single source of truth.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClaimType(StrEnum):
    """Classification of an extracted claim."""

    FACT = "fact"
    OPINION = "opinion"
    FORECAST = "forecast"
    DATA_POINT = "data_point"


class RelationType(StrEnum):
    """Typed relationship between two claims in the conflict graph."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    SUPERSEDES = "SUPERSEDES"


class ConfidenceLevel(StrEnum):
    """Coarse confidence bucket for the final answer."""

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """An atomic factual assertion extracted from a passage."""

    claim_id: str = Field(..., description="Unique claim identifier.")
    text: str = Field(..., description="The claim as a standalone assertion.")
    claim_type: ClaimType = Field(..., description="Semantic type of the claim.")
    source_doc_id: str = Field(..., description="ID of the document this claim was extracted from.")
    temporal_marker: str | None = Field(
        None, description="Optional temporal reference found in the claim (e.g. '2023', 'last quarter')."
    )
    confidence_in_extraction: float = Field(
        ..., ge=0.0, le=1.0, description="LLM confidence that this claim was correctly extracted."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "claim_id": "claim_001",
                    "text": "3M reported capital expenditure of $1,577 million in FY2018.",
                    "claim_type": "data_point",
                    "source_doc_id": "finance_bench_fae68c5a",
                    "temporal_marker": "FY2018",
                    "confidence_in_extraction": 0.95,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# ConflictEdge
# ---------------------------------------------------------------------------


class ConflictEdge(BaseModel):
    """A directed edge in the conflict graph connecting two claims."""

    source_claim_id: str = Field(..., description="ID of the source claim.")
    target_claim_id: str = Field(..., description="ID of the target claim.")
    relation: RelationType = Field(..., description="How the source claim relates to the target.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="LLM confidence in this classification."
    )
    reasoning: str = Field(
        ..., description="Short rationale explaining why this relation was assigned."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_claim_id": "claim_001",
                    "target_claim_id": "claim_002",
                    "relation": "CONTRADICTS",
                    "confidence": 0.92,
                    "reasoning": "Claim 001 states Saltzman is a musician, while claim 002 states she is a politician — mutually exclusive occupations.",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# ScoredClaim  (extends Claim)
# ---------------------------------------------------------------------------


class ScoredClaim(Claim):
    """A claim enriched with composite trustworthiness scoring."""

    temporal_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Recency score — 1.0 for the most recent source."
    )
    authority_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Source credibility score."
    )
    graph_evidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Net supporting evidence from the conflict graph."
    )
    final_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Weighted aggregate of the three sub-scores."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "claim_id": "claim_001",
                    "text": "3M reported capital expenditure of $1,577 million in FY2018.",
                    "claim_type": "data_point",
                    "source_doc_id": "finance_bench_fae68c5a",
                    "temporal_marker": "FY2018",
                    "confidence_in_extraction": 0.95,
                    "temporal_score": 0.72,
                    "authority_score": 0.90,
                    "graph_evidence_score": 0.85,
                    "final_score": 0.83,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# DeliberationResult
# ---------------------------------------------------------------------------


class SourceAttribution(BaseModel):
    """Links part of the answer to the claim and document that supports it."""

    claim_id: str = Field(..., description="ID of the supporting claim.")
    source_doc_id: str = Field(..., description="ID of the source document.")
    relevance: float = Field(
        ..., ge=0.0, le=1.0, description="How relevant this source is to the answer."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "claim_id": "claim_001",
                    "source_doc_id": "finance_bench_fae68c5a",
                    "relevance": 0.95,
                }
            ]
        }
    }


class DeliberationResult(BaseModel):
    """Final structured answer produced by the deliberative synthesis stage."""

    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Generated answer text.")
    confidence: ConfidenceLevel = Field(
        ..., description="Coarse confidence bucket."
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated numerical confidence."
    )
    reasoning_trace: list[str] = Field(
        default_factory=list,
        description="Ordered reasoning steps the model took before answering.",
    )
    source_attribution: list[SourceAttribution] = Field(
        default_factory=list,
        description="Claims and documents the answer is grounded in.",
    )
    conflict_summary: str = Field(
        "", description="Human-readable summary of detected conflicts and how they were resolved."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is Kathy Saltzman's occupation?",
                    "answer": "Kathy Saltzman is a politician who served in the Minnesota Senate.",
                    "confidence": "high",
                    "confidence_score": 0.88,
                    "reasoning_trace": [
                        "Retrieved two passages with conflicting claims about occupation.",
                        "Passage A (parametric memory) claims she is a musician — no corroborating evidence.",
                        "Passage B (counter memory) claims she is a politician — corroborated by Wikipedia.",
                        "Passage B is more recent and from a higher-authority source.",
                        "Resolved in favour of Passage B with high confidence.",
                    ],
                    "source_attribution": [
                        {
                            "claim_id": "claim_002",
                            "source_doc_id": "conflict_qa_3aa57e2f",
                            "relevance": 0.95,
                        }
                    ],
                    "conflict_summary": "Two sources disagreed on occupation (musician vs. politician). Resolved in favour of 'politician' based on Wikipedia evidence and higher authority score.",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# EvalExample
# ---------------------------------------------------------------------------


class EvalDocument(BaseModel):
    """A single source document attached to an evaluation example."""

    doc_id: str = Field(..., description="Unique document identifier.")
    text: str = Field(..., description="Document / passage text.")
    source: str = Field("", description="Origin label or URL.")
    is_correct: bool = Field(
        True, description="Whether this document represents factually correct evidence."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "doc_id": "doc_001",
                    "text": "Kathy Saltzman served in the Minnesota Senate representing District 56.",
                    "source": "counter_memory",
                    "is_correct": True,
                }
            ]
        }
    }


class EvalExample(BaseModel):
    """One evaluation example from the master benchmark dataset."""

    query_id: str = Field(..., description="Unique identifier for this evaluation example.")
    query: str = Field(..., description="The evaluation question text.")
    domain: str = Field(
        ..., description="Source dataset / domain (e.g. 'conflict_qa', 'frames', 'finance_bench')."
    )
    documents: list[EvalDocument] = Field(
        default_factory=list, description="Source documents / passages for this query."
    )
    ground_truth: str | list[str] = Field(
        ..., description="Gold answer — string or list of acceptable answer strings."
    )
    has_contradiction: bool = Field(
        False, description="True when the example is known to contain conflicting evidence."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_id": "conflict_qa_3aa57e2f",
                    "query": "What is Kathy Saltzman's occupation?",
                    "domain": "conflict_qa",
                    "documents": [
                        {
                            "doc_id": "doc_001",
                            "text": "Kathy Saltzman is a professional musician and a flutist.",
                            "source": "parametric_memory",
                            "is_correct": False,
                        },
                        {
                            "doc_id": "doc_002",
                            "text": "Kathy Saltzman served in the Minnesota Senate representing District 56.",
                            "source": "counter_memory",
                            "is_correct": True,
                        },
                    ],
                    "ground_truth": ["politician", "political leader", "political figure"],
                    "has_contradiction": True,
                }
            ]
        }
    }
