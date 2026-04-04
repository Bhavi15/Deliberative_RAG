"""Baseline RAG implementation for comparison.

A standard retrieve-then-read RAG pipeline with no conflict detection,
scoring, or deliberative synthesis.  Used as the comparison baseline
when evaluating the deliberative system.

This is intentionally simple.  The point is to show what happens when you
DON'T have conflict detection — the baseline blindly trusts all sources
equally.
"""

from __future__ import annotations

import structlog

from src.schemas import (
    ConfidenceLevel,
    DeliberationResult,
    SourceAttribution,
)
from src.utils.llm import LLMClient
from src.utils.prompts import load_prompt
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

log = structlog.get_logger()

_DEFAULT_TOP_K = 5


class BaselineRAG:
    """Vanilla retrieve-then-read RAG with no conflict awareness."""

    def __init__(
        self,
        llm: LLMClient,
        qdrant: QdrantManager,
        embedder: EmbeddingModel,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        """Initialise the baseline pipeline.

        Args:
            llm: LLM client for answer generation.
            qdrant: Qdrant manager for vector search.
            embedder: Embedding model for encoding queries.
            top_k: Number of passages to retrieve.
        """
        self._llm = llm
        self._qdrant = qdrant
        self._embedder = embedder
        self._top_k = top_k

    def run(self, query_text: str) -> DeliberationResult:
        """Execute the naive RAG pipeline: retrieve top-k, stuff, generate.

        Args:
            query_text: Raw user query string.

        Returns:
            DeliberationResult with empty reasoning trace and high
            confidence (the baseline has no self-awareness).
        """
        passages = self.retrieve_top_k(query_text)
        answer = self.generate_answer(query_text, passages)

        attributions = [
            SourceAttribution(
                claim_id=f"passage_{i}",
                source_doc_id=p.get("id", "unknown"),
                relevance=p.get("score", 0.0),
            )
            for i, p in enumerate(passages)
        ]

        return DeliberationResult(
            query=query_text,
            answer=answer,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=1.0,
            reasoning_trace=[],
            source_attribution=attributions,
            conflict_summary="",
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def retrieve_top_k(self, query_text: str) -> list[dict]:
        """Retrieve the top-k passages for a query via vector search.

        Args:
            query_text: User query.

        Returns:
            List of passage dicts with ``"text"``, ``"score"``, and metadata.
        """
        query_embedding = self._embedder.embed_text(query_text)
        results = self._qdrant.search(query_embedding, top_k=self._top_k)
        log.info("baseline_retrieval", query=query_text[:60], hits=len(results))
        return results

    def generate_answer(self, query_text: str, passages: list[dict]) -> str:
        """Call the LLM with a naive context-stuffing prompt.

        All retrieved passages are concatenated into a single context
        block — no scoring, no conflict detection, no filtering.

        Args:
            query_text: User query.
            passages: Retrieved passage dicts (must have ``"text"`` key).

        Returns:
            Generated answer string.
        """
        if not passages:
            return "I don't have enough information to answer this question."

        context = "\n\n---\n\n".join(
            f"[Source {i + 1}] {p.get('text', '')}" for i, p in enumerate(passages)
        )

        prompt = load_prompt("baseline_answer", context=context, query=query_text)
        answer = self._llm.invoke(prompt)
        log.info("baseline_generation", query=query_text[:60], answer_len=len(answer))
        return answer.strip()
