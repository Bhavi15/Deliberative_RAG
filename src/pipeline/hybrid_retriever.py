"""Hybrid retrieval: BM25 lexical search + vector semantic search with Reciprocal Rank Fusion.

Combines keyword-based BM25 retrieval with dense vector retrieval to capture
both exact term matches and semantic similarity — the approach used by
production search systems like Perplexity and Bing.
"""

from __future__ import annotations

from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from src.tools.web_search import web_search_structured
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

log = structlog.get_logger()

# Reciprocal Rank Fusion constant (standard value from literature)
_RRF_K = 60


class HybridRetriever:
    """Combines vector search, BM25, and optional web search with RRF fusion.

    The retriever supports three modes:
    - ``"kb_only"``: Qdrant vector + BM25 on indexed corpus
    - ``"web_only"``: Tavily web search
    - ``"hybrid"``: All sources fused via Reciprocal Rank Fusion
    """

    def __init__(
        self,
        qdrant: QdrantManager | None = None,
        embedder: EmbeddingModel | None = None,
        corpus: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            qdrant: Qdrant manager for vector search. May be None if
                only web search is needed.
            embedder: Embedding model for vector queries.
            corpus: Pre-loaded corpus for BM25 indexing. Each dict must have
                a ``"text"`` key. If None, BM25 is skipped.
        """
        self._qdrant = qdrant
        self._embedder = embedder
        self._bm25: BM25Okapi | None = None
        self._corpus = corpus or []

        if self._corpus:
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build the BM25 index from the corpus."""
        tokenized = [doc["text"].lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        log.info("bm25_index_built", corpus_size=len(self._corpus))

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Retrieve passages using the specified strategy.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            mode: Retrieval strategy — ``"kb_only"``, ``"web_only"``,
                or ``"hybrid"`` (default).

        Returns:
            List of passage dicts sorted by fused score, with source metadata.
        """
        ranked_lists: list[list[dict[str, Any]]] = []

        if mode in ("kb_only", "hybrid"):
            # Vector search via Qdrant
            if self._qdrant is not None and self._embedder is not None:
                vector_results = self._vector_search(query, top_k=top_k)
                if vector_results:
                    ranked_lists.append(vector_results)

            # BM25 lexical search
            if self._bm25 is not None:
                bm25_results = self._bm25_search(query, top_k=top_k)
                if bm25_results:
                    ranked_lists.append(bm25_results)

        if mode in ("web_only", "hybrid"):
            web_results = web_search_structured(query, max_results=min(top_k, 5))
            if web_results:
                ranked_lists.append(web_results)

        if not ranked_lists:
            return []

        # Fuse with Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(ranked_lists, k=_RRF_K)

        log.info(
            "hybrid_retrieval_done",
            query=query[:60],
            mode=mode,
            num_lists=len(ranked_lists),
            fused_results=len(fused),
        )
        return fused[:top_k]

    def _vector_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Run vector search via Qdrant.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of passage dicts from vector search.
        """
        try:
            embedding = self._embedder.embed_text(query)
            results = self._qdrant.search(embedding, top_k=top_k)
            # Tag source
            for r in results:
                r["source"] = r.get("source", "knowledge_base")
            return results
        except Exception as exc:
            log.error("vector_search_failed", error=str(exc))
            return []

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Run BM25 lexical search on the corpus.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of passage dicts from BM25 search.
        """
        try:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)

            # Get top-k indices
            scored_indices = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True,
            )[:top_k]

            results: list[dict[str, Any]] = []
            for idx, score in scored_indices:
                if score <= 0:
                    continue
                doc = dict(self._corpus[idx])  # copy
                doc["score"] = float(score)
                doc["source"] = doc.get("source", "knowledge_base_bm25")
                results.append(doc)

            return results
        except Exception as exc:
            log.error("bm25_search_failed", error=str(exc))
            return []


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over all lists L of: 1 / (k + rank_L(d))

    This is the standard fusion method used in production search systems.

    Args:
        ranked_lists: Multiple ranked result lists to fuse.
        k: RRF constant (default 60, from the original paper).

    Returns:
        Single fused and re-ranked list of unique documents.
    """
    doc_scores: dict[str, float] = {}
    doc_data: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, 1):
            doc_id = _get_doc_key(doc)
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_score

            # Keep the doc with the best data (prefer higher original score)
            existing = doc_data.get(doc_id)
            if existing is None or doc.get("score", 0) > existing.get("score", 0):
                doc_data[doc_id] = doc

    # Sort by fused RRF score
    sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)

    results: list[dict[str, Any]] = []
    for doc_id in sorted_ids:
        doc = dict(doc_data[doc_id])
        doc["rrf_score"] = round(doc_scores[doc_id], 6)
        results.append(doc)

    return results


def _get_doc_key(doc: dict[str, Any]) -> str:
    """Generate a unique key for deduplication across sources."""
    doc_id = doc.get("id")
    if doc_id:
        return str(doc_id)
    # Fallback: hash the text
    text = doc.get("text", "")
    return str(hash(text))
