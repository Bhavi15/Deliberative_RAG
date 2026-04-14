"""Vectorless retrieval — cross-encoder reranking without pre-computed embeddings.

Implements retrieval directly on uploaded or in-memory documents using:
1. **BM25 first-pass** — fast lexical candidate selection
2. **Cross-encoder reranking** — accurate query-document scoring without vectors

This module is designed for the uploaded document use case where users provide
conflicting documents at query time and need immediate analysis without
indexing into a vector database.

Why vectorless?
- No pre-embedding step — works on raw text at query time
- Cross-encoders jointly encode (query, passage) pairs for higher accuracy
  than bi-encoder cosine similarity
- Better for small, dynamic document sets (uploads, live web results)
- Complementary to vector search on the indexed KB

Usage::

    retriever = VectorlessRetriever()
    retriever.add_documents(passages)  # list of dicts with "text" key
    results = retriever.retrieve("my query", top_k=5)
"""

from __future__ import annotations

from typing import Any

import structlog
from rank_bm25 import BM25Okapi

log = structlog.get_logger()


class VectorlessRetriever:
    """Retrieves documents without vector embeddings using BM25 + cross-encoder reranking.

    Two-stage pipeline:
    1. BM25 candidate selection (fast, lexical)
    2. Cross-encoder reranking (accurate, model-based)

    If no cross-encoder model is available, falls back to BM25-only ranking.
    """

    def __init__(
        self,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_candidates: int = 20,
    ) -> None:
        """Initialise the vectorless retriever.

        Args:
            use_cross_encoder: Whether to use a cross-encoder for reranking.
                If False, only BM25 ranking is used.
            cross_encoder_model: HuggingFace model name for the cross-encoder.
                Default is a small, fast model suitable for CPU inference.
            bm25_candidates: Number of BM25 candidates to pass to the
                cross-encoder for reranking.
        """
        self._documents: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._cross_encoder = None
        self._use_cross_encoder = use_cross_encoder
        self._cross_encoder_model_name = cross_encoder_model
        self._bm25_candidates = bm25_candidates

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents to the retriever's in-memory index.

        Each document must have a ``"text"`` key. Additional metadata
        (``"id"``, ``"source"``, ``"source_type"``, etc.) is preserved.

        Args:
            documents: List of document dicts to index.
        """
        self._documents.extend(documents)
        self._rebuild_bm25()
        log.info("vectorless_docs_added", total=len(self._documents))

    def clear(self) -> None:
        """Remove all documents from the index."""
        self._documents.clear()
        self._bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant documents for a query.

        Pipeline:
        1. BM25 selects top-N candidates from all documents
        2. Cross-encoder rescores each (query, candidate) pair
        3. Results sorted by cross-encoder score

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of document dicts with added ``"score"`` and
            ``"retrieval_method"`` fields, sorted by relevance.
        """
        if not self._documents:
            return []

        # Stage 1: BM25 candidate selection
        bm25_results = self._bm25_search(query, top_k=self._bm25_candidates)

        if not bm25_results:
            return []

        # Stage 2: Cross-encoder reranking (if enabled)
        if self._use_cross_encoder and len(bm25_results) > 1:
            reranked = self._cross_encoder_rerank(query, bm25_results)
            if reranked:
                return reranked[:top_k]

        # Fallback: BM25-only results
        return bm25_results[:top_k]

    def retrieve_all_scored(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Score ALL documents against the query without BM25 pre-filtering.

        Useful when the document set is small (< 50 docs) and you want
        the most accurate ranking. Uses cross-encoder directly on every doc.

        Args:
            query: The search query.

        Returns:
            All documents scored and sorted by relevance.
        """
        if not self._documents:
            return []

        if self._use_cross_encoder:
            reranked = self._cross_encoder_rerank(query, self._documents)
            if reranked:
                return reranked

        # Fallback: BM25 on everything
        return self._bm25_search(query, top_k=len(self._documents))

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from the current document set."""
        if not self._documents:
            self._bm25 = None
            return

        tokenized = [
            doc.get("text", "").lower().split()
            for doc in self._documents
        ]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Run BM25 search on the document set.

        Args:
            query: Search query.
            top_k: Max results.

        Returns:
            Top-k documents sorted by BM25 score.
        """
        if self._bm25 is None:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            doc = dict(self._documents[idx])
            doc["score"] = float(score)
            doc["retrieval_method"] = "bm25"
            results.append(doc)

        return results

    def _cross_encoder_rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Rerank candidates using a cross-encoder model.

        The cross-encoder jointly encodes (query, passage) pairs and
        outputs a relevance score. This is more accurate than cosine
        similarity on independent embeddings because the model can
        attend across both inputs.

        Args:
            query: The search query.
            candidates: BM25 candidate documents to rerank.

        Returns:
            Reranked documents sorted by cross-encoder score, or None
            if the cross-encoder is unavailable.
        """
        if self._cross_encoder is None:
            self._cross_encoder = self._load_cross_encoder()

        if self._cross_encoder is None:
            return None

        try:
            # Build (query, passage) pairs
            pairs = [
                (query, doc.get("text", "")[:512])  # truncate for efficiency
                for doc in candidates
            ]

            # Score all pairs
            scores = self._cross_encoder.predict(pairs)

            # Attach scores and sort
            reranked: list[dict[str, Any]] = []
            for doc, score in zip(candidates, scores):
                doc_copy = dict(doc)
                doc_copy["score"] = float(score)
                doc_copy["retrieval_method"] = "cross_encoder"
                reranked.append(doc_copy)

            reranked.sort(key=lambda d: d["score"], reverse=True)
            return reranked

        except Exception as exc:
            log.warning("cross_encoder_rerank_failed", error=str(exc))
            return None

    def _load_cross_encoder(self):
        """Lazy-load the cross-encoder model.

        Returns:
            CrossEncoder instance, or None if loading fails.
        """
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(
                self._cross_encoder_model_name,
                max_length=512,
            )
            log.info(
                "cross_encoder_loaded",
                model=self._cross_encoder_model_name,
            )
            return model
        except Exception as exc:
            log.warning(
                "cross_encoder_load_failed",
                model=self._cross_encoder_model_name,
                error=str(exc),
            )
            return None


# ---------------------------------------------------------------------------
# Convenience function for uploaded document analysis
# ---------------------------------------------------------------------------


def analyze_uploaded_documents(
    query: str,
    file_paths: list[str],
    top_k: int = 10,
    use_cross_encoder: bool = True,
) -> list[dict[str, Any]]:
    """Parse uploaded documents and retrieve relevant passages without vectors.

    End-to-end convenience function that:
    1. Parses each file into chunks
    2. Builds a BM25 index + cross-encoder reranker
    3. Retrieves the most relevant chunks for the query

    Args:
        query: The user's search query.
        file_paths: Paths to uploaded document files.
        top_k: Number of top passages to return.
        use_cross_encoder: Whether to use cross-encoder reranking.

    Returns:
        List of relevant passage dicts sorted by relevance score.
    """
    from src.tools.document_parser import parse_document_structured

    all_passages: list[dict[str, Any]] = []
    for fp in file_paths:
        passages = parse_document_structured(fp)
        all_passages.extend(passages)

    if not all_passages:
        return []

    retriever = VectorlessRetriever(use_cross_encoder=use_cross_encoder)
    retriever.add_documents(all_passages)

    # For small doc sets (< 50 passages), score everything directly
    if len(all_passages) <= 50:
        return retriever.retrieve_all_scored(query)[:top_k]

    return retriever.retrieve(query, top_k=top_k)
