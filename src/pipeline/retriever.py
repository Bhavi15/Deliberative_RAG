"""Stage 2 — Multi-Strategy Retrieval.

Executes vector search against Qdrant for each sub-query, then merges
and deduplicates the resulting passages.
"""

from __future__ import annotations

import structlog

from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.qdrant_client import QdrantManager

log = structlog.get_logger()


def retrieve_for_sub_query(
    sub_query: str,
    qdrant: QdrantManager,
    embedder: EmbeddingModel,
    top_k: int = 10,
) -> list[dict]:
    """Run vector search for a single sub-query.

    Args:
        sub_query: The sub-query text to retrieve passages for.
        qdrant: Qdrant manager instance.
        embedder: Embedding model for encoding the query.
        top_k: Maximum number of results per query.

    Returns:
        List of passage dicts with ``"text"``, ``"score"``, and metadata.
    """
    query_embedding = embedder.embed_text(sub_query)
    results = qdrant.search(query_embedding, top_k=top_k)
    log.info("retrieval_done", sub_query=sub_query[:60], hits=len(results))
    return results


def merge_and_deduplicate(passage_groups: list[list[dict]]) -> list[dict]:
    """Merge passage lists from multiple sub-queries and remove duplicates.

    Deduplicates by passage ``id`` (or ``text`` hash if id is absent),
    keeping the highest-scoring copy.

    Args:
        passage_groups: Per-sub-query passage lists.

    Returns:
        Flat, deduplicated list of passage dicts sorted by score descending.
    """
    seen: dict[str, dict] = {}
    for group in passage_groups:
        for passage in group:
            key = str(passage.get("id", hash(passage.get("text", ""))))
            existing = seen.get(key)
            if existing is None or passage.get("score", 0) > existing.get("score", 0):
                seen[key] = passage

    merged = sorted(seen.values(), key=lambda p: p.get("score", 0), reverse=True)
    return merged
