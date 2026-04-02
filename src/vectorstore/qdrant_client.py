"""Qdrant vector store wrapper.

Abstracts collection management, document indexing, and similarity search
against a local-mode Qdrant instance.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.vectorstore.embeddings import EmbeddingModel

QDRANT_PATH = Path("./data/qdrant_data")
COLLECTION_NAME = "eval_documents"
VECTOR_DIM = 384


class QdrantManager:
    """Manages a local Qdrant collection for evaluation documents."""

    def __init__(
        self,
        path: str | Path = QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        vector_dim: int = VECTOR_DIM,
    ) -> None:
        """Initialise Qdrant in local (embedded) mode and ensure the collection exists.

        Args:
            path: Directory for Qdrant's on-disk storage.
            collection_name: Name of the target collection.
            vector_dim: Dimensionality of the stored vectors.
        """
        self._path = Path(path)
        self._collection_name = collection_name
        self._vector_dim = vector_dim

        self._client = QdrantClient(path=str(self._path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_dim,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        """Embed and upsert documents with metadata into the collection.

        Args:
            texts: Raw passage texts (stored in the payload for retrieval).
            embeddings: Pre-computed embedding vectors (same order as texts).
            metadatas: Per-document metadata dicts (same order as texts).

        Returns:
            Number of points upserted.
        """
        points = [
            PointStruct(
                id=uuid.uuid4().hex,
                vector=emb,
                payload={"text": text, **meta},
            )
            for text, emb, meta in zip(texts, embeddings, metadatas)
        ]

        # Qdrant recommends batches of ~100 points
        batch_size = 100
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points[start : start + batch_size],
            )

        return len(points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        """Run nearest-neighbour search with optional metadata filters.

        Args:
            query_embedding: Dense query vector.
            top_k: Maximum number of results.
            filters: Optional ``{field: value}`` dict to filter on payload
                     fields using exact match.

        Returns:
            List of dicts with keys ``"id"``, ``"score"``, ``"text"``, and
            all stored metadata fields.
        """
        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )

        hits = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

        results: list[dict] = []
        for hit in hits:
            results.append({
                "id": hit.id,
                "score": hit.score,
                **(hit.payload or {}),
            })
        return results

    def collection_info(self) -> dict:
        """Return summary info about the collection.

        Returns:
            Dict with ``"name"``, ``"vector_dim"``, ``"point_count"``, and
            ``"status"`` keys.
        """
        info = self._client.get_collection(self._collection_name)
        return {
            "name": self._collection_name,
            "vector_dim": self._vector_dim,
            "point_count": info.points_count,
            "status": info.status.value,
        }

    def close(self) -> None:
        """Close the underlying Qdrant client."""
        self._client.close()
