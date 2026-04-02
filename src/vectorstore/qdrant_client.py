"""Qdrant vector store wrapper.

Abstracts collection management, document indexing, and similarity search
against a local (or remote) Qdrant instance.
"""

from qdrant_client import QdrantClient

from src.schemas import Document, Passage


class QdrantStore:
    """Thin wrapper around QdrantClient for this project's data models."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        """Initialise the store with an existing Qdrant client.

        Args:
            client: A configured QdrantClient instance.
            collection_name: Target collection to read from / write to.
        """
        pass

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it does not already exist.

        Args:
            vector_size: Dimension of the embedding vectors to store.
        """
        pass

    def upsert_documents(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Index documents with their pre-computed embeddings.

        Args:
            documents: Document objects to store.
            embeddings: Corresponding embedding vectors (same order).
        """
        pass

    def vector_search(
        self, query_vector: list[float], top_k: int, score_threshold: float
    ) -> list[Passage]:
        """Run nearest-neighbour search and return matching passages.

        Args:
            query_vector: Query embedding vector.
            top_k: Maximum number of results to return.
            score_threshold: Minimum cosine similarity to include.

        Returns:
            List of Passage objects ranked by similarity score.
        """
        pass

    def metadata_filter_search(self, filters: dict, top_k: int) -> list[Passage]:
        """Retrieve passages matching structured metadata filters.

        Args:
            filters: Qdrant filter payload (field conditions).
            top_k: Maximum number of results to return.

        Returns:
            List of Passage objects matching the filter.
        """
        pass
