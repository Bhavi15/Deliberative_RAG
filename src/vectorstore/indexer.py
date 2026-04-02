"""Document indexing pipeline.

Handles loading raw documents, chunking them into passages, embedding each
passage, and upserting the result into the Qdrant vector store.
"""

from src.schemas import Document, Passage
from src.vectorstore.embeddings import Embedder
from src.vectorstore.qdrant_client import QdrantStore


class Indexer:
    """Orchestrates the full document → passage → embedding → upsert flow."""

    def __init__(self, store: QdrantStore, embedder: Embedder) -> None:
        """Initialise the indexer with a store and embedder.

        Args:
            store: Configured QdrantStore to write into.
            embedder: Embedder instance to generate passage vectors.
        """
        pass

    def index_documents(self, documents: list[Document]) -> int:
        """Chunk, embed, and index a list of documents.

        Args:
            documents: Raw Document objects to process.

        Returns:
            Total number of passages indexed.
        """
        pass

    def chunk_document(self, document: Document) -> list[Passage]:
        """Split a document into overlapping passage chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of Passage objects derived from the document text.
        """
        pass

    def index_dataset(self, dataset_name: str) -> int:
        """Load a benchmark dataset and index all its documents.

        Args:
            dataset_name: One of ``"conflict_qa"``, ``"frames"``, or ``"finance_bench"``.

        Returns:
            Total number of passages indexed.
        """
        pass
