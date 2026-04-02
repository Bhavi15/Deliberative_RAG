"""Sentence-transformer embedding wrapper.

Provides a thin interface over sentence-transformers so the rest of the
codebase is not coupled to the underlying library.
"""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wraps a sentence-transformers model with a consistent interface."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Load the sentence-transformer model.

        Args:
            model_name: HuggingFace model ID or local path.
        """
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: Text to embed.

        Returns:
            Dense embedding vector as a list of floats.
        """
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of strings efficiently in a single forward pass.

        Args:
            texts: List of strings to embed.
            batch_size: Encoding batch size passed to sentence-transformers.

        Returns:
            List of embedding vectors in the same order as input.
        """
        embeddings = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True,
        )
        return embeddings.tolist()

    @property
    def vector_size(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        return self._model.get_sentence_embedding_dimension()
