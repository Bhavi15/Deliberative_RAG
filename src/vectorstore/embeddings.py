"""Sentence-transformer embedding wrapper.

Provides a thin interface over sentence-transformers so the rest of the
codebase is not coupled to the underlying library.
"""

from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a sentence-transformers model with a consistent interface."""

    def __init__(self, model_name: str) -> None:
        """Load the sentence-transformer model.

        Args:
            model_name: HuggingFace model ID or local path.
        """
        pass

    def embed(self, text: str) -> list[float]:
        """Embed a single string.

        Args:
            text: Text to embed.

        Returns:
            Dense embedding vector as a list of floats.
        """
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings in a single forward pass.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors in the same order as input.
        """
        pass

    @property
    def vector_size(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        pass
