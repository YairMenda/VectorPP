"""Embedding helper functions for Vector++.

Provides helper functions to generate embeddings using OpenAI API and
HuggingFace sentence-transformers.
"""

from typing import List, Optional
import os


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class OpenAIError(EmbeddingError):
    """Error from OpenAI API."""
    pass


class MissingAPIKeyError(EmbeddingError):
    """API key not found."""
    pass


class OpenAIEmbeddings:
    """Generate embeddings using OpenAI API.

    Requires the 'openai' package to be installed:
        pip install openai

    Args:
        api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
        model: The embedding model to use. Defaults to "text-embedding-3-small".

    Example:
        >>> embeddings = OpenAIEmbeddings()
        >>> vector = embeddings.embed("Hello, world!")
        >>> vectors = embeddings.embed_batch(["Hello", "World"])
    """

    # Common OpenAI embedding models and their dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small"
    ):
        """Initialize OpenAI embeddings client.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model: OpenAI embedding model name.

        Raises:
            MissingAPIKeyError: If no API key is provided or found in environment.
            ImportError: If openai package is not installed.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise MissingAPIKeyError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self._client = None

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions for the current model."""
        return self.MODEL_DIMENSIONS.get(self.model, 1536)

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            OpenAIError: If the API call fails.
        """
        try:
            client = self._get_client()
            response = client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except ImportError:
            raise
        except Exception as e:
            raise OpenAIError(f"OpenAI API error: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            OpenAIError: If the API call fails.
        """
        if not texts:
            return []

        try:
            client = self._get_client()
            response = client.embeddings.create(
                input=texts,
                model=self.model
            )
            # Sort by index to ensure order matches input
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except ImportError:
            raise
        except Exception as e:
            raise OpenAIError(f"OpenAI API error: {e}") from e


def get_openai_embedding(
    text: str,
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small"
) -> List[float]:
    """Convenience function to get a single OpenAI embedding.

    Args:
        text: The text to embed.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: OpenAI embedding model name.

    Returns:
        List of floats representing the embedding vector.

    Example:
        >>> vector = get_openai_embedding("Hello, world!")
        >>> len(vector)
        1536
    """
    embeddings = OpenAIEmbeddings(api_key=api_key, model=model)
    return embeddings.embed(text)


def get_openai_embeddings_batch(
    texts: List[str],
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Convenience function to get multiple OpenAI embeddings.

    Args:
        texts: List of texts to embed.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: OpenAI embedding model name.

    Returns:
        List of embedding vectors, one per input text.

    Example:
        >>> vectors = get_openai_embeddings_batch(["Hello", "World"])
        >>> len(vectors)
        2
    """
    embeddings = OpenAIEmbeddings(api_key=api_key, model=model)
    return embeddings.embed_batch(texts)
