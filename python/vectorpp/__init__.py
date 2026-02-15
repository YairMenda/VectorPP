"""Vector++ Python Client SDK.

A Python client for the Vector++ high-performance in-memory vector database.
"""

from .client import (
    VectorPPClient,
    SearchResult,
    VectorPPError,
    ConnectionError,
    DimensionMismatchError,
    VectorNotFoundError,
    CapacityExceededError,
)

from .embeddings import (
    OpenAIEmbeddings,
    get_openai_embedding,
    get_openai_embeddings_batch,
    EmbeddingError,
    OpenAIError,
    MissingAPIKeyError,
)

__version__ = "0.1.0"

__all__ = [
    # Client
    "VectorPPClient",
    "SearchResult",
    "VectorPPError",
    "ConnectionError",
    "DimensionMismatchError",
    "VectorNotFoundError",
    "CapacityExceededError",
    # Embeddings - OpenAI
    "OpenAIEmbeddings",
    "get_openai_embedding",
    "get_openai_embeddings_batch",
    "EmbeddingError",
    "OpenAIError",
    "MissingAPIKeyError",
]
