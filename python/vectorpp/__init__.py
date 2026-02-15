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

__version__ = "0.1.0"

__all__ = [
    "VectorPPClient",
    "SearchResult",
    "VectorPPError",
    "ConnectionError",
    "DimensionMismatchError",
    "VectorNotFoundError",
    "CapacityExceededError",
]
