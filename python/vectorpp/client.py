"""Vector++ Python Client SDK.

Provides a simple client interface to interact with the Vector++ gRPC server.
"""

from typing import List, Optional
from dataclasses import dataclass

import grpc

from . import vectordb_pb2
from . import vectordb_pb2_grpc


class VectorPPError(Exception):
    """Base exception for VectorPP client errors."""
    pass


class ConnectionError(VectorPPError):
    """Raised when unable to connect to the server."""
    pass


class DimensionMismatchError(VectorPPError):
    """Raised when vector dimensions don't match the database configuration."""
    pass


class VectorNotFoundError(VectorPPError):
    """Raised when a vector ID is not found."""
    pass


class CapacityExceededError(VectorPPError):
    """Raised when the database has reached its capacity limit."""
    pass


@dataclass
class SearchResult:
    """A single search result containing vector ID, similarity score, and metadata."""
    id: str
    score: float
    metadata: str


class VectorPPClient:
    """Client for interacting with a Vector++ gRPC server.

    Example:
        >>> client = VectorPPClient("localhost:50051")
        >>> vector_id = client.insert([0.1, 0.2, 0.3], metadata="category1")
        >>> results = client.search([0.1, 0.2, 0.3], k=5)
        >>> success = client.delete(vector_id)
    """

    def __init__(self, host: str = "localhost", port: int = 50051):
        """Initialize the VectorPP client.

        Args:
            host: Server hostname or IP address.
            port: Server port number.
        """
        self._address = f"{host}:{port}"
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[vectordb_pb2_grpc.VectorDBStub] = None

    def connect(self) -> None:
        """Establish connection to the Vector++ server.

        Raises:
            ConnectionError: If unable to connect to the server.
        """
        try:
            self._channel = grpc.insecure_channel(self._address)
            self._stub = vectordb_pb2_grpc.VectorDBStub(self._channel)
        except grpc.RpcError as e:
            raise ConnectionError(f"Failed to connect to {self._address}: {e}")

    def close(self) -> None:
        """Close the connection to the server."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> "VectorPPClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def _ensure_connected(self) -> None:
        """Ensure the client is connected, auto-connecting if needed."""
        if self._stub is None:
            self.connect()

    def _handle_grpc_error(self, e: grpc.RpcError) -> None:
        """Convert gRPC errors to appropriate Python exceptions."""
        code = e.code()
        details = e.details()

        if code == grpc.StatusCode.INVALID_ARGUMENT:
            if "dimension" in details.lower():
                raise DimensionMismatchError(details)
            raise VectorPPError(details)
        elif code == grpc.StatusCode.NOT_FOUND:
            raise VectorNotFoundError(details)
        elif code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            raise CapacityExceededError(details)
        elif code == grpc.StatusCode.UNAVAILABLE:
            raise ConnectionError(f"Server unavailable: {details}")
        else:
            raise VectorPPError(f"gRPC error ({code}): {details}")

    def insert(self, vector: List[float], metadata: str = "") -> str:
        """Insert a vector into the database.

        Args:
            vector: The embedding vector as a list of floats.
            metadata: Optional metadata string (e.g., category).

        Returns:
            The UUID assigned to the inserted vector.

        Raises:
            DimensionMismatchError: If vector dimensions don't match database config.
            CapacityExceededError: If the database has reached its capacity.
            ConnectionError: If unable to connect to the server.
            VectorPPError: For other errors.
        """
        self._ensure_connected()

        request = vectordb_pb2.InsertRequest(
            vector=vector,
            metadata=metadata
        )

        try:
            response = self._stub.Insert(request)
            return response.id
        except grpc.RpcError as e:
            self._handle_grpc_error(e)

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filter_metadata: str = ""
    ) -> List[SearchResult]:
        """Search for the most similar vectors.

        Args:
            query_vector: The query embedding vector.
            k: Number of results to return (default 10).
            filter_metadata: Optional metadata filter string.

        Returns:
            List of SearchResult objects sorted by similarity (highest first).

        Raises:
            DimensionMismatchError: If vector dimensions don't match database config.
            ConnectionError: If unable to connect to the server.
            VectorPPError: For other errors.
        """
        self._ensure_connected()

        request = vectordb_pb2.SearchRequest(
            query_vector=query_vector,
            top_k=k,
            filter_metadata=filter_metadata
        )

        try:
            response = self._stub.Search(request)
            return [
                SearchResult(
                    id=result.id,
                    score=result.score,
                    metadata=result.metadata
                )
                for result in response.results
            ]
        except grpc.RpcError as e:
            self._handle_grpc_error(e)

    def delete(self, vector_id: str) -> bool:
        """Delete a vector by its ID.

        Args:
            vector_id: The UUID of the vector to delete.

        Returns:
            True if the vector was deleted successfully.

        Raises:
            VectorNotFoundError: If the vector ID was not found.
            ConnectionError: If unable to connect to the server.
            VectorPPError: For other errors.
        """
        self._ensure_connected()

        request = vectordb_pb2.DeleteRequest(id=vector_id)

        try:
            response = self._stub.Delete(request)
            return response.success
        except grpc.RpcError as e:
            self._handle_grpc_error(e)
