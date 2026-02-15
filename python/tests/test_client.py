"""Tests for the VectorPP Python client."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import grpc

from vectorpp import (
    VectorPPClient,
    SearchResult,
    VectorPPError,
    ConnectionError,
    DimensionMismatchError,
    VectorNotFoundError,
    CapacityExceededError,
)
from vectorpp import vectordb_pb2


class TestVectorPPClientImports(unittest.TestCase):
    """Test that all client components can be imported."""

    def test_client_class_exists(self):
        """VectorPPClient class should be importable."""
        self.assertTrue(callable(VectorPPClient))

    def test_search_result_dataclass(self):
        """SearchResult should be a usable dataclass."""
        result = SearchResult(id="test-id", score=0.95, metadata="category")
        self.assertEqual(result.id, "test-id")
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.metadata, "category")

    def test_exception_hierarchy(self):
        """All exceptions should inherit from VectorPPError."""
        self.assertTrue(issubclass(ConnectionError, VectorPPError))
        self.assertTrue(issubclass(DimensionMismatchError, VectorPPError))
        self.assertTrue(issubclass(VectorNotFoundError, VectorPPError))
        self.assertTrue(issubclass(CapacityExceededError, VectorPPError))


class TestVectorPPClientInit(unittest.TestCase):
    """Test client initialization."""

    def test_default_init(self):
        """Client should initialize with default host and port."""
        client = VectorPPClient()
        self.assertEqual(client._address, "localhost:50051")
        self.assertIsNone(client._channel)
        self.assertIsNone(client._stub)

    def test_custom_host_port(self):
        """Client should accept custom host and port."""
        client = VectorPPClient(host="192.168.1.100", port=8080)
        self.assertEqual(client._address, "192.168.1.100:8080")


class TestVectorPPClientMethods(unittest.TestCase):
    """Test client methods with mocked gRPC."""

    def setUp(self):
        """Set up mocked client for each test."""
        self.client = VectorPPClient()
        self.mock_stub = Mock()
        self.client._stub = self.mock_stub
        self.client._channel = Mock()

    def test_insert_returns_id(self):
        """Insert should return the UUID from the server."""
        mock_response = Mock()
        mock_response.id = "test-uuid-123"
        self.mock_stub.Insert.return_value = mock_response

        result = self.client.insert([0.1, 0.2, 0.3], metadata="test")

        self.assertEqual(result, "test-uuid-123")
        self.mock_stub.Insert.assert_called_once()
        call_args = self.mock_stub.Insert.call_args[0][0]
        # Use almostEqual for floats (protobuf uses 32-bit floats)
        self.assertEqual(len(call_args.vector), 3)
        self.assertAlmostEqual(call_args.vector[0], 0.1, places=5)
        self.assertAlmostEqual(call_args.vector[1], 0.2, places=5)
        self.assertAlmostEqual(call_args.vector[2], 0.3, places=5)
        self.assertEqual(call_args.metadata, "test")

    def test_search_returns_results(self):
        """Search should return list of SearchResult objects."""
        mock_result1 = Mock(id="id1", score=0.95, metadata="cat1")
        mock_result2 = Mock(id="id2", score=0.85, metadata="cat2")
        mock_response = Mock()
        mock_response.results = [mock_result1, mock_result2]
        self.mock_stub.Search.return_value = mock_response

        results = self.client.search([0.1, 0.2, 0.3], k=5)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].id, "id1")
        self.assertEqual(results[0].score, 0.95)
        self.assertEqual(results[1].id, "id2")

    def test_search_with_filter(self):
        """Search should pass filter_metadata to the server."""
        mock_response = Mock()
        mock_response.results = []
        self.mock_stub.Search.return_value = mock_response

        self.client.search([0.1, 0.2], k=10, filter_metadata="movies")

        call_args = self.mock_stub.Search.call_args[0][0]
        self.assertEqual(call_args.filter_metadata, "movies")

    def test_delete_returns_success(self):
        """Delete should return the success boolean from server."""
        mock_response = Mock()
        mock_response.success = True
        self.mock_stub.Delete.return_value = mock_response

        result = self.client.delete("test-uuid")

        self.assertTrue(result)
        call_args = self.mock_stub.Delete.call_args[0][0]
        self.assertEqual(call_args.id, "test-uuid")

    def test_delete_not_found(self):
        """Delete should raise VectorNotFoundError for NOT_FOUND status."""
        mock_error = grpc.RpcError()
        mock_error.code = Mock(return_value=grpc.StatusCode.NOT_FOUND)
        mock_error.details = Mock(return_value="Vector not found")
        self.mock_stub.Delete.side_effect = mock_error

        with self.assertRaises(VectorNotFoundError):
            self.client.delete("nonexistent-uuid")

    def test_insert_dimension_mismatch(self):
        """Insert should raise DimensionMismatchError for dimension errors."""
        mock_error = grpc.RpcError()
        mock_error.code = Mock(return_value=grpc.StatusCode.INVALID_ARGUMENT)
        mock_error.details = Mock(return_value="Vector dimension mismatch")
        self.mock_stub.Insert.side_effect = mock_error

        with self.assertRaises(DimensionMismatchError):
            self.client.insert([0.1, 0.2])

    def test_insert_capacity_exceeded(self):
        """Insert should raise CapacityExceededError when limit reached."""
        mock_error = grpc.RpcError()
        mock_error.code = Mock(return_value=grpc.StatusCode.RESOURCE_EXHAUSTED)
        mock_error.details = Mock(return_value="Capacity limit reached")
        self.mock_stub.Insert.side_effect = mock_error

        with self.assertRaises(CapacityExceededError):
            self.client.insert([0.1, 0.2, 0.3])


class TestVectorPPClientContextManager(unittest.TestCase):
    """Test context manager functionality."""

    @patch("vectorpp.client.grpc.insecure_channel")
    @patch("vectorpp.client.vectordb_pb2_grpc.VectorDBStub")
    def test_context_manager(self, mock_stub_class, mock_channel):
        """Client should work as a context manager."""
        mock_channel_instance = Mock()
        mock_channel.return_value = mock_channel_instance

        with VectorPPClient() as client:
            self.assertIsNotNone(client._channel)
            self.assertIsNotNone(client._stub)

        mock_channel_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
