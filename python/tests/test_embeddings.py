"""Tests for the embeddings helper functions."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vectorpp.embeddings import (
    OpenAIEmbeddings,
    get_openai_embedding,
    get_openai_embeddings_batch,
    EmbeddingError,
    OpenAIError,
    MissingAPIKeyError,
)


class TestOpenAIEmbeddingsInit(unittest.TestCase):
    """Test OpenAIEmbeddings initialization."""

    def test_missing_api_key_raises_error(self):
        """Should raise MissingAPIKeyError when no API key provided."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY if it exists
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            with self.assertRaises(MissingAPIKeyError) as ctx:
                OpenAIEmbeddings()
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_api_key_from_parameter(self):
        """Should accept API key as parameter."""
        embeddings = OpenAIEmbeddings(api_key="test-api-key")
        self.assertEqual(embeddings._api_key, "test-api-key")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"})
    def test_api_key_from_environment(self):
        """Should read API key from environment variable."""
        embeddings = OpenAIEmbeddings()
        self.assertEqual(embeddings._api_key, "env-api-key")

    def test_custom_model(self):
        """Should accept custom model name."""
        embeddings = OpenAIEmbeddings(api_key="test-key", model="text-embedding-3-large")
        self.assertEqual(embeddings.model, "text-embedding-3-large")

    def test_default_model(self):
        """Should use text-embedding-3-small as default model."""
        embeddings = OpenAIEmbeddings(api_key="test-key")
        self.assertEqual(embeddings.model, "text-embedding-3-small")


class TestOpenAIEmbeddingsDimensions(unittest.TestCase):
    """Test model dimension properties."""

    def test_small_model_dimensions(self):
        """text-embedding-3-small should have 1536 dimensions."""
        embeddings = OpenAIEmbeddings(api_key="test-key", model="text-embedding-3-small")
        self.assertEqual(embeddings.dimensions, 1536)

    def test_large_model_dimensions(self):
        """text-embedding-3-large should have 3072 dimensions."""
        embeddings = OpenAIEmbeddings(api_key="test-key", model="text-embedding-3-large")
        self.assertEqual(embeddings.dimensions, 3072)

    def test_ada_model_dimensions(self):
        """text-embedding-ada-002 should have 1536 dimensions."""
        embeddings = OpenAIEmbeddings(api_key="test-key", model="text-embedding-ada-002")
        self.assertEqual(embeddings.dimensions, 1536)

    def test_unknown_model_default_dimensions(self):
        """Unknown models should default to 1536 dimensions."""
        embeddings = OpenAIEmbeddings(api_key="test-key", model="unknown-model")
        self.assertEqual(embeddings.dimensions, 1536)


class TestOpenAIEmbeddingsEmbed(unittest.TestCase):
    """Test single text embedding."""

    def setUp(self):
        """Set up embeddings instance with mocked client."""
        self.embeddings = OpenAIEmbeddings(api_key="test-key")
        # Inject a mock client directly
        self.mock_client = Mock()
        self.embeddings._client = self.mock_client

    def test_embed_returns_vector(self):
        """embed() should return the embedding vector."""
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = Mock()
        mock_response.data = [mock_embedding_data]
        self.mock_client.embeddings.create.return_value = mock_response

        result = self.embeddings.embed("Hello, world!")

        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.mock_client.embeddings.create.assert_called_once_with(
            input="Hello, world!",
            model="text-embedding-3-small"
        )

    def test_embed_api_error_raises_openai_error(self):
        """embed() should raise OpenAIError on API failure."""
        self.mock_client.embeddings.create.side_effect = Exception("API rate limit exceeded")

        with self.assertRaises(OpenAIError) as ctx:
            self.embeddings.embed("Test text")
        self.assertIn("API rate limit exceeded", str(ctx.exception))


class TestOpenAIEmbeddingsEmbedBatch(unittest.TestCase):
    """Test batch embedding."""

    def setUp(self):
        """Set up embeddings instance with mocked client."""
        self.embeddings = OpenAIEmbeddings(api_key="test-key")
        # Inject a mock client directly
        self.mock_client = Mock()
        self.embeddings._client = self.mock_client

    def test_embed_batch_returns_vectors(self):
        """embed_batch() should return list of embedding vectors."""
        mock_data1 = Mock(index=0, embedding=[0.1, 0.2, 0.3])
        mock_data2 = Mock(index=1, embedding=[0.4, 0.5, 0.6])
        mock_response = Mock()
        mock_response.data = [mock_data2, mock_data1]  # Out of order
        self.mock_client.embeddings.create.return_value = mock_response

        result = self.embeddings.embed_batch(["Hello", "World"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])  # Index 0 first
        self.assertEqual(result[1], [0.4, 0.5, 0.6])  # Index 1 second

    def test_embed_batch_empty_list(self):
        """embed_batch() should return empty list for empty input."""
        result = self.embeddings.embed_batch([])
        self.assertEqual(result, [])

    def test_embed_batch_api_error_raises_openai_error(self):
        """embed_batch() should raise OpenAIError on API failure."""
        self.mock_client.embeddings.create.side_effect = Exception("API error")

        with self.assertRaises(OpenAIError):
            self.embeddings.embed_batch(["Test 1", "Test 2"])


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    @patch("vectorpp.embeddings.OpenAIEmbeddings")
    def test_get_openai_embedding(self, mock_embeddings_class):
        """get_openai_embedding() should create embeddings and call embed()."""
        mock_instance = Mock()
        mock_instance.embed.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_instance

        result = get_openai_embedding("Test text", api_key="test-key")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_embeddings_class.assert_called_once_with(
            api_key="test-key",
            model="text-embedding-3-small"
        )
        mock_instance.embed.assert_called_once_with("Test text")

    @patch("vectorpp.embeddings.OpenAIEmbeddings")
    def test_get_openai_embeddings_batch(self, mock_embeddings_class):
        """get_openai_embeddings_batch() should create embeddings and call embed_batch()."""
        mock_instance = Mock()
        mock_instance.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings_class.return_value = mock_instance

        result = get_openai_embeddings_batch(["Text 1", "Text 2"], api_key="test-key")

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        mock_embeddings_class.assert_called_once_with(
            api_key="test-key",
            model="text-embedding-3-small"
        )
        mock_instance.embed_batch.assert_called_once_with(["Text 1", "Text 2"])

    @patch("vectorpp.embeddings.OpenAIEmbeddings")
    def test_get_openai_embedding_custom_model(self, mock_embeddings_class):
        """get_openai_embedding() should pass custom model."""
        mock_instance = Mock()
        mock_instance.embed.return_value = [0.1]
        mock_embeddings_class.return_value = mock_instance

        get_openai_embedding("Test", model="text-embedding-3-large")

        mock_embeddings_class.assert_called_once_with(
            api_key=None,
            model="text-embedding-3-large"
        )


class TestExceptionHierarchy(unittest.TestCase):
    """Test exception class hierarchy."""

    def test_openai_error_inherits_from_embedding_error(self):
        """OpenAIError should inherit from EmbeddingError."""
        self.assertTrue(issubclass(OpenAIError, EmbeddingError))

    def test_missing_api_key_error_inherits_from_embedding_error(self):
        """MissingAPIKeyError should inherit from EmbeddingError."""
        self.assertTrue(issubclass(MissingAPIKeyError, EmbeddingError))


if __name__ == "__main__":
    unittest.main()
