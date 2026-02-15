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
    HuggingFaceEmbeddings,
    get_huggingface_embedding,
    get_huggingface_embeddings_batch,
    EmbeddingError,
    OpenAIError,
    HuggingFaceError,
    MissingAPIKeyError,
)
import numpy as np


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

    def test_huggingface_error_inherits_from_embedding_error(self):
        """HuggingFaceError should inherit from EmbeddingError."""
        self.assertTrue(issubclass(HuggingFaceError, EmbeddingError))


# ============================================================================
# HuggingFace Embeddings Tests
# ============================================================================


class TestHuggingFaceEmbeddingsInit(unittest.TestCase):
    """Test HuggingFaceEmbeddings initialization."""

    def test_default_model(self):
        """Should use all-MiniLM-L6-v2 as default model."""
        embeddings = HuggingFaceEmbeddings()
        self.assertEqual(embeddings.model_name, "all-MiniLM-L6-v2")

    def test_custom_model(self):
        """Should accept custom model name."""
        embeddings = HuggingFaceEmbeddings(model="all-mpnet-base-v2")
        self.assertEqual(embeddings.model_name, "all-mpnet-base-v2")

    def test_custom_device(self):
        """Should accept custom device."""
        embeddings = HuggingFaceEmbeddings(device="cpu")
        self.assertEqual(embeddings.device, "cpu")

    def test_default_device_is_none(self):
        """Should default to None device (auto-detect)."""
        embeddings = HuggingFaceEmbeddings()
        self.assertIsNone(embeddings.device)


class TestHuggingFaceEmbeddingsDimensions(unittest.TestCase):
    """Test HuggingFace model dimension properties."""

    def test_minilm_l6_dimensions(self):
        """all-MiniLM-L6-v2 should have 384 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        self.assertEqual(embeddings.dimensions, 384)

    def test_minilm_l12_dimensions(self):
        """all-MiniLM-L12-v2 should have 384 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L12-v2")
        self.assertEqual(embeddings.dimensions, 384)

    def test_mpnet_dimensions(self):
        """all-mpnet-base-v2 should have 768 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="all-mpnet-base-v2")
        self.assertEqual(embeddings.dimensions, 768)

    def test_paraphrase_minilm_dimensions(self):
        """paraphrase-MiniLM-L6-v2 should have 384 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="paraphrase-MiniLM-L6-v2")
        self.assertEqual(embeddings.dimensions, 384)

    def test_paraphrase_mpnet_dimensions(self):
        """paraphrase-mpnet-base-v2 should have 768 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="paraphrase-mpnet-base-v2")
        self.assertEqual(embeddings.dimensions, 768)

    def test_multi_qa_minilm_dimensions(self):
        """multi-qa-MiniLM-L6-cos-v1 should have 384 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="multi-qa-MiniLM-L6-cos-v1")
        self.assertEqual(embeddings.dimensions, 384)

    def test_multi_qa_mpnet_dimensions(self):
        """multi-qa-mpnet-base-cos-v1 should have 768 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="multi-qa-mpnet-base-cos-v1")
        self.assertEqual(embeddings.dimensions, 768)

    def test_unknown_model_default_dimensions(self):
        """Unknown models should default to 384 dimensions."""
        embeddings = HuggingFaceEmbeddings(model="unknown-model")
        self.assertEqual(embeddings.dimensions, 384)


class TestHuggingFaceEmbeddingsEmbed(unittest.TestCase):
    """Test single text embedding with HuggingFace."""

    def setUp(self):
        """Set up embeddings instance with mocked model."""
        self.embeddings = HuggingFaceEmbeddings()
        # Inject a mock model directly
        self.mock_model = Mock()
        self.embeddings._model = self.mock_model

    def test_embed_returns_vector(self):
        """embed() should return the embedding vector as a list."""
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.mock_model.encode.return_value = mock_embedding

        result = self.embeddings.embed("Hello, world!")

        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.mock_model.encode.assert_called_once_with(
            "Hello, world!",
            convert_to_numpy=True
        )

    def test_embed_error_raises_huggingface_error(self):
        """embed() should raise HuggingFaceError on failure."""
        self.mock_model.encode.side_effect = Exception("Model loading failed")

        with self.assertRaises(HuggingFaceError) as ctx:
            self.embeddings.embed("Test text")
        self.assertIn("Model loading failed", str(ctx.exception))


class TestHuggingFaceEmbeddingsEmbedBatch(unittest.TestCase):
    """Test batch embedding with HuggingFace."""

    def setUp(self):
        """Set up embeddings instance with mocked model."""
        self.embeddings = HuggingFaceEmbeddings()
        # Inject a mock model directly
        self.mock_model = Mock()
        self.embeddings._model = self.mock_model

    def test_embed_batch_returns_vectors(self):
        """embed_batch() should return list of embedding vectors."""
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        self.mock_model.encode.return_value = mock_embeddings

        result = self.embeddings.embed_batch(["Hello", "World"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])

    def test_embed_batch_empty_list(self):
        """embed_batch() should return empty list for empty input."""
        result = self.embeddings.embed_batch([])
        self.assertEqual(result, [])

    def test_embed_batch_error_raises_huggingface_error(self):
        """embed_batch() should raise HuggingFaceError on failure."""
        self.mock_model.encode.side_effect = Exception("Batch processing failed")

        with self.assertRaises(HuggingFaceError):
            self.embeddings.embed_batch(["Test 1", "Test 2"])


class TestHuggingFaceConvenienceFunctions(unittest.TestCase):
    """Test HuggingFace convenience functions."""

    @patch("vectorpp.embeddings.HuggingFaceEmbeddings")
    def test_get_huggingface_embedding(self, mock_embeddings_class):
        """get_huggingface_embedding() should create embeddings and call embed()."""
        mock_instance = Mock()
        mock_instance.embed.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_instance

        result = get_huggingface_embedding("Test text")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_embeddings_class.assert_called_once_with(
            model="all-MiniLM-L6-v2",
            device=None
        )
        mock_instance.embed.assert_called_once_with("Test text")

    @patch("vectorpp.embeddings.HuggingFaceEmbeddings")
    def test_get_huggingface_embeddings_batch(self, mock_embeddings_class):
        """get_huggingface_embeddings_batch() should create embeddings and call embed_batch()."""
        mock_instance = Mock()
        mock_instance.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings_class.return_value = mock_instance

        result = get_huggingface_embeddings_batch(["Text 1", "Text 2"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        mock_embeddings_class.assert_called_once_with(
            model="all-MiniLM-L6-v2",
            device=None
        )
        mock_instance.embed_batch.assert_called_once_with(["Text 1", "Text 2"])

    @patch("vectorpp.embeddings.HuggingFaceEmbeddings")
    def test_get_huggingface_embedding_custom_model(self, mock_embeddings_class):
        """get_huggingface_embedding() should pass custom model."""
        mock_instance = Mock()
        mock_instance.embed.return_value = [0.1]
        mock_embeddings_class.return_value = mock_instance

        get_huggingface_embedding("Test", model="all-mpnet-base-v2")

        mock_embeddings_class.assert_called_once_with(
            model="all-mpnet-base-v2",
            device=None
        )

    @patch("vectorpp.embeddings.HuggingFaceEmbeddings")
    def test_get_huggingface_embedding_custom_device(self, mock_embeddings_class):
        """get_huggingface_embedding() should pass custom device."""
        mock_instance = Mock()
        mock_instance.embed.return_value = [0.1]
        mock_embeddings_class.return_value = mock_instance

        get_huggingface_embedding("Test", device="cuda")

        mock_embeddings_class.assert_called_once_with(
            model="all-MiniLM-L6-v2",
            device="cuda"
        )


if __name__ == "__main__":
    unittest.main()
