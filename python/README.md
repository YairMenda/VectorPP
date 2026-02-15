# Vector++ Python Client SDK

A Python client for the Vector++ high-performance in-memory vector database.

## Installation

```bash
pip install vectorpp
```

### Optional Dependencies

For embedding generation:

```bash
# OpenAI embeddings
pip install openai

# HuggingFace embeddings (sentence-transformers)
pip install sentence-transformers
```

## Quick Start

```python
from vectorpp import VectorPPClient

# Connect to the server
client = VectorPPClient(host="localhost", port=50051)

# Insert a vector
vector_id = client.insert([0.1, 0.2, 0.3, 0.4], metadata="document1")
print(f"Inserted vector with ID: {vector_id}")

# Search for similar vectors
results = client.search([0.1, 0.2, 0.3, 0.4], k=5)
for result in results:
    print(f"ID: {result.id}, Score: {result.score:.4f}, Metadata: {result.metadata}")

# Delete a vector
success = client.delete(vector_id)
print(f"Deleted: {success}")

# Close the connection
client.close()
```

## Using Context Manager

```python
from vectorpp import VectorPPClient

with VectorPPClient(host="localhost", port=50051) as client:
    vector_id = client.insert([0.1, 0.2, 0.3, 0.4], metadata="example")
    results = client.search([0.1, 0.2, 0.3, 0.4], k=3)
    # Connection automatically closed when exiting the block
```

## API Reference

### VectorPPClient

#### Constructor

```python
VectorPPClient(host: str = "localhost", port: int = 50051)
```

#### Methods

**`insert(vector, metadata="") -> str`**

Insert a vector into the database.

- `vector`: List of floats representing the embedding
- `metadata`: Optional string metadata (e.g., category, label)
- Returns: UUID of the inserted vector

```python
vector_id = client.insert([0.1, 0.2, 0.3], metadata="category1")
```

**`search(query_vector, k=10, filter_metadata="") -> List[SearchResult]`**

Search for similar vectors.

- `query_vector`: List of floats representing the query embedding
- `k`: Number of results to return (default: 10)
- `filter_metadata`: Optional metadata filter
- Returns: List of `SearchResult` objects sorted by similarity (highest first)

```python
results = client.search([0.1, 0.2, 0.3], k=5)
for result in results:
    print(f"ID: {result.id}, Score: {result.score}, Metadata: {result.metadata}")
```

**`delete(vector_id) -> bool`**

Delete a vector by ID.

- `vector_id`: UUID of the vector to delete
- Returns: True if deletion was successful

```python
success = client.delete("550e8400-e29b-41d4-a716-446655440000")
```

**`connect()` / `close()`**

Manually manage the connection lifecycle.

```python
client.connect()
# ... operations ...
client.close()
```

### SearchResult

A dataclass containing search result information:

- `id`: UUID of the matching vector
- `score`: Cosine similarity score (0.0 to 1.0)
- `metadata`: Associated metadata string

## Embedding Helpers

### OpenAI Embeddings

Generate embeddings using OpenAI's API. Requires the `openai` package and an API key.

```python
from vectorpp import OpenAIEmbeddings, get_openai_embedding

# Using the class
embeddings = OpenAIEmbeddings(api_key="sk-...")  # or set OPENAI_API_KEY env var
vector = embeddings.embed("Hello, world!")
vectors = embeddings.embed_batch(["Hello", "World"])

# Using convenience functions
vector = get_openai_embedding("Hello, world!")
vectors = get_openai_embeddings_batch(["Hello", "World"])
```

#### Supported Models

| Model | Dimensions |
|-------|------------|
| text-embedding-3-small (default) | 1536 |
| text-embedding-3-large | 3072 |
| text-embedding-ada-002 | 1536 |

```python
# Using a specific model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
print(embeddings.dimensions)  # 3072
```

### HuggingFace Embeddings

Generate embeddings using HuggingFace sentence-transformers. Runs locally, no API key required.

```python
from vectorpp import HuggingFaceEmbeddings, get_huggingface_embedding

# Using the class
embeddings = HuggingFaceEmbeddings()
vector = embeddings.embed("Hello, world!")
vectors = embeddings.embed_batch(["Hello", "World"])

# Using convenience functions
vector = get_huggingface_embedding("Hello, world!")
vectors = get_huggingface_embeddings_batch(["Hello", "World"])
```

#### Supported Models

| Model | Dimensions |
|-------|------------|
| all-MiniLM-L6-v2 (default) | 384 |
| all-MiniLM-L12-v2 | 384 |
| all-mpnet-base-v2 | 768 |
| paraphrase-MiniLM-L6-v2 | 384 |
| paraphrase-mpnet-base-v2 | 768 |
| multi-qa-MiniLM-L6-cos-v1 | 384 |
| multi-qa-mpnet-base-cos-v1 | 768 |

```python
# Using a specific model with GPU
embeddings = HuggingFaceEmbeddings(model="all-mpnet-base-v2", device="cuda")
print(embeddings.dimensions)  # 768
```

## Complete Example: Document Search

```python
from vectorpp import VectorPPClient, HuggingFaceEmbeddings

# Initialize embedding model
embeddings = HuggingFaceEmbeddings()

# Sample documents
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms to learn from data",
    "Neural networks are inspired by biological brains",
    "Data science combines statistics and programming",
]

# Connect and insert documents
with VectorPPClient() as client:
    # Insert all documents
    doc_ids = []
    for i, doc in enumerate(documents):
        vector = embeddings.embed(doc)
        doc_id = client.insert(vector, metadata=doc)
        doc_ids.append(doc_id)
        print(f"Inserted: {doc[:50]}...")

    # Search for similar documents
    query = "AI and machine learning"
    query_vector = embeddings.embed(query)

    print(f"\nSearching for: '{query}'")
    results = client.search(query_vector, k=3)

    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f} - {result.metadata}")
```

## Error Handling

The client provides specific exception classes for different error conditions:

```python
from vectorpp import (
    VectorPPClient,
    VectorPPError,
    ConnectionError,
    DimensionMismatchError,
    VectorNotFoundError,
    CapacityExceededError,
)

client = VectorPPClient()

try:
    client.insert([0.1, 0.2])  # Wrong dimensions
except DimensionMismatchError as e:
    print(f"Dimension error: {e}")

try:
    client.delete("nonexistent-id")
except VectorNotFoundError as e:
    print(f"Not found: {e}")

try:
    client.insert([0.1] * 384)  # When database is full
except CapacityExceededError as e:
    print(f"Capacity exceeded: {e}")

try:
    client.search([0.1] * 384)  # When server is down
except ConnectionError as e:
    print(f"Connection error: {e}")
```

### Exception Hierarchy

```
VectorPPError (base)
├── ConnectionError
├── DimensionMismatchError
├── VectorNotFoundError
└── CapacityExceededError

EmbeddingError (base)
├── OpenAIError
├── HuggingFaceError
└── MissingAPIKeyError
```

## Server Configuration

The client connects to a Vector++ server. Make sure the server is running and configured with:

- Matching vector dimensions (e.g., 384 for MiniLM, 1536 for OpenAI)
- Sufficient capacity for your dataset

Example server configuration:
```json
{
    "server": {
        "port": 50051,
        "thread_count": 4
    },
    "database": {
        "dimensions": 384,
        "max_vectors": 100000
    }
}
```

## Requirements

- Python 3.8+
- grpcio
- protobuf

Optional:
- openai (for OpenAI embeddings)
- sentence-transformers (for HuggingFace embeddings)
