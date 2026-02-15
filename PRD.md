# PRD: Vector++ - High-Performance In-Memory Vector Database

## Introduction

Vector++ is a high-performance, in-memory vector database implemented in C++17. It provides fast similarity search capabilities for embedding vectors, exposed via a gRPC interface with a Python client for easy integration. The project demonstrates systems programming expertise combined with AI/ML domain knowledge.

**The Pitch:** "Everyone uses Pinecone. I decided to build a mini-Pinecone from scratch in C++ to understand how high-performance similarity search actually works."

**Target Resume Bullet:** Designed and built a multi-threaded Vector Database in C++ handling 10k queries/sec, featuring a custom thread-pool implementation for parallel cosine-similarity search exposed via gRPC.

---

## Goals

- Build a functional vector database supporting Insert, Search, and Delete operations
- Implement HNSW (Hierarchical Navigable Small World) indexing using hnswlib for fast similarity search
- Create a custom thread pool implementation for educational value and parallel query processing
- Expose functionality via gRPC with a clean Python client SDK
- Achieve 10k+ queries/second on a 10K vector dataset
- Support configurable vector dimensions per database instance
- Provide a compelling demo with movie/document similarity search

---

## User Stories

### US-001: Initialize Vector Database
**Description:** As a developer, I want to create a new vector database instance with configurable dimensions so that I can store embeddings of any size.

**Acceptance Criteria:**
- [x] Database accepts dimension parameter at creation time (e.g., 1536 for OpenAI, 384 for MiniLM)
- [x] Database rejects vectors that don't match configured dimensions
- [x] Database loads configuration from YAML/JSON config file
- [x] Configurable soft limit for maximum number of vectors
- [x] Unit tests pass for initialization scenarios

- **Priority:** 1
- **Status:** false
- **Notes:** Foundation for all other features

---

### US-002: Insert Vector with Auto-Generated ID
**Description:** As a developer, I want to insert embedding vectors and receive auto-generated UUIDs so that I can store and reference vectors uniquely.

**Acceptance Criteria:**
- [x] Insert accepts a vector (list of floats) and optional metadata (category string)
- [x] Server generates and returns a UUID for each inserted vector
- [x] Insert rejects vectors with wrong dimensions (returns gRPC error)
- [x] Insert respects soft memory limit (returns error when limit reached)
- [x] Inserted vectors are immediately searchable
- [x] Unit tests pass for insert operations

- **Priority:** 1
- **Status:** true
- **Notes:** Core CRUD operation

---

### US-003: Search for Similar Vectors
**Description:** As a developer, I want to search for the top-K most similar vectors using cosine similarity so that I can find relevant matches.

**Acceptance Criteria:**
- [x] Search accepts a query vector and K (number of results)
- [x] Returns top-K results sorted by cosine similarity (highest first)
- [x] Each result includes: vector ID, similarity score, metadata
- [x] Search uses HNSW index for fast approximate nearest neighbor search
- [x] Optional: filter results by metadata category
- [x] Search parallelized using custom thread pool (depends on US-005)
- [x] Unit tests and benchmarks pass

- **Priority:** 1
- **Status:** false
- **Notes:** Core feature - must be fast. Thread pool parallelization deferred to US-005.

---

### US-004: Delete Vector by ID
**Description:** As a developer, I want to delete vectors by their UUID so that I can remove outdated or incorrect data.

**Acceptance Criteria:**
- [x] Delete accepts a vector UUID
- [x] Returns success if vector existed and was deleted
- [x] Returns appropriate error if vector ID not found
- [x] Deleted vectors no longer appear in search results
- [x] Unit tests pass for delete operations

- **Priority:** 2
- **Status:** true
- **Notes:** Basic CRUD completion

---

### US-005: Custom Thread Pool Implementation
**Description:** As a developer building this project, I want a custom thread pool implementation to demonstrate concurrency knowledge and parallelize search operations.

**Acceptance Criteria:**
- [x] Thread pool uses std::thread + std::queue + condition variables
- [x] Configurable number of worker threads
- [x] Supports submitting tasks and getting futures back
- [x] Graceful shutdown (completes pending tasks)
- [x] Used by search operation to parallelize across index segments
- [x] Unit tests for thread pool correctness
- [x] Benchmarks comparing single-threaded vs multi-threaded search

- **Priority:** 1
- **Status:** false
- **Notes:** Educational value - key differentiator for resume

---

### US-006: gRPC Server Interface
**Description:** As a developer, I want a gRPC server exposing Insert, Search, and Delete operations so that clients can interact with the database over the network.

**Acceptance Criteria:**
- [x] Proto file defines VectorDB service with Insert, Search, Delete RPCs
- [x] All RPCs are unary (simple request/response)
- [x] Server listens on configurable port (from config file)
- [x] Proper gRPC error codes for all error conditions
- [x] Server logs requests to stdout/stderr
- [x] Server handles concurrent requests safely
- [x] Integration tests for gRPC endpoints

- **Priority:** 1
- **Status:** false
- **Notes:** Primary interface for the database

---

### US-007: Python Client SDK
**Description:** As a Python developer, I want a client library that wraps the gRPC interface so that I can easily interact with Vector++.

**Acceptance Criteria:**
- [x] Python package with VectorPPClient class
- [x] Methods: insert(vector, metadata) -> id, search(vector, k) -> results, delete(id) -> bool
- [x] Connection to server via host:port configuration
- [x] Clear error handling with Python exceptions
- [x] Helper functions to generate embeddings using OpenAI API
- [x] Helper functions to generate embeddings using HuggingFace (sentence-transformers)
- [ ] README with usage examples

- **Priority:** 2
- **Status:** false
- **Notes:** Makes the demo possible

---

### US-008: Movie Similarity Demo
**Description:** As a portfolio reviewer, I want to see a compelling demo that shows Vector++ finding similar movies so that I understand the practical application.

**Acceptance Criteria:**
- [ ] Script loads movie dataset (IMDB top 1000 or MovieLens)
- [ ] Generates embeddings for movie titles/descriptions using OpenAI or HuggingFace
- [ ] Inserts all movie embeddings into Vector++
- [ ] Interactive search: user enters a movie, sees top 5 similar movies
- [ ] Displays similarity scores and response time
- [ ] Demo runs end-to-end in under 1 minute (excluding embedding generation)

- **Priority:** 2
- **Status:** false
- **Notes:** The "sexy" demo for interviews

---

### US-009: Benchmark Suite
**Description:** As a developer, I want benchmarks measuring queries/second so that I can validate performance claims and optimize.

**Acceptance Criteria:**
- [ ] Benchmark measures insert throughput (vectors/second)
- [ ] Benchmark measures search throughput (queries/second)
- [ ] Tests at various scales: 1K, 5K, 10K vectors
- [ ] Tests with different thread counts: 1, 2, 4, 8 threads
- [ ] Results exported to JSON/CSV for visualization
- [ ] Includes vector dimension variations (384, 768, 1536)
- [ ] Documents hardware specs used for benchmarks

- **Priority:** 2
- **Status:** false
- **Notes:** Validates the "10k queries/sec" claim

---

### US-010: Configuration System
**Description:** As an operator, I want to configure the server via a config file so that I can customize behavior without recompiling.

**Acceptance Criteria:**
- [ ] YAML or JSON config file support
- [ ] Configurable: server port, thread count, max vectors, vector dimensions
- [ ] Configurable: HNSW parameters (M, ef_construction, ef_search)
- [ ] Default config file included in repository
- [ ] Server logs loaded configuration on startup
- [ ] Unit tests for config parsing

- **Priority:** 3
- **Status:** false
- **Notes:** Operational convenience

---

## Functional Requirements

### Core Database
- **FR-1:** The system must store vectors as contiguous arrays of 32-bit floats
- **FR-2:** The system must support configurable vector dimensions (set at database creation)
- **FR-3:** The system must generate RFC 4122 UUIDs for each inserted vector
- **FR-4:** The system must store optional metadata (category string) with each vector
- **FR-5:** The system must enforce a configurable soft limit on total vectors stored

### Search
- **FR-6:** The system must use cosine similarity as the distance metric
- **FR-7:** The system must use hnswlib for HNSW index implementation
- **FR-8:** The system must return top-K results sorted by descending similarity
- **FR-9:** The system must support filtering search results by metadata category
- **FR-10:** The system must parallelize search using the custom thread pool

### Concurrency
- **FR-11:** The thread pool must use std::thread, std::queue, and std::condition_variable
- **FR-12:** The thread pool must support configurable worker thread count
- **FR-13:** The thread pool must provide std::future for task results
- **FR-14:** The system must be thread-safe for concurrent Insert/Search/Delete operations

### Interface
- **FR-15:** The gRPC service must expose Insert, Search, and Delete as unary RPCs
- **FR-16:** The server must return appropriate gRPC status codes for all errors
- **FR-17:** The server must log all requests to stdout with timestamps

### Python Client
- **FR-18:** The Python client must wrap all gRPC operations
- **FR-19:** The Python client must provide embedding helpers for OpenAI and HuggingFace

---

## Non-Goals (Out of Scope)

- **No persistence:** Data is not persisted to disk (pure in-memory for MVP)
- **No replication:** Single-node only, no distributed architecture
- **No authentication:** No auth/security for MVP (assume trusted network)
- **No streaming RPCs:** Unary calls only, no server/client streaming
- **No GPU acceleration:** CPU-only implementation
- **No custom HNSW:** Use hnswlib, don't implement HNSW from scratch
- **No web UI:** CLI and Python client only
- **No automatic embedding:** Server stores vectors only, embedding done client-side

---

## Technical Considerations

### Architecture Overview
```
┌─────────────────────────────────────────────────────────────┐
│                      Python Client                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ VectorPP    │  │  Embedding   │  │  Demo Scripts     │  │
│  │ Client      │  │  Helpers     │  │  (movies, docs)   │  │
│  └──────┬──────┘  └──────────────┘  └───────────────────┘  │
└─────────┼───────────────────────────────────────────────────┘
          │ gRPC
┌─────────▼───────────────────────────────────────────────────┐
│                    C++ gRPC Server                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  VectorDB Service                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │
│  │  │  Insert  │  │  Search  │  │     Delete       │  │   │
│  │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │   │
│  └───────┼─────────────┼─────────────────┼────────────┘   │
│          │             │                 │                 │
│  ┌───────▼─────────────▼─────────────────▼────────────┐   │
│  │                  Vector Store                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │ HNSW Index  │  │  Metadata   │  │ ID Mapping │ │   │
│  │  │ (hnswlib)   │  │   Store     │  │  (UUID)    │ │   │
│  │  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └────────────────────────┬───────────────────────────┘   │
│                           │                                │
│  ┌────────────────────────▼───────────────────────────┐   │
│  │              Custom Thread Pool                     │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │   │
│  │  │Worker 1│ │Worker 2│ │Worker 3│ │Worker N│      │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Dependencies
| Dependency | Purpose | Installation |
|------------|---------|--------------|
| hnswlib | HNSW index implementation | vcpkg / header-only |
| gRPC | RPC framework | vcpkg |
| Protobuf | Serialization for gRPC | vcpkg |
| nlohmann/json | Config file parsing (JSON) | vcpkg / header-only |
| Google Test | Unit testing | vcpkg |
| Google Benchmark | Performance benchmarks | vcpkg |

### Build System
- CMake 3.16+ as build system
- vcpkg for dependency management
- Support Windows (primary) and Linux
- C++17 standard required

### Project Structure
```
VectorPP/
├── CMakeLists.txt
├── vcpkg.json                 # vcpkg manifest
├── config/
│   └── default.yaml           # Default configuration
├── proto/
│   └── vectordb.proto         # gRPC service definition
├── src/
│   ├── main.cpp               # Server entry point
│   ├── server/
│   │   ├── grpc_server.hpp/cpp
│   │   └── service_impl.hpp/cpp
│   ├── core/
│   │   ├── vector_store.hpp/cpp
│   │   ├── hnsw_index.hpp/cpp
│   │   └── metadata_store.hpp/cpp
│   ├── concurrency/
│   │   └── thread_pool.hpp/cpp
│   ├── config/
│   │   └── config_loader.hpp/cpp
│   └── utils/
│       ├── uuid.hpp/cpp
│       └── logging.hpp/cpp
├── tests/
│   ├── unit/
│   │   ├── test_thread_pool.cpp
│   │   ├── test_vector_store.cpp
│   │   └── test_config.cpp
│   └── benchmark/
│       └── benchmark_search.cpp
├── python/
│   ├── vectorpp/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── embeddings.py
│   ├── setup.py
│   └── examples/
│       ├── movie_demo.py
│       └── document_demo.py
└── README.md
```

### Performance Targets
| Metric | Target | Scale |
|--------|--------|-------|
| Insert throughput | 50,000 vectors/sec | 10K vectors, 1536 dims |
| Search throughput | 10,000 queries/sec | 10K vectors, top-10 |
| Search latency (p99) | < 10ms | 10K vectors, top-10 |
| Memory usage | < 500MB | 10K vectors, 1536 dims |

---

## Success Metrics

- **Functionality:** All CRUD operations work correctly via gRPC
- **Performance:** Achieve 10k queries/second on 10K vector dataset
- **Demo Impact:** Movie similarity demo runs smoothly, finds relevant results
- **Code Quality:** Clean architecture, comprehensive tests, documented code
- **Portability:** Builds and runs on both Windows and Linux
- **Resume Value:** Project clearly demonstrates C++ systems skills + AI domain knowledge

---

## Resolved Decisions

1. **HNSW Parameters:** M=16, ef_construction=200, ef_search=50 (good balance of speed/accuracy)
2. **UUID Library:** Standard approach - use `std::random_device` + `<random>` for UUID v4 generation (cross-platform, no extra dependency)
3. **Config Format:** JSON using nlohmann/json (simpler, header-only)
4. **Embedding Model for Demo:** HuggingFace sentence-transformers `all-MiniLM-L6-v2` (free, 384 dimensions, fast)
5. **Movie Dataset:** IMDB Top 1000 (available on Kaggle, simple CSV format)

## Open Questions

*None - all major decisions resolved.*

---

## Implementation Phases

### Phase 1: Foundation
- Project setup (CMake, vcpkg, directory structure)
- Custom thread pool implementation with tests
- Configuration system (YAML/JSON parsing)
- Basic logging utilities

### Phase 2: Core Database
- Vector store with HNSW index (hnswlib integration)
- Metadata storage
- UUID generation
- Insert, Search, Delete operations
- Unit tests for all operations

### Phase 3: gRPC Interface
- Proto file definition
- gRPC server implementation
- Service implementation connecting to vector store
- Integration tests

### Phase 4: Python Client
- gRPC client wrapper
- Embedding helpers (OpenAI, HuggingFace)
- Package setup (setup.py/pyproject.toml)
- Usage examples

### Phase 5: Demo & Benchmarks
- Movie dataset loading and embedding
- Interactive demo script
- Benchmark suite
- Results export to JSON/CSV
- Documentation and README

---

## Appendix: Proto File Draft

```protobuf
syntax = "proto3";

package vectorpp;

service VectorDB {
  rpc Insert(InsertRequest) returns (InsertResponse);
  rpc Search(SearchRequest) returns (SearchResponse);
  rpc Delete(DeleteRequest) returns (DeleteResponse);
}

message InsertRequest {
  repeated float vector = 1;
  string metadata = 2;  // Optional category string
}

message InsertResponse {
  string id = 1;  // UUID
}

message SearchRequest {
  repeated float query_vector = 1;
  int32 top_k = 2;
  string filter_metadata = 3;  // Optional category filter
}

message SearchResult {
  string id = 1;
  float score = 2;
  string metadata = 3;
}

message SearchResponse {
  repeated SearchResult results = 1;
}

message DeleteRequest {
  string id = 1;
}

message DeleteResponse {
  bool success = 1;
}
```
