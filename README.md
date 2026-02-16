# Vector++

A high-performance, in-memory vector database in C++17. Insert embedding vectors, search by cosine similarity (top-K), and delete by ID?exposed over gRPC. Think of it as a minimal Pinecone-style store built from scratch to see how fast similarity search works.

**Features:** HNSW index (via hnswlib), configurable dimensions and limits, JSON config, custom thread pool for parallel search, UUIDs for vectors, optional metadata. No persistence (in-memory only), no auth, single node.

**Stack:** C++17, CMake, gRPC/Protobuf (vcpkg), hnswlib, nlohmann/json, Google Test.

## Build

Requires CMake 3.16+ and vcpkg with `grpc` and `protobuf` (and their transitive deps). Other deps (googletest, nlohmann-json, hnswlib) are fetched by CMake.

```bash
# From repo root, with vcpkg in path or CMAKE_TOOLCHAIN_FILE set
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path/to]/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## Run

```bash
# Start server (uses config from repo or defaults)
./build/Release/vectorpp_server   # Windows
./build/vectorpp_server            # Linux
```

## Test

```bash
cd build && ctest -C Release
```

Unit tests: `test_vector_store`, `test_config_loader`, `test_thread_pool`. There is a search benchmark: `benchmark_search`.

## Demo

The movie similarity demo showcases Vector++ by finding semantically similar movies using embeddings.

### Setup

```bash
# Install Python dependencies
cd python
pip install grpcio protobuf sentence-transformers pandas
```

### Run the Demo

First, start the server in one terminal:

```bash
./build/Release/vectorpp_server   # Windows
./build/vectorpp_server            # Linux
```

Then run the demo in another terminal:

```bash
cd python

# Quick demo with sample movies (20 built-in movies)
python examples/movie_demo.py --generate-sample

# Single query mode
python examples/movie_demo.py --generate-sample --query "space adventure" --top-k 5

# Interactive mode (type queries, 'quit' to exit)
python examples/movie_demo.py --generate-sample

# With custom dataset (IMDB Top 1000 CSV from Kaggle)
python examples/movie_demo.py --dataset imdb_top_1000.csv
```

### Demo Options

| Option | Description |
|--------|-------------|
| `--generate-sample` | Use 20 built-in sample movies |
| `--dataset FILE` | Load movies from CSV file |
| `--query TEXT` | Single search query (non-interactive) |
| `--top-k N` | Number of results (default: 5) |
| `--model MODEL` | Embedding model (default: `all-MiniLM-L6-v2`) |
| `--host HOST` | Server host (default: `localhost`) |
| `--port PORT` | Server port (default: `50051`) |

### Interactive Commands

In interactive mode, type a movie description to search, or use:
- `top N` - Change number of results
- `genre GENRE` - Filter by genre
- `clear` - Clear genre filter
- `quit` / `exit` - Exit the demo

## Project layout

- `src/core/` ? vector store, HNSW, metadata, UUIDs
- `src/concurrency/` ? thread pool
- `src/config/` ? JSON config loader
- `src/server/` ? gRPC service and server
- `proto/vectordb.proto` ? service definition (Insert, Search, Delete)
- `tests/unit/`, `tests/benchmark/`, `tests/integration/`

Full requirements, user stories, and design are in [PRD.md](PRD.md).
