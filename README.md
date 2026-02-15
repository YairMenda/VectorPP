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

## Project layout

- `src/core/` ? vector store, HNSW, metadata, UUIDs
- `src/concurrency/` ? thread pool
- `src/config/` ? JSON config loader
- `src/server/` ? gRPC service and server
- `proto/vectordb.proto` ? service definition (Insert, Search, Delete)
- `tests/unit/`, `tests/benchmark/`, `tests/integration/`

Full requirements, user stories, and design are in [PRD.md](PRD.md).
