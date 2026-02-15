#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <hnswlib/hnswlib.h>
#include "concurrency/thread_pool.hpp"

namespace vectorpp {

// Exception for dimension mismatch
class DimensionMismatchError : public std::runtime_error {
public:
    DimensionMismatchError(size_t expected, size_t actual)
        : std::runtime_error("Dimension mismatch: expected " + std::to_string(expected) +
                            ", got " + std::to_string(actual)),
          expected_(expected), actual_(actual) {}

    size_t expected() const { return expected_; }
    size_t actual() const { return actual_; }

private:
    size_t expected_;
    size_t actual_;
};

// Exception for capacity limit reached
class CapacityLimitError : public std::runtime_error {
public:
    explicit CapacityLimitError(size_t limit)
        : std::runtime_error("Vector store capacity limit reached: " + std::to_string(limit)),
          limit_(limit) {}

    size_t limit() const { return limit_; }

private:
    size_t limit_;
};

// Configuration for VectorStore
struct VectorStoreConfig {
    size_t dimensions = 384;      // Default for MiniLM
    size_t max_vectors = 100000;  // Soft limit

    // HNSW parameters
    size_t hnsw_m = 16;
    size_t hnsw_ef_construction = 200;
    size_t hnsw_ef_search = 50;

    // Thread pool configuration for parallel search
    size_t thread_pool_size = 0;  // 0 = hardware_concurrency
};

// Search result structure (named VectorSearchResult to avoid conflict with protobuf)
struct VectorSearchResult {
    std::string id;
    float score;
    std::string metadata;
};

// Main vector store class
class VectorStore {
public:
    explicit VectorStore(const VectorStoreConfig& config);
    ~VectorStore() = default;

    // Non-copyable, movable
    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;
    VectorStore(VectorStore&&) = default;
    VectorStore& operator=(VectorStore&&) = default;

    // Core operations
    std::string insert(const std::vector<float>& vector, const std::string& metadata = "");
    std::vector<VectorSearchResult> search(const std::vector<float>& query, size_t k,
                                     const std::string& filter_metadata = "") const;
    bool remove(const std::string& id);

    // Parallel batch search - processes multiple queries using thread pool
    std::vector<std::vector<VectorSearchResult>> searchBatch(
        const std::vector<std::vector<float>>& queries,
        size_t k,
        const std::string& filter_metadata = "") const;

    // Accessors
    size_t dimensions() const { return config_.dimensions; }
    size_t size() const { return current_size_; }
    size_t max_vectors() const { return config_.max_vectors; }
    bool contains(const std::string& id) const;

private:
    // Internal search helper (no lock - caller must hold mutex_)
    std::vector<VectorSearchResult> searchInternal(const std::vector<float>& query, size_t k,
                                              const std::string& filter_metadata) const;

    VectorStoreConfig config_;

    // HNSW index
    std::unique_ptr<hnswlib::InnerProductSpace> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

    // Mappings
    std::unordered_map<std::string, size_t> uuid_to_label_;  // UUID -> internal label
    std::unordered_map<size_t, std::string> label_to_uuid_;  // internal label -> UUID
    std::unordered_map<std::string, std::string> metadata_;  // UUID -> metadata

    size_t next_label_ = 0;
    size_t current_size_ = 0;

    mutable std::mutex mutex_;

    // Thread pool for parallel search operations
    mutable std::unique_ptr<ThreadPool> thread_pool_;
};

} // namespace vectorpp
