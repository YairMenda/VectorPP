#include "core/vector_store.hpp"
#include "utils/uuid.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>

namespace vectorpp {

VectorStore::VectorStore(const VectorStoreConfig& config)
    : config_(config) {
    // Use InnerProductSpace for cosine similarity
    // We normalize vectors on insert, so inner product = cosine similarity
    space_ = std::make_unique<hnswlib::InnerProductSpace>(config_.dimensions);

    // Initialize HNSW index
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(),
        config_.max_vectors,
        config_.hnsw_m,
        config_.hnsw_ef_construction
    );

    index_->setEf(config_.hnsw_ef_search);

    // Initialize thread pool for parallel search
    size_t pool_size = config_.thread_pool_size;
    if (pool_size == 0) {
        pool_size = std::thread::hardware_concurrency();
        if (pool_size == 0) pool_size = 4;  // Fallback default
    }
    thread_pool_ = std::make_unique<ThreadPool>(pool_size);
}

std::string VectorStore::insert(const std::vector<float>& vector, const std::string& metadata) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check dimensions
    if (vector.size() != config_.dimensions) {
        throw DimensionMismatchError(config_.dimensions, vector.size());
    }

    // Check capacity
    if (current_size_ >= config_.max_vectors) {
        throw CapacityLimitError(config_.max_vectors);
    }

    // Normalize vector for cosine similarity
    std::vector<float> normalized = vector;
    float norm = 0.0f;
    for (float v : normalized) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& v : normalized) {
            v /= norm;
        }
    }

    // Generate UUID
    std::string uuid = generate_uuid();

    // Add to HNSW index
    size_t label = next_label_++;
    index_->addPoint(normalized.data(), label);

    // Store mappings
    uuid_to_label_[uuid] = label;
    label_to_uuid_[label] = uuid;
    metadata_[uuid] = metadata;
    current_size_++;

    return uuid;
}

std::vector<VectorSearchResult> VectorStore::search(const std::vector<float>& query, size_t k,
                                               const std::string& filter_metadata) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return searchInternal(query, k, filter_metadata);
}

std::vector<VectorSearchResult> VectorStore::searchInternal(const std::vector<float>& query, size_t k,
                                                       const std::string& filter_metadata) const {
    // Note: Caller must hold mutex_

    // Check dimensions
    if (query.size() != config_.dimensions) {
        throw DimensionMismatchError(config_.dimensions, query.size());
    }

    if (current_size_ == 0) {
        return {};
    }

    // Normalize query vector
    std::vector<float> normalized = query;
    float norm = 0.0f;
    for (float v : normalized) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& v : normalized) {
            v /= norm;
        }
    }

    // Adjust k to not exceed current size
    k = std::min(k, current_size_);

    // Search
    auto results = index_->searchKnn(normalized.data(), k);

    // Convert to VectorSearchResult
    std::vector<VectorSearchResult> output;
    output.reserve(results.size());

    while (!results.empty()) {
        auto [dist, label] = results.top();
        results.pop();

        auto label_it = label_to_uuid_.find(label);
        if (label_it == label_to_uuid_.end()) {
            continue;  // Deleted vector
        }

        const std::string& uuid = label_it->second;
        auto meta_it = metadata_.find(uuid);
        std::string meta = (meta_it != metadata_.end()) ? meta_it->second : "";

        // Apply filter if specified
        if (!filter_metadata.empty() && meta != filter_metadata) {
            continue;
        }

        // Inner product distance -> similarity (1 - dist for normalized vectors)
        // hnswlib inner product returns 1 - cos_sim for normalized vectors
        float similarity = 1.0f - dist;

        output.push_back({uuid, similarity, meta});
    }

    // Sort by score descending (results came in ascending order from priority queue)
    std::sort(output.begin(), output.end(),
              [](const VectorSearchResult& a, const VectorSearchResult& b) {
                  return a.score > b.score;
              });

    return output;
}

std::vector<std::vector<VectorSearchResult>> VectorStore::searchBatch(
    const std::vector<std::vector<float>>& queries,
    size_t k,
    const std::string& filter_metadata) const {

    if (queries.empty()) {
        return {};
    }

    // Submit all search tasks to the thread pool
    std::vector<std::future<std::vector<VectorSearchResult>>> futures;
    futures.reserve(queries.size());

    for (const auto& query : queries) {
        // Capture query by value for thread safety
        futures.push_back(thread_pool_->submit([this, query, k, filter_metadata]() {
            std::lock_guard<std::mutex> lock(mutex_);
            return searchInternal(query, k, filter_metadata);
        }));
    }

    // Collect results in order
    std::vector<std::vector<VectorSearchResult>> results;
    results.reserve(queries.size());

    for (auto& future : futures) {
        results.push_back(future.get());
    }

    return results;
}

bool VectorStore::remove(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = uuid_to_label_.find(id);
    if (it == uuid_to_label_.end()) {
        return false;
    }

    size_t label = it->second;

    // Mark as deleted in HNSW (hnswlib supports marking points as deleted)
    index_->markDelete(label);

    // Remove from mappings
    uuid_to_label_.erase(it);
    label_to_uuid_.erase(label);
    metadata_.erase(id);
    current_size_--;

    return true;
}

bool VectorStore::contains(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return uuid_to_label_.find(id) != uuid_to_label_.end();
}

} // namespace vectorpp
