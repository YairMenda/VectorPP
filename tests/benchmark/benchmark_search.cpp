#include <gtest/gtest.h>
#include "core/vector_store.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

using namespace vectorpp;

// Helper to generate random vectors
std::vector<float> generateRandomVector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

// Test fixture for search benchmarks
class SearchBenchmarkTest : public ::testing::Test {
protected:
    static constexpr size_t DIMENSIONS = 384;  // MiniLM default
    static constexpr size_t NUM_VECTORS = 5000;
    static constexpr size_t NUM_QUERIES = 100;
    static constexpr size_t TOP_K = 10;
    static constexpr int SEED = 42;

    std::unique_ptr<VectorStore> store_;
    std::vector<std::vector<float>> queries_;
    std::mt19937 rng_;

    void SetUp() override {
        rng_.seed(SEED);

        // Create store with multi-threaded config
        VectorStoreConfig config;
        config.dimensions = DIMENSIONS;
        config.max_vectors = NUM_VECTORS + 1000;
        config.thread_pool_size = 4;  // Use 4 threads for parallel search
        store_ = std::make_unique<VectorStore>(config);

        // Insert vectors
        std::cout << "Inserting " << NUM_VECTORS << " vectors (" << DIMENSIONS << " dimensions)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            auto vec = generateRandomVector(DIMENSIONS, rng_);
            store_->insert(vec, "category_" + std::to_string(i % 10));
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto insert_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double insert_rate = (NUM_VECTORS * 1000.0) / insert_ms;
        std::cout << "Insert complete: " << insert_ms << " ms ("
                  << std::fixed << std::setprecision(0) << insert_rate << " vectors/sec)" << std::endl;

        // Generate query vectors
        queries_.reserve(NUM_QUERIES);
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            queries_.push_back(generateRandomVector(DIMENSIONS, rng_));
        }
    }
};

// Benchmark: Single-threaded search (sequential)
TEST_F(SearchBenchmarkTest, SingleThreadedSearch) {
    std::cout << "\n=== Single-Threaded Search Benchmark ===" << std::endl;
    std::cout << "Queries: " << NUM_QUERIES << ", top-k: " << TOP_K << std::endl;

    // Warm-up run
    for (size_t i = 0; i < 10; ++i) {
        store_->search(queries_[i], TOP_K);
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& query : queries_) {
        auto results = store_->search(query, TOP_K);
        // Verify we got results (prevents optimization)
        ASSERT_FALSE(results.empty());
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_latency_us = static_cast<double>(duration_us) / NUM_QUERIES;
    double queries_per_sec = (NUM_QUERIES * 1000000.0) / duration_us;

    std::cout << "Total time: " << duration_us / 1000.0 << " ms" << std::endl;
    std::cout << "Avg latency: " << avg_latency_us << " us/query" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(0) << queries_per_sec << " queries/sec" << std::endl;

    // Store metrics for comparison
    RecordProperty("SingleThreadedTotalUs", static_cast<int>(duration_us));
    RecordProperty("SingleThreadedQPS", static_cast<int>(queries_per_sec));
}

// Benchmark: Multi-threaded batch search (parallel)
TEST_F(SearchBenchmarkTest, MultiThreadedBatchSearch) {
    std::cout << "\n=== Multi-Threaded Batch Search Benchmark ===" << std::endl;
    std::cout << "Queries: " << NUM_QUERIES << ", top-k: " << TOP_K << ", threads: 4" << std::endl;

    // Warm-up run
    std::vector<std::vector<float>> warmup_queries(queries_.begin(), queries_.begin() + 10);
    store_->searchBatch(warmup_queries, TOP_K);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();

    auto batch_results = store_->searchBatch(queries_, TOP_K);

    auto end = std::chrono::high_resolution_clock::now();

    // Verify we got results
    ASSERT_EQ(batch_results.size(), NUM_QUERIES);
    for (const auto& results : batch_results) {
        ASSERT_FALSE(results.empty());
    }

    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_latency_us = static_cast<double>(duration_us) / NUM_QUERIES;
    double queries_per_sec = (NUM_QUERIES * 1000000.0) / duration_us;

    std::cout << "Total time: " << duration_us / 1000.0 << " ms" << std::endl;
    std::cout << "Avg latency: " << avg_latency_us << " us/query" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(0) << queries_per_sec << " queries/sec" << std::endl;

    RecordProperty("MultiThreadedTotalUs", static_cast<int>(duration_us));
    RecordProperty("MultiThreadedQPS", static_cast<int>(queries_per_sec));
}

// Benchmark: Compare different thread counts
TEST_F(SearchBenchmarkTest, ThreadCountComparison) {
    std::cout << "\n=== Thread Count Comparison ===" << std::endl;
    std::cout << "Comparing single-threaded vs multi-threaded (4 threads)" << std::endl;

    // Single-threaded: sequential search calls
    auto st_start = std::chrono::high_resolution_clock::now();
    for (const auto& query : queries_) {
        store_->search(query, TOP_K);
    }
    auto st_end = std::chrono::high_resolution_clock::now();
    auto st_us = std::chrono::duration_cast<std::chrono::microseconds>(st_end - st_start).count();
    double st_qps = (NUM_QUERIES * 1000000.0) / st_us;

    // Multi-threaded: batch search with thread pool
    auto mt_start = std::chrono::high_resolution_clock::now();
    store_->searchBatch(queries_, TOP_K);
    auto mt_end = std::chrono::high_resolution_clock::now();
    auto mt_us = std::chrono::duration_cast<std::chrono::microseconds>(mt_end - mt_start).count();
    double mt_qps = (NUM_QUERIES * 1000000.0) / mt_us;

    double speedup = static_cast<double>(st_us) / mt_us;
    double qps_improvement = (mt_qps / st_qps - 1.0) * 100.0;

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Single-threaded: " << st_us / 1000.0 << " ms ("
              << std::fixed << std::setprecision(0) << st_qps << " qps)" << std::endl;
    std::cout << "  Multi-threaded:  " << mt_us / 1000.0 << " ms ("
              << std::fixed << std::setprecision(0) << mt_qps << " qps)" << std::endl;
    std::cout << "  Speedup: " << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "  QPS improvement: " << std::setprecision(1) << qps_improvement << "%" << std::endl;

    RecordProperty("Speedup", static_cast<int>(speedup * 100));
    RecordProperty("QPSImprovement", static_cast<int>(qps_improvement));
}

// Benchmark: Scale test with varying query batch sizes
TEST_F(SearchBenchmarkTest, BatchSizeScaling) {
    std::cout << "\n=== Batch Size Scaling Benchmark ===" << std::endl;

    std::vector<size_t> batch_sizes = {10, 25, 50, 100};

    std::cout << std::setw(10) << "Batch"
              << std::setw(15) << "Sequential"
              << std::setw(15) << "Parallel"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (size_t batch_size : batch_sizes) {
        std::vector<std::vector<float>> batch_queries(queries_.begin(),
                                                       queries_.begin() + batch_size);

        // Sequential
        auto seq_start = std::chrono::high_resolution_clock::now();
        for (const auto& query : batch_queries) {
            store_->search(query, TOP_K);
        }
        auto seq_end = std::chrono::high_resolution_clock::now();
        auto seq_us = std::chrono::duration_cast<std::chrono::microseconds>(seq_end - seq_start).count();

        // Parallel
        auto par_start = std::chrono::high_resolution_clock::now();
        store_->searchBatch(batch_queries, TOP_K);
        auto par_end = std::chrono::high_resolution_clock::now();
        auto par_us = std::chrono::duration_cast<std::chrono::microseconds>(par_end - par_start).count();

        double speedup = static_cast<double>(seq_us) / par_us;

        std::cout << std::setw(10) << batch_size
                  << std::setw(12) << seq_us / 1000.0 << " ms"
                  << std::setw(12) << par_us / 1000.0 << " ms"
                  << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
}

// Benchmark: Large scale test (10K vectors target)
TEST_F(SearchBenchmarkTest, LargeScalePerformance) {
    std::cout << "\n=== Large Scale Performance Test ===" << std::endl;

    // Create a new store with 10K vectors
    VectorStoreConfig config;
    config.dimensions = DIMENSIONS;
    config.max_vectors = 15000;
    config.thread_pool_size = 4;
    auto large_store = std::make_unique<VectorStore>(config);

    const size_t LARGE_NUM_VECTORS = 10000;
    std::cout << "Building index with " << LARGE_NUM_VECTORS << " vectors..." << std::endl;

    auto build_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_NUM_VECTORS; ++i) {
        auto vec = generateRandomVector(DIMENSIONS, rng_);
        large_store->insert(vec, "cat_" + std::to_string(i % 100));
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "Index build time: " << build_ms << " ms" << std::endl;

    // Run search benchmark on 10K dataset
    const size_t SEARCH_QUERIES = 1000;
    std::vector<std::vector<float>> large_queries;
    large_queries.reserve(SEARCH_QUERIES);
    for (size_t i = 0; i < SEARCH_QUERIES; ++i) {
        large_queries.push_back(generateRandomVector(DIMENSIONS, rng_));
    }

    std::cout << "Running " << SEARCH_QUERIES << " search queries..." << std::endl;

    // Multi-threaded batch search
    auto search_start = std::chrono::high_resolution_clock::now();
    auto results = large_store->searchBatch(large_queries, TOP_K);
    auto search_end = std::chrono::high_resolution_clock::now();

    ASSERT_EQ(results.size(), SEARCH_QUERIES);

    auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();
    double qps = (SEARCH_QUERIES * 1000.0) / search_ms;

    std::cout << "\n10K Vector Dataset Results:" << std::endl;
    std::cout << "  Total search time: " << search_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0) << qps << " queries/sec" << std::endl;
    std::cout << "  Target: 10,000 queries/sec" << std::endl;

    if (qps >= 10000) {
        std::cout << "  STATUS: TARGET MET!" << std::endl;
    } else {
        std::cout << "  STATUS: " << std::setprecision(1) << (qps / 10000.0 * 100.0) << "% of target" << std::endl;
    }

    RecordProperty("LargeScaleQPS", static_cast<int>(qps));
}

// Benchmark: Search throughput (queries/second) at various scales
TEST_F(SearchBenchmarkTest, SearchThroughput) {
    std::cout << "\n=== Search Throughput Benchmark ===" << std::endl;

    // Test different scales: 1K, 5K, 10K vectors
    std::vector<size_t> scales = {1000, 5000, 10000};
    const size_t NUM_SEARCH_QUERIES = 500;  // Queries to run per scale

    std::cout << std::setw(12) << "Vectors"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Throughput (q/s)" << std::endl;
    std::cout << std::string(47, '-') << std::endl;

    for (size_t num_vectors : scales) {
        // Create a fresh store for each test
        VectorStoreConfig config;
        config.dimensions = DIMENSIONS;
        config.max_vectors = num_vectors + 1000;
        config.thread_pool_size = 4;
        auto test_store = std::make_unique<VectorStore>(config);

        // Insert vectors
        std::mt19937 local_rng(SEED);
        for (size_t i = 0; i < num_vectors; ++i) {
            auto vec = generateRandomVector(DIMENSIONS, local_rng);
            test_store->insert(vec, "category_" + std::to_string(i % 10));
        }

        // Pre-generate query vectors to exclude generation time from measurement
        std::vector<std::vector<float>> search_queries;
        search_queries.reserve(NUM_SEARCH_QUERIES);
        for (size_t i = 0; i < NUM_SEARCH_QUERIES; ++i) {
            search_queries.push_back(generateRandomVector(DIMENSIONS, local_rng));
        }

        // Warm-up run (10 queries)
        for (size_t i = 0; i < 10; ++i) {
            test_store->search(search_queries[i], TOP_K);
        }

        // Timed search - use batch search for parallel processing
        auto start = std::chrono::high_resolution_clock::now();

        auto results = test_store->searchBatch(search_queries, TOP_K);

        auto end = std::chrono::high_resolution_clock::now();

        // Verify results
        ASSERT_EQ(results.size(), NUM_SEARCH_QUERIES);

        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // Avoid division by zero
        double throughput = duration_ms > 0 ? (NUM_SEARCH_QUERIES * 1000.0) / duration_ms : 0;

        std::cout << std::setw(12) << num_vectors
                  << std::setw(12) << duration_ms << " ms"
                  << std::setw(17) << std::fixed << std::setprecision(0) << throughput << " q/s" << std::endl;

        // Record property for 10K scale
        if (num_vectors == 10000) {
            RecordProperty("Search10K_Throughput", static_cast<int>(throughput));

            // Check against target: 10,000 queries/sec
            std::cout << "\n10K Search Results:" << std::endl;
            std::cout << "  Throughput: " << throughput << " queries/sec" << std::endl;
            std::cout << "  Target: 10,000 queries/sec" << std::endl;

            if (throughput >= 10000) {
                std::cout << "  STATUS: TARGET MET!" << std::endl;
            } else {
                std::cout << "  STATUS: " << std::setprecision(1) << (throughput / 10000.0 * 100.0) << "% of target" << std::endl;
            }
        }
    }
}

// Benchmark: Insert throughput (vectors/second)
TEST_F(SearchBenchmarkTest, InsertThroughput) {
    std::cout << "\n=== Insert Throughput Benchmark ===" << std::endl;

    // Test different scales: 1K, 5K, 10K vectors
    std::vector<size_t> scales = {1000, 5000, 10000};

    std::cout << std::setw(12) << "Vectors"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Throughput (v/s)" << std::endl;
    std::cout << std::string(47, '-') << std::endl;

    for (size_t num_vectors : scales) {
        // Create a fresh store for each test
        VectorStoreConfig config;
        config.dimensions = DIMENSIONS;
        config.max_vectors = num_vectors + 1000;
        config.thread_pool_size = 4;
        auto test_store = std::make_unique<VectorStore>(config);

        // Pre-generate all vectors to exclude generation time from measurement
        std::vector<std::vector<float>> vectors;
        vectors.reserve(num_vectors);
        std::mt19937 local_rng(SEED);
        for (size_t i = 0; i < num_vectors; ++i) {
            vectors.push_back(generateRandomVector(DIMENSIONS, local_rng));
        }

        // Timed insert
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < num_vectors; ++i) {
            test_store->insert(vectors[i], "category_" + std::to_string(i % 10));
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double throughput = (num_vectors * 1000.0) / duration_ms;

        std::cout << std::setw(12) << num_vectors
                  << std::setw(12) << duration_ms << " ms"
                  << std::setw(17) << std::fixed << std::setprecision(0) << throughput << " v/s" << std::endl;

        // Record property for 10K scale
        if (num_vectors == 10000) {
            RecordProperty("Insert10K_Throughput", static_cast<int>(throughput));

            // Check against target: 50,000 vectors/sec
            std::cout << "\n10K Insert Results:" << std::endl;
            std::cout << "  Throughput: " << throughput << " vectors/sec" << std::endl;
            std::cout << "  Target: 50,000 vectors/sec" << std::endl;

            if (throughput >= 50000) {
                std::cout << "  STATUS: TARGET MET!" << std::endl;
            } else {
                std::cout << "  STATUS: " << std::setprecision(1) << (throughput / 50000.0 * 100.0) << "% of target" << std::endl;
            }
        }
    }
}
