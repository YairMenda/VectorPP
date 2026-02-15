#include <gtest/gtest.h>
#include "core/vector_store.hpp"
#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <sstream>
#include <thread>
#include <set>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

using namespace vectorpp;

// Hardware spec collection structures and functions
struct HardwareSpecs {
    std::string os_name;
    std::string os_version;
    std::string cpu_name;
    int cpu_cores_logical;
    int cpu_cores_physical;
    uint64_t total_memory_mb;
    std::string architecture;
    std::string compiler;
    std::string compiler_version;
    std::string build_type;
};

// Get CPU name/brand string
std::string getCpuBrandString() {
#ifdef _WIN32
    int cpu_info[4] = {0};
    char brand[49] = {0};

    __cpuid(cpu_info, 0x80000000);
    int max_id = cpu_info[0];

    if (max_id >= 0x80000004) {
        __cpuid(cpu_info, 0x80000002);
        memcpy(brand, cpu_info, sizeof(cpu_info));
        __cpuid(cpu_info, 0x80000003);
        memcpy(brand + 16, cpu_info, sizeof(cpu_info));
        __cpuid(cpu_info, 0x80000004);
        memcpy(brand + 32, cpu_info, sizeof(cpu_info));

        // Trim leading spaces
        std::string result(brand);
        size_t start = result.find_first_not_of(' ');
        if (start != std::string::npos) {
            result = result.substr(start);
        }
        return result;
    }
    return "Unknown CPU";
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos && colon + 2 < line.size()) {
                return line.substr(colon + 2);
            }
        }
    }
    return "Unknown CPU";
#else
    return "Unknown CPU";
#endif
}

// Get OS information
std::pair<std::string, std::string> getOsInfo() {
#ifdef _WIN32
    OSVERSIONINFOEX osvi;
    ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
    osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

    // Note: GetVersionEx is deprecated but works for our purposes
    #pragma warning(disable: 4996)
    if (GetVersionEx((OSVERSIONINFO*)&osvi)) {
        std::ostringstream version;
        version << osvi.dwMajorVersion << "." << osvi.dwMinorVersion
                << " (Build " << osvi.dwBuildNumber << ")";
        return {"Windows", version.str()};
    }
    #pragma warning(default: 4996)
    return {"Windows", "Unknown version"};
#elif defined(__linux__)
    struct utsname buf;
    if (uname(&buf) == 0) {
        return {buf.sysname, std::string(buf.release) + " " + buf.version};
    }
    return {"Linux", "Unknown version"};
#elif defined(__APPLE__)
    return {"macOS", "Unknown version"};
#else
    return {"Unknown OS", "Unknown version"};
#endif
}

// Get physical CPU core count
int getPhysicalCores() {
#ifdef _WIN32
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);

    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<char> buffer(length);
        auto info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
            int count = 0;
            char* ptr = buffer.data();
            while (ptr < buffer.data() + length) {
                auto current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
                if (current->Relationship == RelationProcessorCore) {
                    count++;
                }
                ptr += current->Size;
            }
            return count;
        }
    }
    // Fallback: assume hyperthreading with 2 threads per core
    return static_cast<int>(std::thread::hardware_concurrency()) / 2;
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::set<std::pair<int, int>> cores; // physical_id, core_id pairs
    int physical_id = 0, core_id = 0;

    while (std::getline(cpuinfo, line)) {
        if (line.find("physical id") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                physical_id = std::stoi(line.substr(colon + 1));
            }
        } else if (line.find("core id") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                core_id = std::stoi(line.substr(colon + 1));
                cores.insert({physical_id, core_id});
            }
        }
    }

    if (!cores.empty()) {
        return static_cast<int>(cores.size());
    }
    // Fallback
    return static_cast<int>(std::thread::hardware_concurrency()) / 2;
#else
    return static_cast<int>(std::thread::hardware_concurrency()) / 2;
#endif
}

// Get total system memory in MB
uint64_t getTotalMemoryMB() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&memInfo)) {
        return memInfo.ullTotalPhys / (1024 * 1024);
    }
    return 0;
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            uint64_t kb;
            sscanf(line.c_str(), "MemTotal: %lu kB", &kb);
            return kb / 1024;
        }
    }
    return 0;
#else
    return 0;
#endif
}

// Collect all hardware specs
HardwareSpecs collectHardwareSpecs() {
    HardwareSpecs specs;

    auto [os_name, os_version] = getOsInfo();
    specs.os_name = os_name;
    specs.os_version = os_version;

    specs.cpu_name = getCpuBrandString();
    specs.cpu_cores_logical = static_cast<int>(std::thread::hardware_concurrency());
    specs.cpu_cores_physical = getPhysicalCores();
    specs.total_memory_mb = getTotalMemoryMB();

#ifdef _WIN64
    specs.architecture = "x86_64";
#elif defined(_WIN32)
    specs.architecture = "x86";
#elif defined(__x86_64__)
    specs.architecture = "x86_64";
#elif defined(__aarch64__)
    specs.architecture = "ARM64";
#else
    specs.architecture = "Unknown";
#endif

#ifdef _MSC_VER
    specs.compiler = "MSVC";
    specs.compiler_version = std::to_string(_MSC_VER);
#elif defined(__clang__)
    specs.compiler = "Clang";
    specs.compiler_version = std::to_string(__clang_major__) + "." +
                             std::to_string(__clang_minor__) + "." +
                             std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    specs.compiler = "GCC";
    specs.compiler_version = std::to_string(__GNUC__) + "." +
                             std::to_string(__GNUC_MINOR__) + "." +
                             std::to_string(__GNUC_PATCHLEVEL__);
#else
    specs.compiler = "Unknown";
    specs.compiler_version = "Unknown";
#endif

#ifdef NDEBUG
    specs.build_type = "Release";
#else
    specs.build_type = "Debug";
#endif

    return specs;
}

// Convert hardware specs to JSON
nlohmann::json hardwareSpecsToJson(const HardwareSpecs& specs) {
    nlohmann::json j;
    j["os"]["name"] = specs.os_name;
    j["os"]["version"] = specs.os_version;
    j["cpu"]["name"] = specs.cpu_name;
    j["cpu"]["cores_logical"] = specs.cpu_cores_logical;
    j["cpu"]["cores_physical"] = specs.cpu_cores_physical;
    j["memory"]["total_mb"] = specs.total_memory_mb;
    j["architecture"] = specs.architecture;
    j["compiler"]["name"] = specs.compiler;
    j["compiler"]["version"] = specs.compiler_version;
    j["build_type"] = specs.build_type;
    return j;
}

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

// Benchmark: Thread count scaling (1, 2, 4, 8 threads)
TEST_F(SearchBenchmarkTest, ThreadCountScaling) {
    std::cout << "\n=== Thread Count Scaling Benchmark ===" << std::endl;

    const size_t TEST_VECTORS = 5000;       // Fixed dataset size
    const size_t NUM_SEARCH_QUERIES = 500;  // Queries to run
    std::vector<size_t> thread_counts = {1, 2, 4, 8};

    std::cout << std::setw(10) << "Threads"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Throughput (q/s)"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    double baseline_time_ms = 0;

    for (size_t num_threads : thread_counts) {
        // Create a fresh store with the specified thread count
        VectorStoreConfig config;
        config.dimensions = DIMENSIONS;
        config.max_vectors = TEST_VECTORS + 1000;
        config.thread_pool_size = num_threads;
        auto test_store = std::make_unique<VectorStore>(config);

        // Insert vectors (same for all thread counts)
        std::mt19937 local_rng(SEED);
        for (size_t i = 0; i < TEST_VECTORS; ++i) {
            auto vec = generateRandomVector(DIMENSIONS, local_rng);
            test_store->insert(vec, "category_" + std::to_string(i % 10));
        }

        // Pre-generate query vectors
        std::vector<std::vector<float>> search_queries;
        search_queries.reserve(NUM_SEARCH_QUERIES);
        for (size_t i = 0; i < NUM_SEARCH_QUERIES; ++i) {
            search_queries.push_back(generateRandomVector(DIMENSIONS, local_rng));
        }

        // Warm-up run (10 queries)
        for (size_t i = 0; i < 10; ++i) {
            test_store->search(search_queries[i], TOP_K);
        }

        // Timed search using batch (parallel) search
        auto start = std::chrono::high_resolution_clock::now();
        auto results = test_store->searchBatch(search_queries, TOP_K);
        auto end = std::chrono::high_resolution_clock::now();

        // Verify results
        ASSERT_EQ(results.size(), NUM_SEARCH_QUERIES);

        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double throughput = duration_ms > 0 ? (NUM_SEARCH_QUERIES * 1000.0) / duration_ms : 0;

        // Calculate speedup from baseline (1 thread)
        double speedup = 1.0;
        if (num_threads == 1) {
            baseline_time_ms = static_cast<double>(duration_ms);
        } else if (baseline_time_ms > 0 && duration_ms > 0) {
            speedup = baseline_time_ms / duration_ms;
        }

        std::cout << std::setw(10) << num_threads
                  << std::setw(12) << duration_ms << " ms"
                  << std::setw(17) << std::fixed << std::setprecision(0) << throughput << " q/s"
                  << std::setw(12) << std::setprecision(2) << speedup << "x" << std::endl;

        // Record properties for each thread count
        std::string prop_name = "Thread" + std::to_string(num_threads) + "_QPS";
        RecordProperty(prop_name, static_cast<int>(throughput));

        if (num_threads == 8) {
            RecordProperty("Thread8_Speedup", static_cast<int>(speedup * 100));
        }
    }

    std::cout << "\nNotes:" << std::endl;
    std::cout << "  - Dataset: " << TEST_VECTORS << " vectors, " << DIMENSIONS << " dimensions" << std::endl;
    std::cout << "  - Queries: " << NUM_SEARCH_QUERIES << ", top-k: " << TOP_K << std::endl;
    std::cout << "  - Speedup is relative to single-threaded performance" << std::endl;
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

// Helper to get current timestamp as ISO 8601 string
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
#else
    localtime_r(&time_t_now, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

// Benchmark: Export results to JSON and CSV
TEST_F(SearchBenchmarkTest, ExportResults) {
    std::cout << "\n=== Benchmark Results Export ===" << std::endl;

    // Collect all benchmark results
    nlohmann::json results;
    results["metadata"]["timestamp"] = getCurrentTimestamp();
    results["metadata"]["dimensions"] = static_cast<int>(DIMENSIONS);

    // Add hardware specs to metadata
    HardwareSpecs specs = collectHardwareSpecs();
    results["metadata"]["hardware"] = hardwareSpecsToJson(specs);

    // Test parameters
    std::vector<size_t> scales = {1000, 5000, 10000};
    std::vector<size_t> thread_counts = {1, 2, 4, 8};
    const size_t NUM_SEARCH_QUERIES = 500;

    // === Insert Throughput Tests ===
    nlohmann::json insert_results = nlohmann::json::array();
    std::cout << "\nCollecting insert throughput data..." << std::endl;

    for (size_t num_vectors : scales) {
        VectorStoreConfig config;
        config.dimensions = DIMENSIONS;
        config.max_vectors = num_vectors + 1000;
        config.thread_pool_size = 4;
        auto test_store = std::make_unique<VectorStore>(config);

        // Pre-generate vectors
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

        nlohmann::json entry;
        entry["num_vectors"] = num_vectors;
        entry["time_ms"] = duration_ms;
        entry["throughput_vectors_per_sec"] = throughput;
        entry["target_vectors_per_sec"] = 50000;
        entry["target_met"] = throughput >= 50000;
        insert_results.push_back(entry);

        std::cout << "  " << num_vectors << " vectors: " << throughput << " v/s" << std::endl;
    }
    results["insert_throughput"] = insert_results;

    // === Search Throughput Tests ===
    nlohmann::json search_results = nlohmann::json::array();
    std::cout << "\nCollecting search throughput data..." << std::endl;

    for (size_t num_vectors : scales) {
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

        // Generate queries
        std::vector<std::vector<float>> search_queries;
        search_queries.reserve(NUM_SEARCH_QUERIES);
        for (size_t i = 0; i < NUM_SEARCH_QUERIES; ++i) {
            search_queries.push_back(generateRandomVector(DIMENSIONS, local_rng));
        }

        // Warm-up
        for (size_t i = 0; i < 10; ++i) {
            test_store->search(search_queries[i], TOP_K);
        }

        // Timed search
        auto start = std::chrono::high_resolution_clock::now();
        auto batch_results = test_store->searchBatch(search_queries, TOP_K);
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_EQ(batch_results.size(), NUM_SEARCH_QUERIES);

        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double throughput = duration_ms > 0 ? (NUM_SEARCH_QUERIES * 1000.0) / duration_ms : 0;

        nlohmann::json entry;
        entry["num_vectors"] = num_vectors;
        entry["num_queries"] = NUM_SEARCH_QUERIES;
        entry["top_k"] = TOP_K;
        entry["time_ms"] = duration_ms;
        entry["throughput_queries_per_sec"] = throughput;
        entry["target_queries_per_sec"] = 10000;
        entry["target_met"] = throughput >= 10000;
        search_results.push_back(entry);

        std::cout << "  " << num_vectors << " vectors: " << throughput << " q/s" << std::endl;
    }
    results["search_throughput"] = search_results;

    // === Thread Scaling Tests ===
    nlohmann::json thread_results = nlohmann::json::array();
    std::cout << "\nCollecting thread scaling data..." << std::endl;

    const size_t THREAD_TEST_VECTORS = 5000;
    double baseline_time = 0;

    for (size_t num_threads : thread_counts) {
        VectorStoreConfig config;
        config.dimensions = DIMENSIONS;
        config.max_vectors = THREAD_TEST_VECTORS + 1000;
        config.thread_pool_size = num_threads;
        auto test_store = std::make_unique<VectorStore>(config);

        // Insert vectors
        std::mt19937 local_rng(SEED);
        for (size_t i = 0; i < THREAD_TEST_VECTORS; ++i) {
            auto vec = generateRandomVector(DIMENSIONS, local_rng);
            test_store->insert(vec, "category_" + std::to_string(i % 10));
        }

        // Generate queries
        std::vector<std::vector<float>> search_queries;
        search_queries.reserve(NUM_SEARCH_QUERIES);
        for (size_t i = 0; i < NUM_SEARCH_QUERIES; ++i) {
            search_queries.push_back(generateRandomVector(DIMENSIONS, local_rng));
        }

        // Warm-up
        for (size_t i = 0; i < 10; ++i) {
            test_store->search(search_queries[i], TOP_K);
        }

        // Timed search
        auto start = std::chrono::high_resolution_clock::now();
        auto batch_results = test_store->searchBatch(search_queries, TOP_K);
        auto end = std::chrono::high_resolution_clock::now();

        ASSERT_EQ(batch_results.size(), NUM_SEARCH_QUERIES);

        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double throughput = duration_ms > 0 ? (NUM_SEARCH_QUERIES * 1000.0) / duration_ms : 0;

        double speedup = 1.0;
        if (num_threads == 1) {
            baseline_time = static_cast<double>(duration_ms);
        } else if (baseline_time > 0 && duration_ms > 0) {
            speedup = baseline_time / duration_ms;
        }

        nlohmann::json entry;
        entry["num_threads"] = num_threads;
        entry["num_vectors"] = THREAD_TEST_VECTORS;
        entry["num_queries"] = NUM_SEARCH_QUERIES;
        entry["time_ms"] = duration_ms;
        entry["throughput_queries_per_sec"] = throughput;
        entry["speedup"] = speedup;
        thread_results.push_back(entry);

        std::cout << "  " << num_threads << " threads: " << throughput << " q/s (speedup: " << speedup << "x)" << std::endl;
    }
    results["thread_scaling"] = thread_results;

    // === Write JSON file ===
    std::string json_filename = "benchmark_results.json";
    std::ofstream json_file(json_filename);
    if (json_file.is_open()) {
        json_file << results.dump(2);
        json_file.close();
        std::cout << "\nJSON results exported to: " << json_filename << std::endl;
    } else {
        std::cerr << "Failed to write JSON file: " << json_filename << std::endl;
    }

    // === Write CSV file ===
    std::string csv_filename = "benchmark_results.csv";
    std::ofstream csv_file(csv_filename);
    if (csv_file.is_open()) {
        // Header
        csv_file << "test_type,num_vectors,num_threads,num_queries,top_k,time_ms,throughput,speedup,target,target_met\n";

        // Insert throughput
        for (const auto& entry : insert_results) {
            csv_file << "insert,"
                     << entry["num_vectors"].get<int>() << ","
                     << "4,"  // default threads
                     << ","   // no queries
                     << ","   // no top_k
                     << entry["time_ms"].get<int>() << ","
                     << std::fixed << std::setprecision(2) << entry["throughput_vectors_per_sec"].get<double>() << ","
                     << ","   // no speedup
                     << entry["target_vectors_per_sec"].get<int>() << ","
                     << (entry["target_met"].get<bool>() ? "true" : "false") << "\n";
        }

        // Search throughput
        for (const auto& entry : search_results) {
            csv_file << "search,"
                     << entry["num_vectors"].get<int>() << ","
                     << "4,"  // default threads
                     << entry["num_queries"].get<int>() << ","
                     << entry["top_k"].get<int>() << ","
                     << entry["time_ms"].get<int>() << ","
                     << std::fixed << std::setprecision(2) << entry["throughput_queries_per_sec"].get<double>() << ","
                     << ","   // no speedup
                     << entry["target_queries_per_sec"].get<int>() << ","
                     << (entry["target_met"].get<bool>() ? "true" : "false") << "\n";
        }

        // Thread scaling
        for (const auto& entry : thread_results) {
            csv_file << "thread_scaling,"
                     << entry["num_vectors"].get<int>() << ","
                     << entry["num_threads"].get<int>() << ","
                     << entry["num_queries"].get<int>() << ","
                     << TOP_K << ","
                     << entry["time_ms"].get<int>() << ","
                     << std::fixed << std::setprecision(2) << entry["throughput_queries_per_sec"].get<double>() << ","
                     << std::setprecision(2) << entry["speedup"].get<double>() << ","
                     << ","   // no target
                     << "\n"; // no target_met
        }

        csv_file.close();
        std::cout << "CSV results exported to: " << csv_filename << std::endl;
    } else {
        std::cerr << "Failed to write CSV file: " << csv_filename << std::endl;
    }

    // Record properties for test reporting
    RecordProperty("JsonExported", 1);
    RecordProperty("CsvExported", 1);
}

// Benchmark: Vector dimension variations (384, 768, 1536)
TEST_F(SearchBenchmarkTest, DimensionVariations) {
    std::cout << "\n=== Dimension Variations Benchmark ===" << std::endl;

    // Test different dimensions: 384 (MiniLM), 768 (BERT), 1536 (OpenAI)
    std::vector<size_t> dimensions = {384, 768, 1536};
    const size_t TEST_VECTORS = 5000;
    const size_t NUM_SEARCH_QUERIES = 500;

    std::cout << "\nInsert Throughput by Dimension:" << std::endl;
    std::cout << std::setw(12) << "Dimensions"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Throughput (v/s)" << std::endl;
    std::cout << std::string(47, '-') << std::endl;

    // Store results for later comparison
    std::vector<std::pair<size_t, double>> insert_results;
    std::vector<std::pair<size_t, double>> search_results;

    for (size_t dims : dimensions) {
        // Create a fresh store with the specified dimensions
        VectorStoreConfig config;
        config.dimensions = dims;
        config.max_vectors = TEST_VECTORS + 1000;
        config.thread_pool_size = 4;
        auto test_store = std::make_unique<VectorStore>(config);

        // Pre-generate vectors with this dimension
        std::vector<std::vector<float>> vectors;
        vectors.reserve(TEST_VECTORS);
        std::mt19937 local_rng(SEED);
        for (size_t i = 0; i < TEST_VECTORS; ++i) {
            vectors.push_back(generateRandomVector(dims, local_rng));
        }

        // Timed insert
        auto insert_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < TEST_VECTORS; ++i) {
            test_store->insert(vectors[i], "category_" + std::to_string(i % 10));
        }
        auto insert_end = std::chrono::high_resolution_clock::now();

        auto insert_ms = std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start).count();
        double insert_throughput = (TEST_VECTORS * 1000.0) / insert_ms;
        insert_results.push_back({dims, insert_throughput});

        std::cout << std::setw(12) << dims
                  << std::setw(12) << insert_ms << " ms"
                  << std::setw(17) << std::fixed << std::setprecision(0) << insert_throughput << " v/s" << std::endl;

        // Record property for this dimension
        std::string prop_name = "Insert_Dim" + std::to_string(dims) + "_Throughput";
        RecordProperty(prop_name, static_cast<int>(insert_throughput));
    }

    std::cout << "\nSearch Throughput by Dimension:" << std::endl;
    std::cout << std::setw(12) << "Dimensions"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Throughput (q/s)" << std::endl;
    std::cout << std::string(47, '-') << std::endl;

    for (size_t dims : dimensions) {
        // Create a fresh store with the specified dimensions
        VectorStoreConfig config;
        config.dimensions = dims;
        config.max_vectors = TEST_VECTORS + 1000;
        config.thread_pool_size = 4;
        auto test_store = std::make_unique<VectorStore>(config);

        // Insert vectors
        std::mt19937 local_rng(SEED);
        for (size_t i = 0; i < TEST_VECTORS; ++i) {
            auto vec = generateRandomVector(dims, local_rng);
            test_store->insert(vec, "category_" + std::to_string(i % 10));
        }

        // Generate queries with matching dimensions
        std::vector<std::vector<float>> search_queries;
        search_queries.reserve(NUM_SEARCH_QUERIES);
        for (size_t i = 0; i < NUM_SEARCH_QUERIES; ++i) {
            search_queries.push_back(generateRandomVector(dims, local_rng));
        }

        // Warm-up run
        for (size_t i = 0; i < 10; ++i) {
            test_store->search(search_queries[i], TOP_K);
        }

        // Timed search
        auto search_start = std::chrono::high_resolution_clock::now();
        auto batch_results = test_store->searchBatch(search_queries, TOP_K);
        auto search_end = std::chrono::high_resolution_clock::now();

        ASSERT_EQ(batch_results.size(), NUM_SEARCH_QUERIES);

        auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();
        double search_throughput = search_ms > 0 ? (NUM_SEARCH_QUERIES * 1000.0) / search_ms : 0;
        search_results.push_back({dims, search_throughput});

        std::cout << std::setw(12) << dims
                  << std::setw(12) << search_ms << " ms"
                  << std::setw(17) << std::fixed << std::setprecision(0) << search_throughput << " q/s" << std::endl;

        // Record property for this dimension
        std::string prop_name = "Search_Dim" + std::to_string(dims) + "_Throughput";
        RecordProperty(prop_name, static_cast<int>(search_throughput));
    }

    // Summary
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  - Higher dimensions require more computation per vector" << std::endl;
    std::cout << "  - 384 dims: MiniLM (sentence-transformers)" << std::endl;
    std::cout << "  - 768 dims: BERT base" << std::endl;
    std::cout << "  - 1536 dims: OpenAI ada-002" << std::endl;
    std::cout << "  - Dataset: " << TEST_VECTORS << " vectors, " << NUM_SEARCH_QUERIES << " queries" << std::endl;
}

// Test: Document hardware specifications
TEST_F(SearchBenchmarkTest, DocumentHardwareSpecs) {
    std::cout << "\n=== Hardware Specifications ===" << std::endl;

    HardwareSpecs specs = collectHardwareSpecs();

    // Display hardware specs in console
    std::cout << "\nSystem Information:" << std::endl;
    std::cout << "  OS:           " << specs.os_name << " " << specs.os_version << std::endl;
    std::cout << "  Architecture: " << specs.architecture << std::endl;
    std::cout << std::endl;

    std::cout << "CPU Information:" << std::endl;
    std::cout << "  Model:        " << specs.cpu_name << std::endl;
    std::cout << "  Physical cores: " << specs.cpu_cores_physical << std::endl;
    std::cout << "  Logical cores:  " << specs.cpu_cores_logical << std::endl;
    std::cout << std::endl;

    std::cout << "Memory Information:" << std::endl;
    std::cout << "  Total RAM:    " << specs.total_memory_mb << " MB ("
              << std::fixed << std::setprecision(1)
              << (specs.total_memory_mb / 1024.0) << " GB)" << std::endl;
    std::cout << std::endl;

    std::cout << "Build Information:" << std::endl;
    std::cout << "  Compiler:     " << specs.compiler << " " << specs.compiler_version << std::endl;
    std::cout << "  Build type:   " << specs.build_type << std::endl;
    std::cout << std::endl;

    // Export hardware specs to JSON file
    nlohmann::json hw_json = hardwareSpecsToJson(specs);
    hw_json["timestamp"] = getCurrentTimestamp();

    std::string hw_filename = "hardware_specs.json";
    std::ofstream hw_file(hw_filename);
    if (hw_file.is_open()) {
        hw_file << hw_json.dump(2);
        hw_file.close();
        std::cout << "Hardware specs exported to: " << hw_filename << std::endl;
    } else {
        std::cerr << "Failed to write hardware specs file: " << hw_filename << std::endl;
    }

    // Record properties for test reporting
    RecordProperty("CPU", specs.cpu_name);
    RecordProperty("CPUCoresPhysical", specs.cpu_cores_physical);
    RecordProperty("CPUCoresLogical", specs.cpu_cores_logical);
    RecordProperty("MemoryMB", static_cast<int>(specs.total_memory_mb));
    RecordProperty("OS", specs.os_name);
    RecordProperty("Compiler", specs.compiler);
    RecordProperty("BuildType", specs.build_type);

    // Verify specs were collected (basic sanity checks)
    ASSERT_GT(specs.cpu_cores_logical, 0);
    ASSERT_GT(specs.total_memory_mb, 0);
    ASSERT_FALSE(specs.os_name.empty());
    ASSERT_FALSE(specs.cpu_name.empty());
}
