#include <gtest/gtest.h>
#include "config/config_loader.hpp"
#include <fstream>
#include <cstdio>

using namespace vectorpp;

// Helper to create a temporary config file
class TempConfigFile {
public:
    explicit TempConfigFile(const std::string& content) {
        // Use a unique filename
        filename_ = "test_config_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".json";
        std::ofstream file(filename_);
        file << content;
        file.close();
    }

    ~TempConfigFile() {
        std::remove(filename_.c_str());
    }

    const std::string& path() const { return filename_; }

private:
    std::string filename_;
};

// Test loading from string with all fields
TEST(ConfigLoaderTest, LoadFromStringAllFields) {
    std::string json = R"({
        "dimensions": 1536,
        "max_vectors": 50000,
        "hnsw_m": 32,
        "hnsw_ef_construction": 100,
        "hnsw_ef_search": 25
    })";

    VectorStoreConfig config = load_config_from_string(json);

    EXPECT_EQ(config.dimensions, 1536);
    EXPECT_EQ(config.max_vectors, 50000);
    EXPECT_EQ(config.hnsw_m, 32);
    EXPECT_EQ(config.hnsw_ef_construction, 100);
    EXPECT_EQ(config.hnsw_ef_search, 25);
}

// Test loading from string with partial fields
TEST(ConfigLoaderTest, LoadFromStringPartialFields) {
    std::string json = R"({
        "dimensions": 768
    })";

    VectorStoreConfig config = load_config_from_string(json);

    EXPECT_EQ(config.dimensions, 768);
    // Defaults for other fields
    EXPECT_EQ(config.max_vectors, 100000);
    EXPECT_EQ(config.hnsw_m, 16);
    EXPECT_EQ(config.hnsw_ef_construction, 200);
    EXPECT_EQ(config.hnsw_ef_search, 50);
}

// Test loading from empty string returns defaults
TEST(ConfigLoaderTest, LoadFromEmptyString) {
    VectorStoreConfig config = load_config_from_string("");

    EXPECT_EQ(config.dimensions, 384);
    EXPECT_EQ(config.max_vectors, 100000);
}

// Test loading from empty object returns defaults
TEST(ConfigLoaderTest, LoadFromEmptyObject) {
    VectorStoreConfig config = load_config_from_string("{}");

    EXPECT_EQ(config.dimensions, 384);
    EXPECT_EQ(config.max_vectors, 100000);
}

// Test loading from file
TEST(ConfigLoaderTest, LoadFromFile) {
    std::string json = R"({
        "dimensions": 512,
        "max_vectors": 10000
    })";
    TempConfigFile temp(json);

    VectorStoreConfig config = load_config_from_file(temp.path());

    EXPECT_EQ(config.dimensions, 512);
    EXPECT_EQ(config.max_vectors, 10000);
}

// Test loading from non-existent file returns defaults
TEST(ConfigLoaderTest, LoadFromNonExistentFile) {
    VectorStoreConfig config = load_config_from_file("non_existent_file_12345.json");

    EXPECT_EQ(config.dimensions, 384);
    EXPECT_EQ(config.max_vectors, 100000);
}

// Test invalid JSON throws ConfigLoadError
TEST(ConfigLoaderTest, InvalidJsonThrows) {
    EXPECT_THROW(load_config_from_string("{invalid json}"), ConfigLoadError);
    EXPECT_THROW(load_config_from_string("[1, 2, 3]"), ConfigLoadError);
}

// Test invalid dimension value throws
TEST(ConfigLoaderTest, InvalidDimensionsThrows) {
    EXPECT_THROW(load_config_from_string(R"({"dimensions": 0})"), ConfigLoadError);
    EXPECT_THROW(load_config_from_string(R"({"dimensions": -5})"), ConfigLoadError);
    EXPECT_THROW(load_config_from_string(R"({"dimensions": "string"})"), ConfigLoadError);
}

// Test invalid max_vectors value throws
TEST(ConfigLoaderTest, InvalidMaxVectorsThrows) {
    EXPECT_THROW(load_config_from_string(R"({"max_vectors": 0})"), ConfigLoadError);
    EXPECT_THROW(load_config_from_string(R"({"max_vectors": -10})"), ConfigLoadError);
}

// Test that config can be used to create VectorStore
TEST(ConfigLoaderTest, ConfigCreatesVectorStore) {
    std::string json = R"({
        "dimensions": 128,
        "max_vectors": 1000
    })";

    VectorStoreConfig config = load_config_from_string(json);
    VectorStore store(config);

    EXPECT_EQ(store.dimensions(), 128);
    EXPECT_EQ(store.max_vectors(), 1000);
}

// Test loading the actual default config file
TEST(ConfigLoaderTest, LoadDefaultConfigFile) {
    // This test assumes we're running from the build directory
    // Try a few common relative paths
    std::vector<std::string> paths = {
        "../config/default.json",
        "config/default.json",
        "../../config/default.json"
    };

    VectorStoreConfig config;
    bool loaded = false;

    for (const auto& path : paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            file.close();
            config = load_config_from_file(path);
            loaded = true;
            break;
        }
    }

    if (loaded) {
        // Verify it matches our default.json values
        EXPECT_EQ(config.dimensions, 384);
        EXPECT_EQ(config.max_vectors, 100000);
        EXPECT_EQ(config.hnsw_m, 16);
        EXPECT_EQ(config.hnsw_ef_construction, 200);
        EXPECT_EQ(config.hnsw_ef_search, 50);
    }
    // If not loaded, that's okay - default.json might not be in a predictable path
}
