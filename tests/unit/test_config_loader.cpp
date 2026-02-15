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

// ==================== ServerConfig Tests ====================

// Test loading ServerConfig from string with all fields
TEST(ServerConfigLoaderTest, LoadFromStringAllFields) {
    std::string json = R"({
        "address": "127.0.0.1",
        "port": 8080,
        "dimensions": 1536,
        "max_vectors": 50000,
        "hnsw_m": 32,
        "hnsw_ef_construction": 100,
        "hnsw_ef_search": 25
    })";

    ServerConfig config = load_server_config_from_string(json);

    EXPECT_EQ(config.address, "127.0.0.1");
    EXPECT_EQ(config.port, 8080);
    EXPECT_EQ(config.store_config.dimensions, 1536);
    EXPECT_EQ(config.store_config.max_vectors, 50000);
    EXPECT_EQ(config.store_config.hnsw_m, 32);
    EXPECT_EQ(config.store_config.hnsw_ef_construction, 100);
    EXPECT_EQ(config.store_config.hnsw_ef_search, 25);
}

// Test loading ServerConfig with only server fields (store defaults)
TEST(ServerConfigLoaderTest, LoadFromStringServerFieldsOnly) {
    std::string json = R"({
        "address": "localhost",
        "port": 9090
    })";

    ServerConfig config = load_server_config_from_string(json);

    EXPECT_EQ(config.address, "localhost");
    EXPECT_EQ(config.port, 9090);
    // Defaults for store config
    EXPECT_EQ(config.store_config.dimensions, 384);
    EXPECT_EQ(config.store_config.max_vectors, 100000);
}

// Test loading ServerConfig with only store fields (server defaults)
TEST(ServerConfigLoaderTest, LoadFromStringStoreFieldsOnly) {
    std::string json = R"({
        "dimensions": 768,
        "max_vectors": 5000
    })";

    ServerConfig config = load_server_config_from_string(json);

    // Defaults for server fields
    EXPECT_EQ(config.address, "0.0.0.0");
    EXPECT_EQ(config.port, 50051);
    // Specified store fields
    EXPECT_EQ(config.store_config.dimensions, 768);
    EXPECT_EQ(config.store_config.max_vectors, 5000);
}

// Test loading ServerConfig from empty string returns defaults
TEST(ServerConfigLoaderTest, LoadFromEmptyString) {
    ServerConfig config = load_server_config_from_string("");

    EXPECT_EQ(config.address, "0.0.0.0");
    EXPECT_EQ(config.port, 50051);
    EXPECT_EQ(config.store_config.dimensions, 384);
    EXPECT_EQ(config.store_config.max_vectors, 100000);
}

// Test loading ServerConfig from empty object returns defaults
TEST(ServerConfigLoaderTest, LoadFromEmptyObject) {
    ServerConfig config = load_server_config_from_string("{}");

    EXPECT_EQ(config.address, "0.0.0.0");
    EXPECT_EQ(config.port, 50051);
    EXPECT_EQ(config.store_config.dimensions, 384);
}

// Test loading ServerConfig from file
TEST(ServerConfigLoaderTest, LoadFromFile) {
    std::string json = R"({
        "address": "192.168.1.100",
        "port": 12345,
        "dimensions": 512
    })";
    TempConfigFile temp(json);

    ServerConfig config = load_server_config_from_file(temp.path());

    EXPECT_EQ(config.address, "192.168.1.100");
    EXPECT_EQ(config.port, 12345);
    EXPECT_EQ(config.store_config.dimensions, 512);
}

// Test loading ServerConfig from non-existent file returns defaults
TEST(ServerConfigLoaderTest, LoadFromNonExistentFile) {
    ServerConfig config = load_server_config_from_file("non_existent_server_config_12345.json");

    EXPECT_EQ(config.address, "0.0.0.0");
    EXPECT_EQ(config.port, 50051);
    EXPECT_EQ(config.store_config.dimensions, 384);
}

// Test invalid port value (too high) throws
TEST(ServerConfigLoaderTest, InvalidPortTooHighThrows) {
    EXPECT_THROW(load_server_config_from_string(R"({"port": 70000})"), ConfigLoadError);
}

// Test invalid port type throws
TEST(ServerConfigLoaderTest, InvalidPortTypeThrows) {
    EXPECT_THROW(load_server_config_from_string(R"({"port": "8080"})"), ConfigLoadError);
    EXPECT_THROW(load_server_config_from_string(R"({"port": -1})"), ConfigLoadError);
}

// Test invalid address type throws
TEST(ServerConfigLoaderTest, InvalidAddressTypeThrows) {
    EXPECT_THROW(load_server_config_from_string(R"({"address": 12345})"), ConfigLoadError);
    EXPECT_THROW(load_server_config_from_string(R"({"address": true})"), ConfigLoadError);
}

// Test loading actual default.json with server config
TEST(ServerConfigLoaderTest, LoadDefaultConfigFile) {
    std::vector<std::string> paths = {
        "../config/default.json",
        "config/default.json",
        "../../config/default.json"
    };

    ServerConfig config;
    bool loaded = false;

    for (const auto& path : paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            file.close();
            config = load_server_config_from_file(path);
            loaded = true;
            break;
        }
    }

    if (loaded) {
        // Verify it matches our default.json values
        EXPECT_EQ(config.address, "0.0.0.0");
        EXPECT_EQ(config.port, 50051);
        EXPECT_EQ(config.store_config.dimensions, 384);
        EXPECT_EQ(config.store_config.max_vectors, 100000);
        EXPECT_EQ(config.store_config.hnsw_m, 16);
        EXPECT_EQ(config.store_config.hnsw_ef_construction, 200);
        EXPECT_EQ(config.store_config.hnsw_ef_search, 50);
    }
}
