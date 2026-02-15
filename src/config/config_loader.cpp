#include "config/config_loader.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

namespace vectorpp {

namespace {

// Helper to parse JSON from string
nlohmann::json parseJson(const std::string& json_string) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_string);
    } catch (const nlohmann::json::parse_error& e) {
        throw ConfigLoadError("Invalid JSON: " + std::string(e.what()));
    }

    if (!j.is_object()) {
        throw ConfigLoadError("JSON root must be an object");
    }

    return j;
}

// Helper to parse VectorStoreConfig fields from JSON object
void parseVectorStoreConfig(const nlohmann::json& j, VectorStoreConfig& config) {
    // Parse dimensions
    if (j.contains("dimensions")) {
        if (!j["dimensions"].is_number_unsigned()) {
            throw ConfigLoadError("'dimensions' must be a positive integer");
        }
        size_t dims = j["dimensions"].get<size_t>();
        if (dims == 0) {
            throw ConfigLoadError("'dimensions' must be greater than 0");
        }
        config.dimensions = dims;
    }

    // Parse max_vectors
    if (j.contains("max_vectors")) {
        if (!j["max_vectors"].is_number_unsigned()) {
            throw ConfigLoadError("'max_vectors' must be a positive integer");
        }
        size_t max_vec = j["max_vectors"].get<size_t>();
        if (max_vec == 0) {
            throw ConfigLoadError("'max_vectors' must be greater than 0");
        }
        config.max_vectors = max_vec;
    }

    // Parse HNSW parameters
    if (j.contains("hnsw_m")) {
        if (!j["hnsw_m"].is_number_unsigned()) {
            throw ConfigLoadError("'hnsw_m' must be a positive integer");
        }
        config.hnsw_m = j["hnsw_m"].get<size_t>();
    }

    if (j.contains("hnsw_ef_construction")) {
        if (!j["hnsw_ef_construction"].is_number_unsigned()) {
            throw ConfigLoadError("'hnsw_ef_construction' must be a positive integer");
        }
        config.hnsw_ef_construction = j["hnsw_ef_construction"].get<size_t>();
    }

    if (j.contains("hnsw_ef_search")) {
        if (!j["hnsw_ef_search"].is_number_unsigned()) {
            throw ConfigLoadError("'hnsw_ef_search' must be a positive integer");
        }
        config.hnsw_ef_search = j["hnsw_ef_search"].get<size_t>();
    }

    // Parse thread_pool_size
    if (j.contains("thread_pool_size")) {
        if (!j["thread_pool_size"].is_number_unsigned()) {
            throw ConfigLoadError("'thread_pool_size' must be a non-negative integer");
        }
        config.thread_pool_size = j["thread_pool_size"].get<size_t>();
    }
}

// Helper to read file contents
std::string readFileContents(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        return "";  // File doesn't exist
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

} // anonymous namespace

VectorStoreConfig load_config_from_string(const std::string& json_string) {
    VectorStoreConfig config;

    if (json_string.empty()) {
        return config;  // Return defaults for empty string
    }

    nlohmann::json j = parseJson(json_string);
    parseVectorStoreConfig(j, config);

    return config;
}

VectorStoreConfig load_config_from_file(const std::string& file_path) {
    std::string content = readFileContents(file_path);

    if (content.empty()) {
        return VectorStoreConfig{};  // Empty or non-existent file - return defaults
    }

    return load_config_from_string(content);
}

ServerConfig load_server_config_from_string(const std::string& json_string) {
    ServerConfig config;

    if (json_string.empty()) {
        return config;  // Return defaults for empty string
    }

    nlohmann::json j = parseJson(json_string);

    // Parse server network settings
    if (j.contains("address")) {
        if (!j["address"].is_string()) {
            throw ConfigLoadError("'address' must be a string");
        }
        config.address = j["address"].get<std::string>();
    }

    if (j.contains("port")) {
        if (!j["port"].is_number_unsigned()) {
            throw ConfigLoadError("'port' must be a positive integer");
        }
        auto port_val = j["port"].get<uint64_t>();
        if (port_val > 65535) {
            throw ConfigLoadError("'port' must be between 0 and 65535");
        }
        config.port = static_cast<uint16_t>(port_val);
    }

    // Parse vector store config (all fields are at root level)
    parseVectorStoreConfig(j, config.store_config);

    return config;
}

ServerConfig load_server_config_from_file(const std::string& file_path) {
    std::string content = readFileContents(file_path);

    if (content.empty()) {
        return ServerConfig{};  // Empty or non-existent file - return defaults
    }

    return load_server_config_from_string(content);
}

} // namespace vectorpp
