#include "config/config_loader.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

namespace vectorpp {

VectorStoreConfig load_config_from_string(const std::string& json_string) {
    VectorStoreConfig config;

    if (json_string.empty()) {
        return config;  // Return defaults for empty string
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_string);
    } catch (const nlohmann::json::parse_error& e) {
        throw ConfigLoadError("Invalid JSON: " + std::string(e.what()));
    }

    if (!j.is_object()) {
        throw ConfigLoadError("JSON root must be an object");
    }

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

    return config;
}

VectorStoreConfig load_config_from_file(const std::string& file_path) {
    std::ifstream file(file_path);

    if (!file.is_open()) {
        // File doesn't exist - return default config
        return VectorStoreConfig{};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    if (content.empty()) {
        return VectorStoreConfig{};  // Empty file - return defaults
    }

    return load_config_from_string(content);
}

} // namespace vectorpp
