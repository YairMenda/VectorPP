#pragma once

#include "core/vector_store.hpp"
#include <string>
#include <cstdint>
#include <stdexcept>

namespace vectorpp {

// Exception for configuration loading errors
class ConfigLoadError : public std::runtime_error {
public:
    explicit ConfigLoadError(const std::string& message)
        : std::runtime_error("Configuration error: " + message) {}
};

// Full server configuration including network and store settings
struct ServerConfig {
    std::string address = "0.0.0.0";
    uint16_t port = 50051;
    VectorStoreConfig store_config;
};

// Load ServerConfig from a JSON file
// Returns default config if file doesn't exist or is empty
// Throws ConfigLoadError for invalid JSON or invalid values
ServerConfig load_server_config_from_file(const std::string& file_path);

// Load ServerConfig from a JSON string
// Throws ConfigLoadError for invalid JSON or invalid values
ServerConfig load_server_config_from_string(const std::string& json_string);

// Load VectorStoreConfig from a JSON file (legacy, for backwards compatibility)
// Returns default config if file doesn't exist or is empty
// Throws ConfigLoadError for invalid JSON or invalid values
VectorStoreConfig load_config_from_file(const std::string& file_path);

// Load VectorStoreConfig from a JSON string (legacy, for backwards compatibility)
// Throws ConfigLoadError for invalid JSON or invalid values
VectorStoreConfig load_config_from_string(const std::string& json_string);

} // namespace vectorpp
