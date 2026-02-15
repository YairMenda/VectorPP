#pragma once

#include "core/vector_store.hpp"
#include <string>
#include <stdexcept>

namespace vectorpp {

// Exception for configuration loading errors
class ConfigLoadError : public std::runtime_error {
public:
    explicit ConfigLoadError(const std::string& message)
        : std::runtime_error("Configuration error: " + message) {}
};

// Load VectorStoreConfig from a JSON file
// Returns default config if file doesn't exist or is empty
// Throws ConfigLoadError for invalid JSON or invalid values
VectorStoreConfig load_config_from_file(const std::string& file_path);

// Load VectorStoreConfig from a JSON string
// Throws ConfigLoadError for invalid JSON or invalid values
VectorStoreConfig load_config_from_string(const std::string& json_string);

} // namespace vectorpp
