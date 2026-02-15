#include "server/grpc_server.hpp"
#include "config/config_loader.hpp"
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>

namespace {
    std::atomic<bool> shutdown_requested{false};
    vectorpp::GrpcServer* server_ptr = nullptr;
}

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << std::endl;
    shutdown_requested.store(true);
    if (server_ptr) {
        server_ptr->shutdown();
    }
}

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nOptions:\n"
              << "  --config <file>    Path to JSON config file\n"
              << "  --port <port>      Server port (default: 50051)\n"
              << "  --address <addr>   Server address (default: 0.0.0.0)\n"
              << "  --dims <n>         Vector dimensions (default: 384)\n"
              << "  --max-vectors <n>  Maximum vectors (default: 100000)\n"
              << "  --help             Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    vectorpp::ServerConfig config;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--config" && i + 1 < argc) {
            try {
                config.store_config = vectorpp::load_config_from_file(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Error loading config: " << e.what() << std::endl;
                return 1;
            }
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--address" && i + 1 < argc) {
            config.address = argv[++i];
        } else if (arg == "--dims" && i + 1 < argc) {
            config.store_config.dimensions = static_cast<size_t>(std::stoi(argv[++i]));
        } else if (arg == "--max-vectors" && i + 1 < argc) {
            config.store_config.max_vectors = static_cast<size_t>(std::stoi(argv[++i]));
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    try {
        vectorpp::GrpcServer server(config);
        server_ptr = &server;

        // Run the server (blocking)
        server.run();

        server_ptr = nullptr;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
