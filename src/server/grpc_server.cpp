#include "server/grpc_server.hpp"
#include <iostream>
#include <sstream>

namespace vectorpp {

GrpcServer::GrpcServer(const ServerConfig& config)
    : config_(config),
      store_(std::make_shared<VectorStore>(config.store_config)),
      service_(std::make_unique<VectorDBServiceImpl>(store_)) {}

GrpcServer::~GrpcServer() {
    shutdown();
}

std::string GrpcServer::getAddress() const {
    std::ostringstream ss;
    ss << config_.address << ":" << config_.port;
    return ss.str();
}

void GrpcServer::run() {
    grpc::ServerBuilder builder;

    std::string server_address = getAddress();

    // Listen on the given address without any authentication mechanism
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register the service implementation
    builder.RegisterService(service_.get());

    // Build and start the server
    server_ = builder.BuildAndStart();

    if (!server_) {
        std::cerr << "Failed to start server on " << server_address << std::endl;
        return;
    }

    running_.store(true);

    std::cout << "==================================================" << std::endl;
    std::cout << "VectorPP gRPC Server" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Listening on:    " << server_address << std::endl;
    std::cout << "Vector dims:     " << config_.store_config.dimensions << std::endl;
    std::cout << "Max vectors:     " << config_.store_config.max_vectors << std::endl;
    std::cout << "Thread pool:     " << (config_.store_config.thread_pool_size == 0
                                         ? "auto"
                                         : std::to_string(config_.store_config.thread_pool_size))
              << std::endl;
    std::cout << "HNSW M:          " << config_.store_config.hnsw_m << std::endl;
    std::cout << "HNSW ef_constr:  " << config_.store_config.hnsw_ef_construction << std::endl;
    std::cout << "HNSW ef_search:  " << config_.store_config.hnsw_ef_search << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Server started. Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;

    // Wait for the server to shutdown
    server_->Wait();

    running_.store(false);
}

void GrpcServer::start() {
    server_thread_ = std::thread([this]() {
        run();
    });

    // Give the server a moment to start
    while (!running_.load() && server_thread_.joinable()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void GrpcServer::shutdown() {
    if (server_) {
        std::cout << "Shutting down server..." << std::endl;
        server_->Shutdown();
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    running_.store(false);
}

} // namespace vectorpp
