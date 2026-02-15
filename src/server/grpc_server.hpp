#pragma once

#include <grpcpp/grpcpp.h>
#include "server/service_impl.hpp"
#include "config/config_loader.hpp"
#include <memory>
#include <string>
#include <thread>
#include <atomic>

namespace vectorpp {

// gRPC server wrapper that manages the VectorDB service
class GrpcServer {
public:
    explicit GrpcServer(const ServerConfig& config);
    ~GrpcServer();

    // Non-copyable
    GrpcServer(const GrpcServer&) = delete;
    GrpcServer& operator=(const GrpcServer&) = delete;

    // Start the server (blocking)
    void run();

    // Start the server in background thread
    void start();

    // Shutdown the server gracefully
    void shutdown();

    // Check if server is running
    bool isRunning() const { return running_.load(); }

    // Get the listening address
    std::string getAddress() const;

    // Get the port
    uint16_t getPort() const { return config_.port; }

private:
    ServerConfig config_;
    std::shared_ptr<VectorStore> store_;
    std::unique_ptr<VectorDBServiceImpl> service_;
    std::unique_ptr<grpc::Server> server_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
};

} // namespace vectorpp
