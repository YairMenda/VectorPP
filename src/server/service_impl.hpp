#pragma once

#include <grpcpp/grpcpp.h>
#include "vectordb.grpc.pb.h"
#include "core/vector_store.hpp"
#include <memory>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <mutex>

namespace vectorpp {

// Implementation of the VectorDB gRPC service
// Thread-safe: All methods can be called concurrently from multiple gRPC threads
class VectorDBServiceImpl final : public vectorpp::VectorDB::Service {
public:
    explicit VectorDBServiceImpl(std::shared_ptr<VectorStore> store);

    // Insert a vector and return its UUID
    grpc::Status Insert(grpc::ServerContext* context,
                       const vectorpp::InsertRequest* request,
                       vectorpp::InsertResponse* response) override;

    // Search for similar vectors
    grpc::Status Search(grpc::ServerContext* context,
                       const vectorpp::SearchRequest* request,
                       vectorpp::SearchResponse* response) override;

    // Delete a vector by ID
    grpc::Status Delete(grpc::ServerContext* context,
                       const vectorpp::DeleteRequest* request,
                       vectorpp::DeleteResponse* response) override;

private:
    std::shared_ptr<VectorStore> store_;
    mutable std::mutex log_mutex_;  // Protects stdout logging from concurrent access

    // Helper to log requests (thread-safe)
    void logRequest(const std::string& method, const std::string& details = "");
};

} // namespace vectorpp
