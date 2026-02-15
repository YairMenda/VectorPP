#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
#include "vectordb.grpc.pb.h"
#include "server/grpc_server.hpp"
#include <thread>
#include <chrono>
#include <memory>
#include <vector>
#include <random>

namespace {

class GrpcServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        vectorpp::ServerConfig config;
        config.port = 50052;  // Use different port to avoid conflicts
        config.store_config.dimensions = 4;  // Small dimension for tests
        config.store_config.max_vectors = 100;

        server_ = std::make_unique<vectorpp::GrpcServer>(config);
        server_->start();

        // Wait for server to be ready
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Create client channel
        channel_ = grpc::CreateChannel("localhost:50052",
                                       grpc::InsecureChannelCredentials());
        stub_ = vectorpp::VectorDB::NewStub(channel_);
    }

    void TearDown() override {
        stub_.reset();
        channel_.reset();
        if (server_) {
            server_->shutdown();
        }
    }

    std::vector<float> randomVector(size_t dim) {
        std::vector<float> vec(dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (size_t i = 0; i < dim; ++i) {
            vec[i] = dis(gen);
        }
        // Normalize for cosine similarity
        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (float& v : vec) v /= norm;
        }
        return vec;
    }

    std::unique_ptr<vectorpp::GrpcServer> server_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<vectorpp::VectorDB::Stub> stub_;
};

// Test: Insert returns valid UUID
TEST_F(GrpcServerTest, InsertReturnsUUID) {
    vectorpp::InsertRequest request;
    auto vec = randomVector(4);
    for (float v : vec) {
        request.add_vector(v);
    }
    request.set_metadata("test_category");

    vectorpp::InsertResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Insert(&context, request, &response);

    ASSERT_TRUE(status.ok()) << "Insert failed: " << status.error_message();
    ASSERT_FALSE(response.id().empty()) << "UUID should not be empty";
    ASSERT_EQ(response.id().length(), 36) << "UUID should be 36 characters";
}

// Test: Insert with wrong dimension fails
TEST_F(GrpcServerTest, InsertWrongDimensionFails) {
    vectorpp::InsertRequest request;
    // Insert wrong dimension (8 instead of 4)
    for (int i = 0; i < 8; ++i) {
        request.add_vector(static_cast<float>(i));
    }

    vectorpp::InsertResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Insert(&context, request, &response);

    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// Test: Insert with empty vector fails
TEST_F(GrpcServerTest, InsertEmptyVectorFails) {
    vectorpp::InsertRequest request;
    // Don't add any vector values - empty vector

    vectorpp::InsertResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Insert(&context, request, &response);

    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ASSERT_TRUE(status.error_message().find("empty") != std::string::npos);
}

// Test: Search returns results
TEST_F(GrpcServerTest, SearchReturnsResults) {
    // Insert a vector first
    auto vec = randomVector(4);

    vectorpp::InsertRequest insert_req;
    for (float v : vec) {
        insert_req.add_vector(v);
    }
    insert_req.set_metadata("movies");

    vectorpp::InsertResponse insert_resp;
    grpc::ClientContext insert_ctx;
    ASSERT_TRUE(stub_->Insert(&insert_ctx, insert_req, &insert_resp).ok());

    // Search for the same vector
    vectorpp::SearchRequest search_req;
    for (float v : vec) {
        search_req.add_query_vector(v);
    }
    search_req.set_top_k(1);

    vectorpp::SearchResponse search_resp;
    grpc::ClientContext search_ctx;

    grpc::Status status = stub_->Search(&search_ctx, search_req, &search_resp);

    ASSERT_TRUE(status.ok()) << "Search failed: " << status.error_message();
    ASSERT_EQ(search_resp.results_size(), 1);
    ASSERT_EQ(search_resp.results(0).id(), insert_resp.id());
    ASSERT_EQ(search_resp.results(0).metadata(), "movies");
    ASSERT_GT(search_resp.results(0).score(), 0.9f);  // Should be close to 1.0
}

// Test: Search with filter
TEST_F(GrpcServerTest, SearchWithFilter) {
    // Insert vectors with different metadata
    auto vec1 = randomVector(4);
    auto vec2 = randomVector(4);

    // Insert first vector with "movies" metadata
    {
        vectorpp::InsertRequest req;
        for (float v : vec1) req.add_vector(v);
        req.set_metadata("movies");
        vectorpp::InsertResponse resp;
        grpc::ClientContext ctx;
        ASSERT_TRUE(stub_->Insert(&ctx, req, &resp).ok());
    }

    // Insert second vector with "books" metadata
    std::string book_id;
    {
        vectorpp::InsertRequest req;
        for (float v : vec2) req.add_vector(v);
        req.set_metadata("books");
        vectorpp::InsertResponse resp;
        grpc::ClientContext ctx;
        ASSERT_TRUE(stub_->Insert(&ctx, req, &resp).ok());
        book_id = resp.id();
    }

    // Search with filter for "books"
    vectorpp::SearchRequest search_req;
    for (float v : vec2) {
        search_req.add_query_vector(v);
    }
    search_req.set_top_k(10);
    search_req.set_filter_metadata("books");

    vectorpp::SearchResponse search_resp;
    grpc::ClientContext search_ctx;

    grpc::Status status = stub_->Search(&search_ctx, search_req, &search_resp);

    ASSERT_TRUE(status.ok());
    ASSERT_EQ(search_resp.results_size(), 1);
    ASSERT_EQ(search_resp.results(0).id(), book_id);
    ASSERT_EQ(search_resp.results(0).metadata(), "books");
}

// Test: Search with invalid top_k fails
TEST_F(GrpcServerTest, SearchInvalidTopKFails) {
    vectorpp::SearchRequest request;
    for (int i = 0; i < 4; ++i) {
        request.add_query_vector(0.25f);
    }
    request.set_top_k(0);  // Invalid

    vectorpp::SearchResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Search(&context, request, &response);

    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// Test: Search with empty query vector fails
TEST_F(GrpcServerTest, SearchEmptyQueryVectorFails) {
    vectorpp::SearchRequest request;
    // Don't add any query vector values - empty
    request.set_top_k(5);

    vectorpp::SearchResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Search(&context, request, &response);

    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ASSERT_TRUE(status.error_message().find("empty") != std::string::npos);
}

// Test: Search with wrong dimension fails
TEST_F(GrpcServerTest, SearchWrongDimensionFails) {
    // First insert a valid vector
    auto vec = randomVector(4);
    vectorpp::InsertRequest insert_req;
    for (float v : vec) insert_req.add_vector(v);
    vectorpp::InsertResponse insert_resp;
    grpc::ClientContext insert_ctx;
    ASSERT_TRUE(stub_->Insert(&insert_ctx, insert_req, &insert_resp).ok());

    // Search with wrong dimension (8 instead of 4)
    vectorpp::SearchRequest request;
    for (int i = 0; i < 8; ++i) {
        request.add_query_vector(0.125f);
    }
    request.set_top_k(5);

    vectorpp::SearchResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Search(&context, request, &response);

    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// Test: Delete existing vector succeeds
TEST_F(GrpcServerTest, DeleteExistingSucceeds) {
    // Insert a vector
    auto vec = randomVector(4);
    vectorpp::InsertRequest insert_req;
    for (float v : vec) insert_req.add_vector(v);

    vectorpp::InsertResponse insert_resp;
    grpc::ClientContext insert_ctx;
    ASSERT_TRUE(stub_->Insert(&insert_ctx, insert_req, &insert_resp).ok());

    // Delete it
    vectorpp::DeleteRequest delete_req;
    delete_req.set_id(insert_resp.id());

    vectorpp::DeleteResponse delete_resp;
    grpc::ClientContext delete_ctx;

    grpc::Status status = stub_->Delete(&delete_ctx, delete_req, &delete_resp);

    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(delete_resp.success());

    // Verify it's no longer searchable
    vectorpp::SearchRequest search_req;
    for (float v : vec) search_req.add_query_vector(v);
    search_req.set_top_k(10);

    vectorpp::SearchResponse search_resp;
    grpc::ClientContext search_ctx;
    stub_->Search(&search_ctx, search_req, &search_resp);

    // Should not find deleted vector
    for (const auto& result : search_resp.results()) {
        ASSERT_NE(result.id(), insert_resp.id());
    }
}

// Test: Delete non-existent vector returns NOT_FOUND
TEST_F(GrpcServerTest, DeleteNonExistentReturnsNotFound) {
    vectorpp::DeleteRequest request;
    request.set_id("00000000-0000-0000-0000-000000000000");

    vectorpp::DeleteResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Delete(&context, request, &response);

    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
    ASSERT_FALSE(response.success());
}

// Test: Delete with empty ID fails
TEST_F(GrpcServerTest, DeleteEmptyIDFails) {
    vectorpp::DeleteRequest request;
    request.set_id("");

    vectorpp::DeleteResponse response;
    grpc::ClientContext context;

    grpc::Status status = stub_->Delete(&context, request, &response);

    ASSERT_FALSE(status.ok());
    ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

// Test: Multiple concurrent requests
TEST_F(GrpcServerTest, ConcurrentRequests) {
    const int num_vectors = 10;
    std::vector<std::string> ids;
    std::vector<std::thread> threads;

    // Insert vectors concurrently
    std::mutex ids_mutex;
    for (int i = 0; i < num_vectors; ++i) {
        threads.emplace_back([this, i, &ids, &ids_mutex]() {
            auto vec = randomVector(4);
            vectorpp::InsertRequest request;
            for (float v : vec) request.add_vector(v);
            request.set_metadata("thread_" + std::to_string(i));

            vectorpp::InsertResponse response;
            grpc::ClientContext context;
            grpc::Status status = stub_->Insert(&context, request, &response);

            ASSERT_TRUE(status.ok());

            std::lock_guard<std::mutex> lock(ids_mutex);
            ids.push_back(response.id());
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    ASSERT_EQ(ids.size(), num_vectors);

    // Verify all vectors can be searched
    for (int i = 0; i < num_vectors; ++i) {
        vectorpp::SearchRequest request;
        auto vec = randomVector(4);
        for (float v : vec) request.add_query_vector(v);
        request.set_top_k(num_vectors);

        vectorpp::SearchResponse response;
        grpc::ClientContext context;
        grpc::Status status = stub_->Search(&context, request, &response);

        ASSERT_TRUE(status.ok());
        ASSERT_EQ(response.results_size(), num_vectors);
    }
}

}  // namespace
