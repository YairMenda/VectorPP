#include "server/service_impl.hpp"
#include <sstream>

namespace vectorpp {

VectorDBServiceImpl::VectorDBServiceImpl(std::shared_ptr<VectorStore> store)
    : store_(std::move(store)) {}

void VectorDBServiceImpl::logRequest(const std::string& method, const std::string& details) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    // Build log message first, then write atomically with lock
    std::ostringstream log_msg;
    log_msg << "[" << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
            << "." << std::setfill('0') << std::setw(3) << ms.count() << "] "
            << method;
    if (!details.empty()) {
        log_msg << " - " << details;
    }
    log_msg << "\n";

    // Thread-safe output
    std::lock_guard<std::mutex> lock(log_mutex_);
    std::cout << log_msg.str() << std::flush;
}

grpc::Status VectorDBServiceImpl::Insert(grpc::ServerContext* context,
                                         const vectorpp::InsertRequest* request,
                                         vectorpp::InsertResponse* response) {
    try {
        // Convert proto repeated float to std::vector<float>
        std::vector<float> vector(request->vector().begin(), request->vector().end());

        std::ostringstream details;
        details << "vector_dim=" << vector.size();
        if (!request->metadata().empty()) {
            details << ", metadata=\"" << request->metadata() << "\"";
        }
        logRequest("Insert", details.str());

        // Validate vector is not empty
        if (vector.empty()) {
            logRequest("Insert", "FAILED - vector cannot be empty");
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "vector cannot be empty");
        }

        // Perform insert operation
        std::string id = store_->insert(vector, request->metadata());
        response->set_id(id);

        return grpc::Status::OK;

    } catch (const DimensionMismatchError& e) {
        logRequest("Insert", std::string("FAILED - ") + e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());

    } catch (const CapacityLimitError& e) {
        logRequest("Insert", std::string("FAILED - ") + e.what());
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, e.what());

    } catch (const std::exception& e) {
        logRequest("Insert", std::string("FAILED - ") + e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status VectorDBServiceImpl::Search(grpc::ServerContext* context,
                                         const vectorpp::SearchRequest* request,
                                         vectorpp::SearchResponse* response) {
    try {
        // Convert proto repeated float to std::vector<float>
        std::vector<float> query(request->query_vector().begin(), request->query_vector().end());

        std::ostringstream details;
        details << "vector_dim=" << query.size() << ", top_k=" << request->top_k();
        if (!request->filter_metadata().empty()) {
            details << ", filter=\"" << request->filter_metadata() << "\"";
        }
        logRequest("Search", details.str());

        // Validate query vector is not empty
        if (query.empty()) {
            logRequest("Search", "FAILED - query_vector cannot be empty");
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "query_vector cannot be empty");
        }

        // Validate top_k
        if (request->top_k() <= 0) {
            logRequest("Search", "FAILED - top_k must be positive");
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "top_k must be positive");
        }

        // Perform search operation
        auto results = store_->search(query, static_cast<size_t>(request->top_k()),
                                       request->filter_metadata());

        // Convert results to proto format
        // (result is a VectorSearchResult from vector_store.hpp)
        for (const auto& r : results) {
            auto* proto_result = response->add_results();
            proto_result->set_id(r.id);
            proto_result->set_score(r.score);
            proto_result->set_metadata(r.metadata);
        }

        return grpc::Status::OK;

    } catch (const DimensionMismatchError& e) {
        logRequest("Search", std::string("FAILED - ") + e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, e.what());

    } catch (const std::exception& e) {
        logRequest("Search", std::string("FAILED - ") + e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

grpc::Status VectorDBServiceImpl::Delete(grpc::ServerContext* context,
                                         const vectorpp::DeleteRequest* request,
                                         vectorpp::DeleteResponse* response) {
    try {
        logRequest("Delete", "id=" + request->id());

        // Validate ID is not empty
        if (request->id().empty()) {
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "id cannot be empty");
        }

        // Perform delete operation
        bool success = store_->remove(request->id());
        response->set_success(success);

        if (!success) {
            // ID not found - return NOT_FOUND status but still set success=false
            return grpc::Status(grpc::StatusCode::NOT_FOUND,
                              "Vector with id '" + request->id() + "' not found");
        }

        return grpc::Status::OK;

    } catch (const std::exception& e) {
        logRequest("Delete", std::string("FAILED - ") + e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
    }
}

} // namespace vectorpp
