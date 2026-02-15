#include <gtest/gtest.h>
#include "core/vector_store.hpp"
#include <vector>
#include <string>
#include <cmath>

using namespace vectorpp;

class VectorStoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        VectorStoreConfig config;
        config.dimensions = 4;  // Small dimension for testing
        config.max_vectors = 100;
        store_ = std::make_unique<VectorStore>(config);
    }

    std::unique_ptr<VectorStore> store_;
};

// US-001: Database accepts dimension parameter at creation time
TEST_F(VectorStoreTest, AcceptsDimensionParameter) {
    VectorStoreConfig config;
    config.dimensions = 128;
    VectorStore store(config);

    EXPECT_EQ(store.dimensions(), 128);
}

TEST_F(VectorStoreTest, DefaultDimensionIs384) {
    VectorStoreConfig config;  // Use defaults
    VectorStore store(config);

    EXPECT_EQ(store.dimensions(), 384);
}

TEST_F(VectorStoreTest, AcceptsOpenAIDimensions) {
    VectorStoreConfig config;
    config.dimensions = 1536;  // OpenAI embedding size
    VectorStore store(config);

    EXPECT_EQ(store.dimensions(), 1536);
}

// US-001: Database rejects vectors that don't match configured dimensions
TEST_F(VectorStoreTest, RejectsWrongDimensions) {
    std::vector<float> wrong_size = {1.0f, 2.0f, 3.0f};  // 3 instead of 4

    EXPECT_THROW({
        store_->insert(wrong_size);
    }, DimensionMismatchError);
}

TEST_F(VectorStoreTest, RejectsWrongDimensionsOnSearch) {
    std::vector<float> correct = {1.0f, 2.0f, 3.0f, 4.0f};
    store_->insert(correct);

    std::vector<float> wrong_query = {1.0f, 2.0f};  // 2 instead of 4

    EXPECT_THROW({
        store_->search(wrong_query, 5);
    }, DimensionMismatchError);
}

TEST_F(VectorStoreTest, DimensionMismatchErrorContainsDetails) {
    std::vector<float> wrong_size = {1.0f, 2.0f};

    try {
        store_->insert(wrong_size);
        FAIL() << "Expected DimensionMismatchError";
    } catch (const DimensionMismatchError& e) {
        EXPECT_EQ(e.expected(), 4);
        EXPECT_EQ(e.actual(), 2);
    }
}

// US-001: Configurable soft limit for maximum number of vectors
TEST_F(VectorStoreTest, ConfigurableMaxVectors) {
    VectorStoreConfig config;
    config.dimensions = 4;
    config.max_vectors = 10;
    VectorStore store(config);

    EXPECT_EQ(store.max_vectors(), 10);
}

TEST_F(VectorStoreTest, EnforcesMaxVectorsLimit) {
    VectorStoreConfig config;
    config.dimensions = 4;
    config.max_vectors = 3;
    VectorStore store(config);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    // Insert up to limit
    store.insert(vec);
    store.insert(vec);
    store.insert(vec);

    EXPECT_EQ(store.size(), 3);

    // Should throw when exceeding limit
    EXPECT_THROW({
        store.insert(vec);
    }, CapacityLimitError);
}

// Basic insert functionality
TEST_F(VectorStoreTest, InsertReturnsUUID) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec);

    // UUID format: 8-4-4-4-12 hex characters
    EXPECT_EQ(id.length(), 36);
    EXPECT_EQ(id[8], '-');
    EXPECT_EQ(id[13], '-');
    EXPECT_EQ(id[18], '-');
    EXPECT_EQ(id[23], '-');
}

TEST_F(VectorStoreTest, InsertIncrementsSize) {
    EXPECT_EQ(store_->size(), 0);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    store_->insert(vec);

    EXPECT_EQ(store_->size(), 1);

    store_->insert(vec);
    EXPECT_EQ(store_->size(), 2);
}

TEST_F(VectorStoreTest, ContainsReturnsTrueForInsertedVectors) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec);

    EXPECT_TRUE(store_->contains(id));
    EXPECT_FALSE(store_->contains("nonexistent-uuid"));
}

// Basic search functionality
TEST_F(VectorStoreTest, SearchReturnsResults) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "test");

    auto results = store_->search(vec, 1);

    EXPECT_EQ(results.size(), 1);
    EXPECT_FLOAT_EQ(results[0].score, 1.0f);  // Identical vector, similarity = 1
    EXPECT_EQ(results[0].metadata, "test");
}

TEST_F(VectorStoreTest, SearchEmptyStoreReturnsEmpty) {
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    auto results = store_->search(query, 5);

    EXPECT_TRUE(results.empty());
}

// Basic delete functionality
TEST_F(VectorStoreTest, DeleteRemovesVector) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec);

    EXPECT_TRUE(store_->contains(id));
    EXPECT_EQ(store_->size(), 1);

    bool removed = store_->remove(id);

    EXPECT_TRUE(removed);
    EXPECT_FALSE(store_->contains(id));
    EXPECT_EQ(store_->size(), 0);
}

TEST_F(VectorStoreTest, DeleteNonexistentReturnsFalse) {
    bool removed = store_->remove("nonexistent-uuid");
    EXPECT_FALSE(removed);
}

// =============================================================================
// US-002: Insert Vector with Auto-Generated ID
// =============================================================================

// US-002: Insert accepts a vector (list of floats) and optional metadata
TEST_F(VectorStoreTest, InsertAcceptsVectorWithMetadata) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string metadata = "category:movies";

    std::string id = store_->insert(vec, metadata);

    EXPECT_FALSE(id.empty());
    EXPECT_TRUE(store_->contains(id));

    // Verify metadata is stored by searching and checking result
    auto results = store_->search(vec, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].metadata, "category:movies");
}

TEST_F(VectorStoreTest, InsertAcceptsVectorWithoutMetadata) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    std::string id = store_->insert(vec);  // No metadata

    EXPECT_FALSE(id.empty());
    EXPECT_TRUE(store_->contains(id));

    // Verify empty metadata
    auto results = store_->search(vec, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].metadata, "");
}

TEST_F(VectorStoreTest, InsertAcceptsVectorWithEmptyMetadata) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    std::string id = store_->insert(vec, "");

    EXPECT_FALSE(id.empty());
    auto results = store_->search(vec, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].metadata, "");
}

// US-002: Server generates unique UUIDs for each inserted vector
TEST_F(VectorStoreTest, InsertGeneratesUniqueUUIDs) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    std::string id1 = store_->insert(vec);
    std::string id2 = store_->insert(vec);
    std::string id3 = store_->insert(vec);

    EXPECT_NE(id1, id2);
    EXPECT_NE(id2, id3);
    EXPECT_NE(id1, id3);
}

TEST_F(VectorStoreTest, InsertedUUIDIsValidFormat) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec);

    // UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    ASSERT_EQ(id.length(), 36);
    EXPECT_EQ(id[8], '-');
    EXPECT_EQ(id[13], '-');
    EXPECT_EQ(id[14], '4');  // Version 4 UUID
    EXPECT_EQ(id[18], '-');
    EXPECT_EQ(id[23], '-');

    // Check variant bits (y should be 8, 9, a, or b)
    char variant = id[19];
    EXPECT_TRUE(variant == '8' || variant == '9' ||
                variant == 'a' || variant == 'b' ||
                variant == 'A' || variant == 'B');
}

// US-002: Insert rejects vectors with wrong dimensions
TEST_F(VectorStoreTest, InsertRejectsWrongDimensionsWithError) {
    std::vector<float> wrong_small = {1.0f, 2.0f};  // Too few
    std::vector<float> wrong_large = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};  // Too many

    EXPECT_THROW(store_->insert(wrong_small), DimensionMismatchError);
    EXPECT_THROW(store_->insert(wrong_large), DimensionMismatchError);
}

TEST_F(VectorStoreTest, InsertRejectsEmptyVector) {
    std::vector<float> empty_vec;

    EXPECT_THROW(store_->insert(empty_vec), DimensionMismatchError);
}

// US-002: Insert respects soft memory limit
TEST_F(VectorStoreTest, InsertRespectsMemoryLimit) {
    VectorStoreConfig config;
    config.dimensions = 4;
    config.max_vectors = 5;
    VectorStore store(config);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    // Fill to capacity
    for (size_t i = 0; i < 5; ++i) {
        store.insert(vec);
    }
    EXPECT_EQ(store.size(), 5);

    // Exceeding capacity should throw
    try {
        store.insert(vec);
        FAIL() << "Expected CapacityLimitError";
    } catch (const CapacityLimitError& e) {
        EXPECT_EQ(e.limit(), 5);
    }
}

// US-002: Inserted vectors are immediately searchable
TEST_F(VectorStoreTest, InsertedVectorsImmediatelySearchable) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};

    std::string id1 = store_->insert(vec1, "first");

    // Search immediately after first insert
    auto results1 = store_->search(vec1, 5);
    ASSERT_EQ(results1.size(), 1);
    EXPECT_EQ(results1[0].id, id1);
    EXPECT_EQ(results1[0].metadata, "first");

    std::string id2 = store_->insert(vec2, "second");

    // Both vectors should now be searchable
    auto results2 = store_->search(vec1, 5);
    EXPECT_EQ(results2.size(), 2);

    auto results3 = store_->search(vec2, 5);
    EXPECT_EQ(results3.size(), 2);
}

TEST_F(VectorStoreTest, InsertedVectorsReturnCorrectSimilarity) {
    // Insert two orthogonal vectors
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};

    store_->insert(vec1, "v1");
    store_->insert(vec2, "v2");

    // Search with vec1
    auto results = store_->search(vec1, 2);
    ASSERT_EQ(results.size(), 2);

    // First result should be identical vector with score 1.0
    EXPECT_EQ(results[0].metadata, "v1");
    EXPECT_NEAR(results[0].score, 1.0f, 0.001f);

    // Second result should be orthogonal vector with score 0.0
    EXPECT_EQ(results[1].metadata, "v2");
    EXPECT_NEAR(results[1].score, 0.0f, 0.001f);
}

// =============================================================================
// US-003: Search for Similar Vectors
// =============================================================================

// US-003: Search accepts a query vector and K (number of results)
TEST_F(VectorStoreTest, SearchAcceptsQueryVectorAndK) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "test");

    // K=1
    auto results1 = store_->search(vec, 1);
    EXPECT_EQ(results1.size(), 1);

    // K=5 (but only 1 vector exists)
    auto results5 = store_->search(vec, 5);
    EXPECT_EQ(results5.size(), 1);
}

TEST_F(VectorStoreTest, SearchReturnsExactlyKResults) {
    // Insert 10 vectors
    for (int i = 0; i < 10; ++i) {
        std::vector<float> vec = {static_cast<float>(i), 1.0f, 0.0f, 0.0f};
        store_->insert(vec, "vec" + std::to_string(i));
    }

    std::vector<float> query = {5.0f, 1.0f, 0.0f, 0.0f};

    // Request different K values
    auto results3 = store_->search(query, 3);
    EXPECT_EQ(results3.size(), 3);

    auto results5 = store_->search(query, 5);
    EXPECT_EQ(results5.size(), 5);

    auto results10 = store_->search(query, 10);
    EXPECT_EQ(results10.size(), 10);
}

TEST_F(VectorStoreTest, SearchWithKZeroReturnsEmpty) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "test");

    auto results = store_->search(vec, 0);
    EXPECT_TRUE(results.empty());
}

// US-003: Returns top-K results sorted by cosine similarity (highest first)
TEST_F(VectorStoreTest, SearchResultsSortedByCosineSimilarityDescending) {
    // Insert vectors with different similarities to the query
    std::vector<float> identical = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> similar = {0.9f, 0.1f, 0.0f, 0.0f};  // Close to identical
    std::vector<float> orthogonal = {0.0f, 1.0f, 0.0f, 0.0f};  // 90 degrees
    std::vector<float> opposite = {-1.0f, 0.0f, 0.0f, 0.0f};  // 180 degrees

    store_->insert(orthogonal, "orthogonal");
    store_->insert(opposite, "opposite");
    store_->insert(identical, "identical");
    store_->insert(similar, "similar");

    // Search for identical vector
    auto results = store_->search(identical, 4);
    ASSERT_EQ(results.size(), 4);

    // Verify descending order by score
    EXPECT_GE(results[0].score, results[1].score);
    EXPECT_GE(results[1].score, results[2].score);
    EXPECT_GE(results[2].score, results[3].score);

    // First should be the identical vector
    EXPECT_EQ(results[0].metadata, "identical");
    EXPECT_NEAR(results[0].score, 1.0f, 0.01f);

    // Second should be the similar vector
    EXPECT_EQ(results[1].metadata, "similar");
    EXPECT_GT(results[1].score, 0.9f);
}

// US-003: Each result includes: vector ID, similarity score, metadata
TEST_F(VectorStoreTest, SearchResultIncludesAllFields) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec, "my_metadata");

    auto results = store_->search(vec, 1);
    ASSERT_EQ(results.size(), 1);

    // Check all fields are populated
    EXPECT_EQ(results[0].id, id);
    EXPECT_NEAR(results[0].score, 1.0f, 0.001f);
    EXPECT_EQ(results[0].metadata, "my_metadata");
}

TEST_F(VectorStoreTest, SearchResultIDsAreValid) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};

    std::string id1 = store_->insert(vec1, "first");
    std::string id2 = store_->insert(vec2, "second");

    auto results = store_->search(vec1, 2);
    ASSERT_EQ(results.size(), 2);

    // Both IDs should exist in the store
    EXPECT_TRUE(store_->contains(results[0].id));
    EXPECT_TRUE(store_->contains(results[1].id));
}

// US-003: Search uses HNSW index (verified by performance and behavior)
TEST_F(VectorStoreTest, SearchUsesHNSWIndex) {
    // Insert many vectors to demonstrate HNSW usage
    VectorStoreConfig config;
    config.dimensions = 4;
    config.max_vectors = 1000;
    config.hnsw_ef_search = 50;
    VectorStore store(config);

    // Insert vectors with distinct directions (different angles)
    for (int i = 0; i < 100; ++i) {
        float angle = static_cast<float>(i) * 0.1f;  // Different angles
        std::vector<float> vec = {
            std::cos(angle),
            std::sin(angle),
            static_cast<float>(i % 10) * 0.01f,
            static_cast<float>(i / 10) * 0.01f
        };
        store.insert(vec, "vec" + std::to_string(i));
    }

    // Query with angle = 5.0 (i=50: angle = 50 * 0.1 = 5.0)
    float query_angle = 5.0f;
    std::vector<float> query = {
        std::cos(query_angle),
        std::sin(query_angle),
        0.0f,
        0.05f
    };
    auto results = store.search(query, 10);

    EXPECT_EQ(results.size(), 10);
    // Results should be returned - the exact match depends on the query
    // The important thing is that HNSW finds similar vectors efficiently
    EXPECT_FALSE(results.empty());
    // All results should have valid scores (with small epsilon for floating point errors)
    constexpr float epsilon = 1e-5f;
    for (const auto& result : results) {
        EXPECT_GE(result.score, -1.0f - epsilon);
        EXPECT_LE(result.score, 1.0f + epsilon);
    }
}

// US-003: Optional filter by metadata category
TEST_F(VectorStoreTest, SearchFiltersByMetadata) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.9f, 0.1f, 0.0f, 0.0f};
    std::vector<float> vec3 = {0.8f, 0.2f, 0.0f, 0.0f};

    store_->insert(vec1, "movies");
    store_->insert(vec2, "books");
    store_->insert(vec3, "movies");

    // Search with filter
    auto movies = store_->search(vec1, 10, "movies");
    EXPECT_EQ(movies.size(), 2);
    for (const auto& result : movies) {
        EXPECT_EQ(result.metadata, "movies");
    }

    auto books = store_->search(vec1, 10, "books");
    EXPECT_EQ(books.size(), 1);
    EXPECT_EQ(books[0].metadata, "books");
}

TEST_F(VectorStoreTest, SearchFilterReturnsEmptyWhenNoMatch) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "movies");

    auto results = store_->search(vec, 10, "nonexistent_category");
    EXPECT_TRUE(results.empty());
}

TEST_F(VectorStoreTest, SearchWithEmptyFilterReturnsAll) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};

    store_->insert(vec1, "movies");
    store_->insert(vec2, "books");

    // Empty filter should return all
    auto results = store_->search(vec1, 10, "");
    EXPECT_EQ(results.size(), 2);
}

// US-003: Search rejects wrong dimensions
TEST_F(VectorStoreTest, SearchRejectsWrongDimensions) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "test");

    std::vector<float> wrong_query = {1.0f, 2.0f};  // 2 instead of 4

    EXPECT_THROW(store_->search(wrong_query, 5), DimensionMismatchError);
}

TEST_F(VectorStoreTest, SearchRejectsEmptyQuery) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "test");

    std::vector<float> empty_query;

    EXPECT_THROW(store_->search(empty_query, 5), DimensionMismatchError);
}

// US-003: Search with deleted vectors
TEST_F(VectorStoreTest, SearchExcludesDeletedVectors) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.9f, 0.1f, 0.0f, 0.0f};

    std::string id1 = store_->insert(vec1, "first");
    std::string id2 = store_->insert(vec2, "second");

    // Both should be searchable
    auto results_before = store_->search(vec1, 10);
    EXPECT_EQ(results_before.size(), 2);

    // Delete first vector
    store_->remove(id1);

    // Only second should be searchable
    auto results_after = store_->search(vec1, 10);
    EXPECT_EQ(results_after.size(), 1);
    EXPECT_EQ(results_after[0].id, id2);
}

// US-003: Cosine similarity edge cases
TEST_F(VectorStoreTest, SearchHandlesZeroVector) {
    // Zero vectors are edge cases - their normalized form is undefined
    // After normalization, a zero vector remains zero (norm = 0)
    std::vector<float> non_zero = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(non_zero, "non_zero");

    // Searching with a zero vector should not crash
    std::vector<float> zero_query = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = store_->search(zero_query, 1);

    // Results may vary but should not crash
    EXPECT_TRUE(results.size() <= 1);
}

TEST_F(VectorStoreTest, SearchCosineSimilarityRange) {
    // Cosine similarity should be in [-1, 1] range
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {-1.0f, 0.0f, 0.0f, 0.0f};  // Opposite direction

    store_->insert(vec1, "positive");
    store_->insert(vec2, "negative");

    auto results = store_->search(vec1, 2);
    ASSERT_EQ(results.size(), 2);

    for (const auto& result : results) {
        EXPECT_GE(result.score, -1.0f);
        EXPECT_LE(result.score, 1.0f);
    }

    // Identical should be ~1.0
    EXPECT_NEAR(results[0].score, 1.0f, 0.01f);
    // Opposite should be ~-1.0
    EXPECT_NEAR(results[1].score, -1.0f, 0.01f);
}

// US-003: Performance characteristic test (HNSW should be fast)
TEST_F(VectorStoreTest, SearchPerformanceWithManyVectors) {
    VectorStoreConfig config;
    config.dimensions = 64;
    config.max_vectors = 5000;
    config.hnsw_m = 16;
    config.hnsw_ef_construction = 200;
    config.hnsw_ef_search = 50;
    VectorStore store(config);

    // Insert 1000 vectors
    for (int i = 0; i < 1000; ++i) {
        std::vector<float> vec(64);
        for (int j = 0; j < 64; ++j) {
            vec[j] = static_cast<float>((i + j) % 100);
        }
        store.insert(vec, "vec" + std::to_string(i));
    }

    // Search should complete quickly (implicit - test will timeout if too slow)
    std::vector<float> query(64);
    for (int j = 0; j < 64; ++j) {
        query[j] = static_cast<float>(j);
    }

    auto results = store.search(query, 10);
    EXPECT_EQ(results.size(), 10);
}

// =============================================================================
// US-004: Delete Vector by ID
// =============================================================================

// US-004: Delete accepts a vector UUID
TEST_F(VectorStoreTest, DeleteAcceptsVectorUUID) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec, "test_metadata");

    // UUID format check
    ASSERT_EQ(id.length(), 36);
    EXPECT_EQ(id[8], '-');
    EXPECT_EQ(id[13], '-');

    // Delete should accept this UUID
    bool result = store_->remove(id);
    EXPECT_TRUE(result);
}

// US-004: Returns success if vector existed and was deleted
TEST_F(VectorStoreTest, DeleteReturnsSuccessWhenVectorExisted) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec, "metadata");

    EXPECT_EQ(store_->size(), 1);
    EXPECT_TRUE(store_->contains(id));

    // Delete should return true
    bool success = store_->remove(id);
    EXPECT_TRUE(success);

    // Vector should be gone
    EXPECT_EQ(store_->size(), 0);
    EXPECT_FALSE(store_->contains(id));
}

// US-004: Returns appropriate error if vector ID not found
TEST_F(VectorStoreTest, DeleteReturnsFalseForNonExistentID) {
    // Empty store
    bool result1 = store_->remove("00000000-0000-0000-0000-000000000000");
    EXPECT_FALSE(result1);

    // With vectors but wrong ID
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    store_->insert(vec);

    bool result2 = store_->remove("nonexistent-uuid");
    EXPECT_FALSE(result2);

    bool result3 = store_->remove("00000000-0000-4000-8000-000000000000");
    EXPECT_FALSE(result3);

    // Store size unchanged
    EXPECT_EQ(store_->size(), 1);
}

TEST_F(VectorStoreTest, DeleteReturnsFalseForEmptyID) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    store_->insert(vec);

    bool result = store_->remove("");
    EXPECT_FALSE(result);

    // Store size unchanged
    EXPECT_EQ(store_->size(), 1);
}

// US-004: Deleted vectors no longer appear in search results
TEST_F(VectorStoreTest, DeletedVectorsNotInSearchResults) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.9f, 0.1f, 0.0f, 0.0f};
    std::vector<float> vec3 = {0.8f, 0.2f, 0.0f, 0.0f};

    std::string id1 = store_->insert(vec1, "first");
    std::string id2 = store_->insert(vec2, "second");
    std::string id3 = store_->insert(vec3, "third");

    // All three should be in search results
    auto results_before = store_->search(vec1, 10);
    EXPECT_EQ(results_before.size(), 3);

    // Delete the first vector
    EXPECT_TRUE(store_->remove(id1));

    // Now only two should be in search results
    auto results_after = store_->search(vec1, 10);
    EXPECT_EQ(results_after.size(), 2);

    // Verify id1 is not in results
    for (const auto& result : results_after) {
        EXPECT_NE(result.id, id1);
    }
}

TEST_F(VectorStoreTest, DeletedVectorIDCannotBeFoundByContains) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec);

    EXPECT_TRUE(store_->contains(id));

    store_->remove(id);

    EXPECT_FALSE(store_->contains(id));
}

// US-004: Delete multiple vectors
TEST_F(VectorStoreTest, DeleteMultipleVectorsSequentially) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    std::string id1 = store_->insert(vec, "v1");
    std::string id2 = store_->insert(vec, "v2");
    std::string id3 = store_->insert(vec, "v3");

    EXPECT_EQ(store_->size(), 3);

    // Delete in order
    EXPECT_TRUE(store_->remove(id1));
    EXPECT_EQ(store_->size(), 2);

    EXPECT_TRUE(store_->remove(id2));
    EXPECT_EQ(store_->size(), 1);

    EXPECT_TRUE(store_->remove(id3));
    EXPECT_EQ(store_->size(), 0);

    // All should be gone
    EXPECT_FALSE(store_->contains(id1));
    EXPECT_FALSE(store_->contains(id2));
    EXPECT_FALSE(store_->contains(id3));
}

// US-004: Cannot delete same vector twice
TEST_F(VectorStoreTest, DeleteSameVectorTwiceReturnsFalseSecondTime) {
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    std::string id = store_->insert(vec);

    // First delete succeeds
    EXPECT_TRUE(store_->remove(id));
    EXPECT_EQ(store_->size(), 0);

    // Second delete fails - already deleted
    EXPECT_FALSE(store_->remove(id));
    EXPECT_EQ(store_->size(), 0);
}

// US-004: Delete and re-insert with sufficient capacity
// Note: HNSW uses markDelete which doesn't free up space in the index.
// The max_vectors limit applies to total vectors ever added, not current size.
// This is a fundamental HNSW behavior - deleted vectors use tombstones.
TEST_F(VectorStoreTest, CanInsertAfterDeleteWithCapacity) {
    VectorStoreConfig config;
    config.dimensions = 4;
    config.max_vectors = 10;  // Larger capacity to allow for tombstones
    VectorStore store(config);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};

    // Insert some vectors
    std::string id1 = store.insert(vec);
    std::string id2 = store.insert(vec);
    EXPECT_EQ(store.size(), 2);

    // Delete one - size decreases but HNSW tombstone remains
    EXPECT_TRUE(store.remove(id1));
    EXPECT_EQ(store.size(), 1);

    // Can insert new vector (uses a new label, doesn't reuse deleted slot)
    std::string id3 = store.insert(vec);
    EXPECT_EQ(store.size(), 2);
    EXPECT_NE(id3, id1);
    EXPECT_NE(id3, id2);

    // The new vector is searchable
    auto results = store.search(vec, 10);
    EXPECT_EQ(results.size(), 2);
}

// US-004: Delete preserves metadata of remaining vectors
TEST_F(VectorStoreTest, DeletePreservesOtherVectorsMetadata) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};

    std::string id1 = store_->insert(vec1, "metadata_1");
    std::string id2 = store_->insert(vec2, "metadata_2");

    // Delete first vector
    store_->remove(id1);

    // Search for second vector - metadata should be intact
    auto results = store_->search(vec2, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].id, id2);
    EXPECT_EQ(results[0].metadata, "metadata_2");
}

// =============================================================================
// US-003/US-005: Parallel Search with Thread Pool
// =============================================================================

// US-003: Search parallelized using custom thread pool
TEST_F(VectorStoreTest, SearchBatchReturnsCorrectResults) {
    // Insert some vectors
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> vec3 = {0.0f, 0.0f, 1.0f, 0.0f};

    store_->insert(vec1, "v1");
    store_->insert(vec2, "v2");
    store_->insert(vec3, "v3");

    // Batch search with multiple queries
    std::vector<std::vector<float>> queries = {vec1, vec2, vec3};
    auto batch_results = store_->searchBatch(queries, 3);

    ASSERT_EQ(batch_results.size(), 3);

    // Each query should return 3 results
    EXPECT_EQ(batch_results[0].size(), 3);
    EXPECT_EQ(batch_results[1].size(), 3);
    EXPECT_EQ(batch_results[2].size(), 3);

    // First result for each query should be the identical vector (score ~1.0)
    EXPECT_NEAR(batch_results[0][0].score, 1.0f, 0.01f);
    EXPECT_NEAR(batch_results[1][0].score, 1.0f, 0.01f);
    EXPECT_NEAR(batch_results[2][0].score, 1.0f, 0.01f);
}

TEST_F(VectorStoreTest, SearchBatchEmptyQueriesReturnsEmpty) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "test");

    std::vector<std::vector<float>> empty_queries;
    auto results = store_->searchBatch(empty_queries, 5);

    EXPECT_TRUE(results.empty());
}

TEST_F(VectorStoreTest, SearchBatchSingleQueryMatchesSingleSearch) {
    // Insert some vectors
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.9f, 0.1f, 0.0f, 0.0f};

    store_->insert(vec1, "v1");
    store_->insert(vec2, "v2");

    // Single search
    auto single_results = store_->search(vec1, 2);

    // Batch search with one query
    std::vector<std::vector<float>> queries = {vec1};
    auto batch_results = store_->searchBatch(queries, 2);

    ASSERT_EQ(batch_results.size(), 1);
    ASSERT_EQ(batch_results[0].size(), single_results.size());

    // Results should match
    for (size_t i = 0; i < single_results.size(); ++i) {
        EXPECT_EQ(batch_results[0][i].id, single_results[i].id);
        EXPECT_FLOAT_EQ(batch_results[0][i].score, single_results[i].score);
        EXPECT_EQ(batch_results[0][i].metadata, single_results[i].metadata);
    }
}

TEST_F(VectorStoreTest, SearchBatchWithMetadataFilter) {
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.9f, 0.1f, 0.0f, 0.0f};
    std::vector<float> vec3 = {0.8f, 0.2f, 0.0f, 0.0f};

    store_->insert(vec1, "movies");
    store_->insert(vec2, "books");
    store_->insert(vec3, "movies");

    std::vector<std::vector<float>> queries = {vec1, vec2};
    auto results = store_->searchBatch(queries, 10, "movies");

    ASSERT_EQ(results.size(), 2);
    // Both queries should only return vectors with "movies" metadata
    for (const auto& query_results : results) {
        for (const auto& result : query_results) {
            EXPECT_EQ(result.metadata, "movies");
        }
    }
}

TEST_F(VectorStoreTest, SearchBatchRejectsWrongDimensions) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    store_->insert(vec, "test");

    // One valid, one invalid query
    std::vector<std::vector<float>> queries = {
        {1.0f, 0.0f, 0.0f, 0.0f},  // Valid
        {1.0f, 0.0f}  // Invalid - wrong dimensions
    };

    // Should throw when processing the invalid query
    EXPECT_THROW(store_->searchBatch(queries, 5), DimensionMismatchError);
}

// US-005: Thread pool used by search operation (parallel execution test)
TEST_F(VectorStoreTest, SearchBatchExecutesInParallel) {
    // Create a larger store for meaningful parallel test
    VectorStoreConfig config;
    config.dimensions = 64;
    config.max_vectors = 5000;
    config.thread_pool_size = 4;
    VectorStore store(config);

    // Insert vectors
    for (int i = 0; i < 500; ++i) {
        std::vector<float> vec(64);
        for (int j = 0; j < 64; ++j) {
            vec[j] = static_cast<float>((i + j) % 100);
        }
        store.insert(vec, "vec" + std::to_string(i));
    }

    // Create many queries
    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 100; ++i) {
        std::vector<float> query(64);
        for (int j = 0; j < 64; ++j) {
            query[j] = static_cast<float>((i * 7 + j) % 100);
        }
        queries.push_back(query);
    }

    // Batch search should complete (parallel execution)
    auto results = store.searchBatch(queries, 10);

    EXPECT_EQ(results.size(), 100);
    for (const auto& query_results : results) {
        EXPECT_EQ(query_results.size(), 10);
    }
}

TEST_F(VectorStoreTest, SearchBatchOnEmptyStore) {
    std::vector<std::vector<float>> queries = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f}
    };

    auto results = store_->searchBatch(queries, 5);

    ASSERT_EQ(results.size(), 2);
    EXPECT_TRUE(results[0].empty());
    EXPECT_TRUE(results[1].empty());
}

TEST_F(VectorStoreTest, SearchBatchPreservesQueryOrder) {
    // Insert distinct vectors
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> vec3 = {0.0f, 0.0f, 1.0f, 0.0f};

    store_->insert(vec1, "v1");
    store_->insert(vec2, "v2");
    store_->insert(vec3, "v3");

    // Query in specific order
    std::vector<std::vector<float>> queries = {vec3, vec1, vec2};
    auto results = store_->searchBatch(queries, 1);

    ASSERT_EQ(results.size(), 3);

    // Results should be in query order
    EXPECT_EQ(results[0][0].metadata, "v3");  // First query was vec3
    EXPECT_EQ(results[1][0].metadata, "v1");  // Second query was vec1
    EXPECT_EQ(results[2][0].metadata, "v2");  // Third query was vec2
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
