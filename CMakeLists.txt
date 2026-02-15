cmake_minimum_required(VERSION 3.16)
project(VectorPP VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use FetchContent for dependencies
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nlohmann/json (header-only)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# hnswlib (header-only)
FetchContent_Declare(
    hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(hnswlib)

# gRPC and Protobuf (optional for now)
find_package(Protobuf CONFIG QUIET)
find_package(gRPC CONFIG QUIET)

if(gRPC_FOUND AND Protobuf_FOUND)
    set(GRPC_ENABLED TRUE)
    message(STATUS "gRPC and Protobuf found - building full server")
else()
    set(GRPC_ENABLED FALSE)
    message(STATUS "gRPC/Protobuf not found - building core components only")
endif()

# gRPC-dependent configuration
if(GRPC_ENABLED)
    # Get the protoc and grpc_cpp_plugin paths
    find_program(_PROTOBUF_PROTOC protoc)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

    # Proto file location
    set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/vectordb.proto")
    set(PROTO_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto")

    # Create output directory
    file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

    # Generate protobuf and gRPC source files
    set(PROTO_SRCS "${PROTO_OUT_DIR}/vectordb.pb.cc")
    set(PROTO_HDRS "${PROTO_OUT_DIR}/vectordb.pb.h")
    set(GRPC_SRCS "${PROTO_OUT_DIR}/vectordb.grpc.pb.cc")
    set(GRPC_HDRS "${PROTO_OUT_DIR}/vectordb.grpc.pb.h")

    add_custom_command(
        OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out="${PROTO_OUT_DIR}"
             --cpp_out="${PROTO_OUT_DIR}"
             -I "${CMAKE_CURRENT_SOURCE_DIR}/proto"
             --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
             "${PROTO_FILE}"
        DEPENDS "${PROTO_FILE}"
        COMMENT "Generating gRPC/Protobuf source files"
    )
endif()

# Core library
add_library(vectorpp_core
    src/core/vector_store.cpp
    src/utils/uuid.cpp
    src/config/config_loader.cpp
    src/concurrency/thread_pool.cpp
)

target_include_directories(vectorpp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${hnswlib_SOURCE_DIR}
)

target_link_libraries(vectorpp_core PUBLIC
    nlohmann_json::nlohmann_json
)

# gRPC-dependent targets
if(GRPC_ENABLED)
    # Proto library (generated sources)
    add_library(vectorpp_proto
        ${PROTO_SRCS}
        ${GRPC_SRCS}
    )

    target_include_directories(vectorpp_proto PUBLIC
        ${PROTO_OUT_DIR}
    )

    target_link_libraries(vectorpp_proto PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
    )

    # Disable warnings for generated code
    if(MSVC)
        target_compile_options(vectorpp_proto PRIVATE /W0)
    else()
        target_compile_options(vectorpp_proto PRIVATE -w)
    endif()

    # Server library
    add_library(vectorpp_server
        src/server/service_impl.cpp
        src/server/grpc_server.cpp
    )

    target_include_directories(vectorpp_server PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_OUT_DIR}
    )

    target_link_libraries(vectorpp_server PUBLIC
        vectorpp_core
        vectorpp_proto
    )

    # Server executable
    add_executable(vectorpp_server_bin
        src/main.cpp
    )

    target_link_libraries(vectorpp_server_bin PRIVATE
        vectorpp_server
    )

    set_target_properties(vectorpp_server_bin PROPERTIES
        OUTPUT_NAME "vectorpp_server"
    )
endif()

# Unit tests
enable_testing()

add_executable(test_vector_store
    tests/unit/test_vector_store.cpp
)

target_include_directories(test_vector_store PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${hnswlib_SOURCE_DIR}
)

target_link_libraries(test_vector_store PRIVATE
    vectorpp_core
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_vector_store)

add_executable(test_config_loader
    tests/unit/test_config_loader.cpp
)

target_include_directories(test_config_loader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${hnswlib_SOURCE_DIR}
)

target_link_libraries(test_config_loader PRIVATE
    vectorpp_core
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(test_config_loader)

add_executable(test_thread_pool
    tests/unit/test_thread_pool.cpp
)

target_include_directories(test_thread_pool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(test_thread_pool PRIVATE
    vectorpp_core
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(test_thread_pool)

# Benchmark executable for search performance comparison
add_executable(benchmark_search
    tests/benchmark/benchmark_search.cpp
)

target_include_directories(benchmark_search PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${hnswlib_SOURCE_DIR}
)

target_link_libraries(benchmark_search PRIVATE
    vectorpp_core
    GTest::gtest
    GTest::gtest_main
)

gtest_discover_tests(benchmark_search)

# Integration tests for gRPC server (only if gRPC is available)
if(GRPC_ENABLED)
    add_executable(test_grpc_server
        tests/integration/test_grpc_server.cpp
    )

    target_include_directories(test_grpc_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_OUT_DIR}
    )

    target_link_libraries(test_grpc_server PRIVATE
        vectorpp_server
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(test_grpc_server)
endif()
