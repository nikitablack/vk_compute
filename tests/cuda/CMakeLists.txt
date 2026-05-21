cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

enable_language(CXX CUDA)

add_executable(cuda_tests)

set_target_properties(cuda_tests
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_sources(cuda_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_add_chunked.cpp
)

target_link_libraries(cuda_tests
    PRIVATE
        Catch2::Catch2WithMain
        libs::cuda
        libs::warnings_as_errors
)

catch_discover_tests(cuda_tests
    TEST_PREFIX
        "cuda_tests::"
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
)
