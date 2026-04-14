cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

enable_language(CXX)

add_executable(tests)

set_target_properties(tests
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_sources(tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/test_add.cpp
)

target_link_libraries(tests
    PRIVATE
        Catch2::Catch2WithMain
        libs::kernel
        libs::warnings_as_errors
)

catch_discover_tests(tests
    TEST_PREFIX
        "tests::"
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
)
