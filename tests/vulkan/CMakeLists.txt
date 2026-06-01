cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

enable_language(CXX)

add_executable(vk_tests)

set_target_properties(vk_tests
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_sources(vk_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_add.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_minmax_uint_atomic.cpp
)

target_link_libraries(vk_tests
    PRIVATE
        Catch2::Catch2WithMain
        libs::kernel
        libs::warnings_as_errors
)

catch_discover_tests(vk_tests
    TEST_PREFIX
        "vk_tests::"
    WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
)
