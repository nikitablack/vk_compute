cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

add_library(utils INTERFACE)

add_library(libs::utils
    ALIAS
        utils
)

target_sources(utils
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/benchmark_with_percentiles.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/to_span.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/try_expected.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/try_optional.hpp
)

target_include_directories(utils
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(utils PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
