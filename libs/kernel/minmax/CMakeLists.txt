cmake_minimum_required(VERSION 3.25)

add_library(kernel_minmax STATIC "")

add_library(libs::kernel::minmax
    ALIAS
        kernel_minmax
)

target_sources(kernel_minmax
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/minmax/minmax_atomic.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/minmax/minmax_reduction.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/minmax/Result.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/minmax/minmax_atomic.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/minmax/minmax_reduction.cpp
)

target_include_directories(kernel_minmax PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(kernel_minmax
    PUBLIC
        libs::gpu
        libs::kernel::utils
    PRIVATE
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
        spdlog::spdlog
)

set_target_properties(kernel_minmax
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
