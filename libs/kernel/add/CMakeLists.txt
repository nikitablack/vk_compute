cmake_minimum_required(VERSION 3.25)

add_library(kernel_add STATIC "")

add_library(libs::kernel::add
    ALIAS
        kernel_add
)

target_sources(kernel_add
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/add/add.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/add/add.cpp
)

target_include_directories(kernel_add PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(kernel_add
    PUBLIC
        libs::gpu
    PRIVATE
        fmt::fmt
        libs::kernel::utils
        libs::utils
        libs::warnings_as_errors
        spdlog::spdlog
)

set_target_properties(kernel_add
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
