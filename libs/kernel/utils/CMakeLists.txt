cmake_minimum_required(VERSION 3.25)

add_library(kernel_utils STATIC "")

add_library(libs::kernel::utils
    ALIAS
        kernel_utils
)

target_sources(kernel_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/utils/pipeline_helper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/utils/workgroup_size.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/utils/pipeline_helper.cpp
)

target_include_directories(kernel_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(kernel_utils
    PUBLIC
        libs::gpu
    PRIVATE
        fmt::fmt
        libs::kernel::shaders
        libs::utils
        libs::warnings_as_errors
        spdlog::spdlog
)

set_target_properties(kernel_utils
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
