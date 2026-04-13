cmake_minimum_required(VERSION 3.25)

add_library(kernel STATIC "")

add_library(libs::kernel
    ALIAS
        kernel
)

option(VK_ENABLE_RENDERDOC_DEBUG "Enable Renderdoc debug" ON)

if(VK_ENABLE_RENDERDOC_DEBUG)
    target_compile_definitions(kernel
        PRIVATE
            VK_ENABLE_RENDERDOC_DEBUG
    )
endif()


target_sources(kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/utils/benchmark_with_percentiles.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/utils/create_pipeline.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/Add.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/utils/create_pipeline.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/Add.cpp
)

target_include_directories(kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(kernel_shaders "kernel_shaders")
cmrc_add_resource_library(${kernel_shaders} "")

target_link_libraries(kernel
    PUBLIC
        libs::gpu
    PRIVATE
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
        ${kernel_shaders}
        third_party::renderdoc_api
        dl
)

set_target_properties(kernel
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

addCompileShadersCommand(TARGET_NAME ${kernel_shaders}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders
    SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/add_v.comp
)
