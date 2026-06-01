cmake_minimum_required(VERSION 3.25)

add_library(kernel STATIC "")

add_library(libs::kernel
    ALIAS
        kernel
)

target_sources(kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/utils/pipeline_helper.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/add.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/minmax.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/utils/pipeline_helper.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/add.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/minmax.cpp
)

target_include_directories(kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(kernel_shaders "kernel_shaders")
cmrc_add_resource_library(${kernel_shaders} "")

target_link_libraries(kernel
    PUBLIC
        libs::gpu
    PRIVATE
        dl # for renderdoc
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
        spdlog::spdlog
        ${kernel_shaders}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/minmax_uint_atomic.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/minmax_uint_init.comp
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(python_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/bindings.cpp
)

install(TARGETS
            python_kernel
        LIBRARY DESTINATION
            .
)

target_link_libraries(python_kernel
    PRIVATE
        libs::kernel
        libs::utils
)

set_target_properties(python_kernel
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
