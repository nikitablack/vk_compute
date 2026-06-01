cmake_minimum_required(VERSION 3.25)

set(kernel_shaders "kernel_shaders")
cmrc_add_resource_library(${kernel_shaders} "")

add_library(libs::kernel::shaders
    ALIAS
        kernel_shaders
)

addCompileShadersCommand(TARGET_NAME ${kernel_shaders}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders
    SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/add.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/minmax_uint_atomic_global.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/minmax_uint_init.comp
)
