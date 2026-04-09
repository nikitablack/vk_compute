cmake_minimum_required(VERSION 3.25)

add_library(matrix_add STATIC "")

add_library(libs::matrix_add
    ALIAS
        matrix_add
)

target_sources(matrix_add
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/matrix_add/MatrixAdd.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_add/impl/create_matrix_add_pipeline.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_add/impl/create_matrix_add_pipeline.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_add/MatrixAdd.cpp
)

target_include_directories(matrix_add PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(matrix_add PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(matrix_add_shaders "matrix_add_shaders")
cmrc_add_resource_library(${matrix_add_shaders} "")

target_link_libraries(matrix_add
    PUBLIC
        libs::gpu
    PRIVATE
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
        ${matrix_add_shaders}
)

set_target_properties(matrix_add
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

addCompileShadersCommand(TARGET_NAME ${matrix_add_shaders}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders
    SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matrix_add.comp
)
