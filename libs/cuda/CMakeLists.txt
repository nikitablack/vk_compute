cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/cuda")

project(cuda LANGUAGES CXX CUDA)

add_library(cuda STATIC "")

add_library(libs::cuda
    ALIAS
        cuda
)

target_sources(cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/utils/check_error.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/utils/constants.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/add.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add_chunked.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add_chunked.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add_data.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add_data.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add_v4.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add_v4.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/impl/add.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/add.cpp
)

target_include_directories(cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(cuda PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86"
)

target_link_libraries(cuda
    PRIVATE
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
        spdlog::spdlog
)

set(CUDA_FLAGS --Werror all-warnings)
target_compile_options(cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS};>")
