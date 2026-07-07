cmake_minimum_required(VERSION 3.25)

add_subdirectory(shaders)
add_subdirectory(utils)

if(BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()

add_library(kernel STATIC "")

add_library(libs::kernel
    ALIAS
        kernel
)

target_sources(kernel
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/Add.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/MinMaxV1.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel/MinMaxV2.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/Add.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/MinMaxV1.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/MinMaxV2.cpp
)

target_include_directories(kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(kernel
    PUBLIC
        libs::gpu
        libs::kernel::utils
        spdlog::spdlog
    PRIVATE
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
)

set_target_properties(kernel
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
