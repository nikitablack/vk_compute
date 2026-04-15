cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(compute LANGUAGES CXX)

option(BUILD_TESTS "Whether or not to build the tests" OFF)
option(BUILD_CUDA "Whether or not to build Cuda code" ON)

include(FetchContent)

add_subdirectory(third_party)
add_subdirectory(libs)
add_subdirectory(app)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(third_party/catch SYSTEM)
    add_subdirectory(tests)
endif()
