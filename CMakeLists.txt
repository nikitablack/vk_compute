cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(template LANGUAGES CXX)

include(FetchContent)

add_subdirectory(third_party)
add_subdirectory(libs)
add_subdirectory(app)

option(BUILD_TESTS "Whether or not to build the tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(third_party/catch SYSTEM)
    add_subdirectory(tests)
endif()