cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(template LANGUAGES CXX)

include(FetchContent)

add_subdirectory(third_party)
add_subdirectory(libs)
add_subdirectory(app)
