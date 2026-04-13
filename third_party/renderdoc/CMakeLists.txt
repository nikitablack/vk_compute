cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

add_library(renderdoc_api INTERFACE)

add_library(third_party::renderdoc_api
    ALIAS
        renderdoc_api
)

target_sources(renderdoc_api
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/renderdoc_app.h
)

target_include_directories(renderdoc_api
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
