cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(app LANGUAGES CXX)

add_executable(app)

option(VK_ENABLE_RENDERDOC_DEBUG "Enable Renderdoc debug" ON)

if(VK_ENABLE_RENDERDOC_DEBUG)
    target_compile_definitions(app
        PRIVATE
            VK_ENABLE_RENDERDOC_DEBUG
    )
endif()

target_sources(app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(app
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(app
    PRIVATE
        fmt::fmt
        libs::gpu
        libs::kernel
        libs::utils
        libs::warnings_as_errors
        third_party::renderdoc_api
)
