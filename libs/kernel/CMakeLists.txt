cmake_minimum_required(VERSION 3.25)

add_subdirectory(add)
add_subdirectory(minmax)
add_subdirectory(shaders)
add_subdirectory(utils)

if(BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()
