include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileVulkanShaders.cmake)

add_subdirectory(gpu)
add_subdirectory(kernel)
add_subdirectory(warnings)
add_subdirectory(utils)

if(BUILD_CUDA)
    add_subdirectory(cuda)
endif()

