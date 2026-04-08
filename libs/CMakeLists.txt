include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileVulkanShaders.cmake)

add_subdirectory(gpu)
add_subdirectory(warnings)
add_subdirectory(utils)
