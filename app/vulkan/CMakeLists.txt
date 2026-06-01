option(VK_ENABLE_RENDERDOC_DEBUG "Enable Renderdoc debug" ON)

if(VK_ENABLE_RENDERDOC_DEBUG)
    add_compile_definitions(VK_ENABLE_RENDERDOC_DEBUG)
endif()

add_subdirectory(add)
add_subdirectory(minmax)
