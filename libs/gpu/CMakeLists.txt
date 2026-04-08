cmake_minimum_required(VERSION 3.25)

find_package(Vulkan REQUIRED)

add_library(gpu STATIC "")

add_library(libs::gpu
    ALIAS
        gpu
)

option(VK_ENABLE_VULKAN_DEBUG_UTILS "Enable Vulkan debug utils" ON)

target_sources(gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/GpuManager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/VulkanQueue.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Features.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/RequiredFeatures.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/RequiredFeatures.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan12Features.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan12Features.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan13Features.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan13Features.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_instance_version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_instance_version.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_physical_device_support.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_physical_device_support.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_required_instance_extensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_required_instance_extensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_instance.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_instance.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_compute_queue_family.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_compute_queue_family.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_physical_device_properties.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_physical_device_properties.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_supported_physical_devices.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_supported_physical_devices.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredApiVersion.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredDeviceExtensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredDeviceExtensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredInstanceExtensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredInstanceExtensions.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/GpuManager.cpp
)

target_include_directories(gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(shaders_lib "shaders_lib")
cmrc_add_resource_library(${shaders_lib} "")

target_link_libraries(gpu
    PUBLIC
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
    PRIVATE
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
        ${shaders_lib}
)

set_target_properties(gpu
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

if(VK_ENABLE_VULKAN_DEBUG_UTILS)
    target_compile_definitions(gpu
        PRIVATE
            ENABLE_VULKAN_DEBUG_UTILS
    )
endif()

# addCompileShadersCommand(TARGET_NAME ${graphics_shaders_lib}
#     OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders
#     SHADERS
#     ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/fullscreen_triangle.frag
#     ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/fullscreen_triangle.vert
# )
