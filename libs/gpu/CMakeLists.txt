cmake_minimum_required(VERSION 3.25)

find_package(Vulkan REQUIRED)

add_library(gpu STATIC "")

add_library(libs::gpu
    ALIAS
        gpu
)

option(VK_ENABLE_VULKAN_DEBUG_UTILS "Enable Vulkan debug utils" ON)

if(VK_ENABLE_VULKAN_DEBUG_UTILS)
    target_compile_definitions(gpu
        PRIVATE
            ENABLE_VULKAN_DEBUG_UTILS
    )
endif()

target_sources(gpu
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/utils/barrier_helper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/utils/create_shader_module.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/utils/get_push_constant_data.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/utils/init_helper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/utils/read_helper.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/utils/submit.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/CommandManager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/DeviceBuffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/GpuManager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/HostVisibleBuffer.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/StorageDescriptorSetManager.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/VulkanDebugUtils.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/VulkanFunctions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu/VulkanQueue.hpp

    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Features.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/RequiredFeatures.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/RequiredFeatures.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan12Features.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan12Features.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan13Features.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan13Features.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan14Features.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/features/Vulkan14Features.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/allocate_descriptor_set.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/allocate_descriptor_set.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_instance_version.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_instance_version.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_physical_device_support.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_physical_device_support.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_required_instance_extensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/check_required_instance_extensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_allocator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_allocator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_descriptor_pool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_descriptor_pool.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_descriptor_set_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_descriptor_set_layout.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_device.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_device.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_instance.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_instance.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_pipeline_layout.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/create_pipeline_layout.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_compute_queue_family.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_compute_queue_family.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_physical_device_properties.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_physical_device_properties.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_queue.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_queue.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_supported_physical_devices.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/get_supported_physical_devices.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredApiVersion.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredDeviceExtensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredDeviceExtensions.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredInstanceExtensions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/impl/RequiredInstanceExtensions.hpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/utils/barrier_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/utils/create_shader_module.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/utils/init_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/utils/read_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/utils/submit.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/CommandManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/DeviceBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/GpuManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/HostVisibleBuffer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/StorageDescriptorSetManager.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/VulkanFunctions.cpp
)

target_include_directories(gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(gpu
    PUBLIC
        Vulkan::Vulkan
        GPUOpen::VulkanMemoryAllocator
    PRIVATE
        fmt::fmt
        libs::utils
        libs::warnings_as_errors
        spdlog::spdlog
)

set_target_properties(gpu
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
