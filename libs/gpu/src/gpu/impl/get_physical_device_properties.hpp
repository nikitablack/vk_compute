#pragma once

#include <vulkan/vulkan.h>

namespace gpu::impl {

[[nodiscard]] auto get_physical_device_properties(VkPhysicalDevice device) noexcept -> VkPhysicalDeviceProperties2;

}  // namespace gpu::impl
