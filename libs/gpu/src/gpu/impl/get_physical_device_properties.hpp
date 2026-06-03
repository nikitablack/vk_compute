#pragma once

#include <vulkan/vulkan.h>

namespace gpu::impl {

struct GetPhysicalDevicePropertiesResult {
    VkPhysicalDeviceProperties2 properties{};
    VkPhysicalDeviceSubgroupProperties subgroupProperties{};
};

[[nodiscard]] auto get_physical_device_properties(VkPhysicalDevice device  //
                                                  ) noexcept -> GetPhysicalDevicePropertiesResult;

}  // namespace gpu::impl
