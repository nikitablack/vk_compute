#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto create_device(VkPhysicalDevice physicalDevice,  //
                                 uint32_t graphicsQueueFamily,  //
                                 uint32_t queueCount  //
                                 ) noexcept -> std::expected<VkDevice, std::string>;

}  // namespace gpu::impl
