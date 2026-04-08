#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto check_physical_device_support(VkPhysicalDevice device) noexcept -> std::expected<void, std::string>;

}  // namespace gpu::impl
