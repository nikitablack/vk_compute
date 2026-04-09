#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::utils {

[[nodiscard]] auto submit(VkCommandBuffer commandBuffer, VkQueue queue, VkFence fence = VK_NULL_HANDLE) noexcept
    -> std::expected<void, std::string>;

}  // namespace gpu::utils
