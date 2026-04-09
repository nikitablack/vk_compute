#pragma once

#include <vulkan/vulkan.h>

namespace gpu::impl {

auto get_queue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex) noexcept -> VkQueue;

}  // namespace gpu::impl
