#pragma once

#include <vulkan/vulkan.h>

namespace gpu {

struct VulkanQueue {
public:
    uint32_t queueFamily;
    VkQueue queue;
};

}  // namespace gpu
