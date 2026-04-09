#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu {

struct VulkanFunctions {
public:
    [[nodiscard]] static auto initialize(VkInstance instance) noexcept -> std::expected<void, std::string>;

public:
    // debug utils
    static PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;
};

}  // namespace gpu
