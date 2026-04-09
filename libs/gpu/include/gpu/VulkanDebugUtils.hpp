#pragma once

#include <vulkan/vulkan.h>

#include <gpu/VulkanFunctions.hpp>
#include <string>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu {

class VulkanDebugUtils {
public:
    VulkanDebugUtils() = default;

    auto initialize(VkDevice device) noexcept -> void { m_device = device; }

    template <typename T>
    auto setName([[maybe_unused]] T object, [[maybe_unused]] std::string const& name) const noexcept -> void {
#ifdef ENABLE_VULKAN_DEBUG_UTILS
        static_assert(sizeof(T) == 8, "unsupported type");

        if (!m_device) {
            return;
        }

        VkDebugUtilsObjectNameInfoEXT info = vku::InitStructHelper{};
        info.objectType = vku::GetObjectType<T>();
        info.objectHandle = reinterpret_cast<uint64_t>(object);
        info.pObjectName = name.c_str();

        VulkanFunctions::vkSetDebugUtilsObjectNameEXT(m_device, &info);
#endif
    }

private:
    VkDevice m_device{VK_NULL_HANDLE};
};

}  // namespace gpu
