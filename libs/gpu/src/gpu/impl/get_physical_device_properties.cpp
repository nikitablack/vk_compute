#include <gpu/impl/get_physical_device_properties.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto get_physical_device_properties(VkPhysicalDevice device) noexcept -> VkPhysicalDeviceProperties2 {
    VkPhysicalDeviceProperties2 physicalDeviceProperties2 = vku::InitStructHelper{};

    vkGetPhysicalDeviceProperties2(device, &physicalDeviceProperties2);

    return physicalDeviceProperties2;
}

}  // namespace gpu::impl
