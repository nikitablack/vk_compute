#include <gpu/impl/get_physical_device_properties.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto get_physical_device_properties(VkPhysicalDevice device) noexcept -> GetPhysicalDevicePropertiesResult {
    VkPhysicalDeviceSubgroupProperties subgroupProperties = vku::InitStructHelper{};
    VkPhysicalDeviceProperties2 physicalDeviceProperties2 = vku::InitStructHelper{&subgroupProperties};

    vkGetPhysicalDeviceProperties2(device, &physicalDeviceProperties2);

    GetPhysicalDevicePropertiesResult result{};
    result.properties = physicalDeviceProperties2;
    result.subgroupProperties = subgroupProperties;

    return result;
}

}  // namespace gpu::impl
