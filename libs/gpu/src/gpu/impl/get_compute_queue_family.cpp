#include <spdlog/spdlog.h>

#include <gpu/impl/get_compute_queue_family.hpp>
#include <vector>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto get_compute_queue_family(VkPhysicalDevice device, uint32_t requiredQueueCount) noexcept
    -> std::expected<uint32_t, std::string> {
    spdlog::trace("getting compute queue family");

    uint32_t queueFamilyCount{0};
    vkGetPhysicalDeviceQueueFamilyProperties2(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties2> queueFamilies(queueFamilyCount);
    for (size_t i{0}; i < queueFamilies.size(); ++i) {
        queueFamilies[i] = vku::InitStructHelper{};
    }

    vkGetPhysicalDeviceQueueFamilyProperties2(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i{0}; i < queueFamilies.size(); ++i) {
        auto const family{queueFamilies[i]};

        if ((family.queueFamilyProperties.queueCount >= requiredQueueCount) &&
            (family.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            return i;
        }
    }

    return std::unexpected{"failed to find graphics queue"};
}

}  // namespace gpu::impl
