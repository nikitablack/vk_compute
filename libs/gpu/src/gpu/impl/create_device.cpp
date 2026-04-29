#include <spdlog/spdlog.h>

#include <algorithm>
#include <gpu/impl/RequiredDeviceExtensions.hpp>
#include <gpu/impl/create_device.hpp>
#include <gpu/impl/features/RequiredFeatures.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto create_device(VkPhysicalDevice physicalDevice,  //
                   uint32_t queueFamily,  //
                   uint32_t queueCount  //
                   ) noexcept -> std::expected<VkDevice, std::string> {
    spdlog::trace("creating device");

    // priorities
    std::vector<float> queuePriorities(queueCount);
    std::fill(queuePriorities.begin(), queuePriorities.end(), 1.0f);

    VkDeviceQueueCreateInfo queueCreateInfo = vku::InitStructHelper{};
    queueCreateInfo.flags = 0;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = queueCount;
    queueCreateInfo.pQueuePriorities = queuePriorities.data();

    auto const requiredExtensions{RequiredDeviceExtensions::get()};

    std::vector<char const*> extensionsStr{};
    extensionsStr.reserve(requiredExtensions.size());

    for (auto const& extension : requiredExtensions) {
        extensionsStr.push_back(extension.data());
    }

    features::RequiredFeatures requiredFeatures{};

    VkDeviceCreateInfo deviceCreateInfo = vku::InitStructHelper{};
    deviceCreateInfo.pNext = requiredFeatures.getChain();
    deviceCreateInfo.flags = 0;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = nullptr;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensionsStr.size());
    deviceCreateInfo.ppEnabledExtensionNames = extensionsStr.data();
    deviceCreateInfo.pEnabledFeatures = nullptr;

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        return std::unexpected{"failed to create logical device"};
    }

    return device;
}

}  // namespace gpu::impl
