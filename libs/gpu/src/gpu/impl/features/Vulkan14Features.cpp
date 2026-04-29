#include <spdlog/spdlog.h>

#include <gpu/impl/features/Vulkan14Features.hpp>
#include <unordered_map>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace {

auto featureNameToPtr() noexcept -> std::unordered_map<std::string, VkBool32 VkPhysicalDeviceVulkan14Features::*> {
    std::unordered_map<std::string, VkBool32 VkPhysicalDeviceVulkan14Features::*> const nameToPtr{
        {"maintenance5", &VkPhysicalDeviceVulkan14Features::maintenance5}};  // for VkBufferUsageFlags2CreateInfo

    return nameToPtr;
}

}  // namespace

namespace gpu::impl::features {

Vulkan14Features::Vulkan14Features() noexcept {
    m_features = vku::InitStructHelper{};

    for (auto const& p : featureNameToPtr()) {
        m_features.*(p.second) = VK_TRUE;
    }
}

auto Vulkan14Features::addToChain(void** pNext) noexcept -> void** {
    *pNext = &m_features;

    return &m_features.pNext;
}

auto Vulkan14Features::check(VkPhysicalDevice physicalDevice) const noexcept -> bool {
    VkPhysicalDeviceVulkan14Features features14 = vku::InitStructHelper{};
    VkPhysicalDeviceFeatures2 features2 = vku::InitStructHelper{&features14};

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    bool result{true};

    for (auto const& p : featureNameToPtr()) {
        if (features14.*(p.second) == VK_FALSE) {
            result = false;
            spdlog::info("\tVkPhysicalDeviceVulkan14Features::{} is not supported.", p.first);
        }
    }

    return result;
}

auto Vulkan14Features::print() const noexcept -> void {
    spdlog::info("required VkPhysicalDeviceVulkan14Features:");
    for (auto const& p : featureNameToPtr()) {
        spdlog::info("\tVkPhysicalDeviceVulkan14Features::{}", p.first);
    }
}

}  // namespace gpu::impl::features
