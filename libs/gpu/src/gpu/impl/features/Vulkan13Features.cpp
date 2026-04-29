#include <spdlog/spdlog.h>

#include <gpu/impl/features/Vulkan13Features.hpp>
#include <unordered_map>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace {

auto featureNameToPtr() noexcept -> std::unordered_map<std::string, VkBool32 VkPhysicalDeviceVulkan13Features::*> {
    std::unordered_map<std::string, VkBool32 VkPhysicalDeviceVulkan13Features::*> const nameToPtr{
        {"maintenance4", &VkPhysicalDeviceVulkan13Features::maintenance4},
        {"synchronization2", &VkPhysicalDeviceVulkan13Features::synchronization2}};

    return nameToPtr;
}

}  // namespace

namespace gpu::impl::features {

Vulkan13Features::Vulkan13Features() noexcept {
    m_features = vku::InitStructHelper{};

    for (auto const& p : featureNameToPtr()) {
        m_features.*(p.second) = VK_TRUE;
    }
}

auto Vulkan13Features::addToChain(void** pNext) noexcept -> void** {
    *pNext = &m_features;

    return &m_features.pNext;
}

auto Vulkan13Features::check(VkPhysicalDevice physicalDevice) const noexcept -> bool {
    VkPhysicalDeviceVulkan13Features features13 = vku::InitStructHelper{};
    VkPhysicalDeviceFeatures2 features2 = vku::InitStructHelper{&features13};

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    bool result{true};

    for (auto const& p : featureNameToPtr()) {
        if (features13.*(p.second) == VK_FALSE) {
            result = false;
            spdlog::info("\tVkPhysicalDeviceVulkan13Features::{} is not supported.", p.first);
        }
    }

    return result;
}

auto Vulkan13Features::print() const noexcept -> void {
    spdlog::info("required VkPhysicalDeviceVulkan13Features:");
    for (auto const& p : featureNameToPtr()) {
        spdlog::info("\tVkPhysicalDeviceVulkan13Features::{}", p.first);
    }
}

}  // namespace gpu::impl::features
