#include <spdlog/spdlog.h>

#include <gpu/impl/features/RequiredFeatures.hpp>
#include <gpu/impl/features/Vulkan12Features.hpp>
#include <gpu/impl/features/Vulkan13Features.hpp>
#include <gpu/impl/features/Vulkan14Features.hpp>
#include <unordered_map>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace {

auto featureNameToPtr() noexcept -> std::unordered_map<std::string, VkBool32 VkPhysicalDeviceFeatures::*> {
    std::unordered_map<std::string, VkBool32 VkPhysicalDeviceFeatures::*> const nameToPtr{};

    return nameToPtr;
}

}  // namespace

namespace gpu::impl::features {

RequiredFeatures::RequiredFeatures() noexcept {
    m_features2 = vku::InitStructHelper{};

    for (auto const& p : featureNameToPtr()) {
        m_features2.features.*(p.second) = VK_TRUE;
    }

    addFeature<Vulkan12Features>();
    addFeature<Vulkan13Features>();
    addFeature<Vulkan14Features>();
}

auto RequiredFeatures::printImpl() const noexcept -> void {
    spdlog::info("required VkPhysicalDeviceFeatures2:");
    for (auto const& p : featureNameToPtr()) {
        spdlog::info("\tVkPhysicalDeviceFeatures2::{}", p.first);
    }

    for (auto const& f : m_features) {
        f->print();
    }
}

auto RequiredFeatures::checkImpl(VkPhysicalDevice physicalDevice) const noexcept -> bool {
    VkPhysicalDeviceFeatures2 features = vku::InitStructHelper{};

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);

    bool result{true};

    for (auto const& p : featureNameToPtr()) {
        if (features.features.*(p.second) == VK_FALSE) {
            result = false;
            spdlog::info("\tVkPhysicalDeviceFeatures2::{} is not supported.", p.first);
        }
    }

    for (auto const& f : m_features) {
        result &= f->check(physicalDevice);
    }

    return result;
}

auto RequiredFeatures::getChain() noexcept -> VkPhysicalDeviceFeatures2 const* {
    auto ptr{&m_features2.pNext};

    for (auto& f : m_features) {
        ptr = f->addToChain(ptr);
    }

    return &m_features2;
}

auto RequiredFeatures::print() noexcept -> void {
    RequiredFeatures{}.printImpl();
}

auto RequiredFeatures::check(VkPhysicalDevice physicalDevice) noexcept -> bool {
    return RequiredFeatures{}.checkImpl(physicalDevice);
}

}  // namespace gpu::impl::features
