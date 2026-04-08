#pragma once

#include <gpu/impl/features/Features.hpp>
#include <memory>
#include <type_traits>
#include <vector>

namespace gpu::impl::features {

class RequiredFeatures {
public:
    RequiredFeatures() noexcept;

public:
    static auto print() noexcept -> void;
    static auto check(VkPhysicalDevice physicalDevice) noexcept -> bool;
    auto getChain() noexcept -> VkPhysicalDeviceFeatures2 const*;

public:
    template <DerivedFromFeatures F>
    auto addFeature() noexcept -> void {
        m_features.push_back(std::make_unique<F>());
    }

    auto printImpl() const noexcept -> void;
    auto checkImpl(VkPhysicalDevice physicalDevice) const noexcept -> bool;

private:
    VkPhysicalDeviceFeatures2 m_features2{};
    std::vector<std::unique_ptr<Features>> m_features{};
};

}  // namespace gpu::impl::features
