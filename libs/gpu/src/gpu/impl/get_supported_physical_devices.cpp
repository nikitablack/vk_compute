#include <fmt/core.h>

#include <gpu/impl/RequiredDeviceExtensions.hpp>
#include <gpu/impl/check_physical_device_support.hpp>
#include <gpu/impl/features/RequiredFeatures.hpp>
#include <gpu/impl/get_physical_device_properties.hpp>

namespace gpu::impl {

[[nodiscard]] auto get_supported_physical_devices(VkInstance instance) noexcept
    -> std::expected<std::vector<VkPhysicalDevice>, std::string> {
    fmt::println("getting supported physical devices");

    RequiredDeviceExtensions::print();
    features::RequiredFeatures::print();

    uint32_t deviceCount{0};

    if (vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr) != VK_SUCCESS || deviceCount == 0) {
        return std::unexpected{"failed to find GPUs with Vulkan support"};
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);

    if (vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()) != VK_SUCCESS) {
        return std::unexpected{"failed to find GPUs with Vulkan support"};
    }

    std::vector<VkPhysicalDevice> supportedPhysicalDevices{};
    supportedPhysicalDevices.reserve(deviceCount);

    for (auto const device : physicalDevices) {
        auto const props{get_physical_device_properties(device)};

        if (auto const result{check_physical_device_support(device)}; result) {
            supportedPhysicalDevices.push_back(device);
            fmt::println("\t{} is supported", props.properties.deviceName);
        } else {
            fmt::println("\t{} is not supported: \n\t\t{}", props.properties.deviceName, result.error());
        }
    }

    if (supportedPhysicalDevices.empty()) {
        return std::unexpected{"failed to find any supported devices"};
    }

    fmt::println("supported devices:");

    for (size_t i{0}; i < supportedPhysicalDevices.size(); ++i) {
        auto const props{get_physical_device_properties(supportedPhysicalDevices[i])};

        fmt::println("\t{}: {}", i, props.properties.deviceName);
    }

    return supportedPhysicalDevices;
}

}  // namespace gpu::impl
