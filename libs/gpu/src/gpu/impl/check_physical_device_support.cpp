#include <spdlog/spdlog.h>

#include <gpu/impl/RequiredApiVersion.hpp>
#include <gpu/impl/RequiredDeviceExtensions.hpp>
#include <gpu/impl/features/RequiredFeatures.hpp>
#include <gpu/impl/get_physical_device_properties.hpp>
#include <utils/try_expected.hpp>
#include <vector>

namespace {

auto check_required_device_extensions(VkPhysicalDevice device) noexcept -> std::expected<void, std::string> {
    uint32_t extensionCount{0};

    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr) != VK_SUCCESS) {
        return std::unexpected{"failed to get physical device extension properties"};
    }

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);

    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data()) !=
        VK_SUCCESS) {
        return std::unexpected{"failed to get physical device extension properties"};
    }

    for (auto const& reqExt : gpu::impl::RequiredDeviceExtensions::get()) {
        for (auto const& avExt : availableExtensions) {
            if (reqExt == avExt.extensionName) {
                goto cnt;
            }
        }

        return std::unexpected{"required device extension \"" + reqExt + "\" is not supported"};

    cnt:;
    }

    return {};
}

}  // namespace

namespace gpu::impl {

[[nodiscard]] auto check_physical_device_support(VkPhysicalDevice device) noexcept -> std::expected<void, std::string> {
    const auto props{get_physical_device_properties(device)};

    spdlog::info("checking physical device support: {}", props.properties.deviceName);

    // api version
    const uint32_t major{VK_API_VERSION_MAJOR(props.properties.apiVersion)};
    const uint32_t minor{VK_API_VERSION_MINOR(props.properties.apiVersion)};
    const uint32_t patch{VK_API_VERSION_PATCH(props.properties.apiVersion)};

    if ((major < RequiredApiVersion::MAJOR) ||
        (major == RequiredApiVersion::MAJOR && minor < RequiredApiVersion::MINOR)) {
        return std::unexpected{fmt::format("error: device's api version: {}.{}.{}", major, minor, patch)};
    }

    if (!features::RequiredFeatures::check(device)) {
        return std::unexpected{"error: required features are not supported"};
    }

    TRY_EXPECTED_VOID(check_required_device_extensions(device));

    return {};
}

}  // namespace gpu::impl
