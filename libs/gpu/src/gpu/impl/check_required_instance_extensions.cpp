#include <fmt/core.h>
#include <vulkan/vulkan.h>

#include <gpu/impl/RequiredInstanceExtensions.hpp>
#include <gpu/impl/check_required_instance_extensions.hpp>

namespace gpu::impl {

auto check_required_instance_extensions() noexcept -> std::expected<void, std::string> {
    fmt::println("checking required instance extensions");

    RequiredInstanceExtensions::print();

    uint32_t extensionCount{};
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr) != VK_SUCCESS) {
        return std::unexpected{"failed to enumerate instance extension properties"};
    }

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);

    if (vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()) != VK_SUCCESS) {
        return std::unexpected{"failed to enumerate instance extension properties"};
    }

    for (auto const& reqExt : RequiredInstanceExtensions::get()) {
        for (auto const& avExt : availableExtensions) {
            if (reqExt == avExt.extensionName) {
                goto cnt;
            }
        }

        return std::unexpected{"instance extension \"" + reqExt + "\" is not supported"};

    cnt:;
    }

    return {};
}

}  // namespace gpu::impl
