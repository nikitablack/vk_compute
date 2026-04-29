#include <spdlog/spdlog.h>
#include <vulkan/vulkan.h>

#include <gpu/impl/RequiredApiVersion.hpp>
#include <gpu/impl/check_instance_version.hpp>

namespace gpu::impl {

auto check_instance_version() noexcept -> std::expected<void, std::string> {
    spdlog::trace("checking instance version");

    auto const f{
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"))};

    if (!f) {
        return std::unexpected{"detected Vulkan version is < 1.1"};
    }

    uint32_t apiVersion;
    if (vkEnumerateInstanceVersion(&apiVersion) != VK_SUCCESS) {
        return std::unexpected{"failed to enumerate instance version"};
    }

    uint32_t const major{VK_API_VERSION_MAJOR(apiVersion)};
    uint32_t const minor{VK_API_VERSION_MINOR(apiVersion)};
    uint32_t const patch{VK_API_VERSION_PATCH(apiVersion)};

    if ((major < RequiredApiVersion::MAJOR) ||
        (major == RequiredApiVersion::MAJOR && minor < RequiredApiVersion::MINOR)) {
        return std::unexpected{fmt::format("error: detected Vulkan version: {}.{}.{}", major, minor, patch)};
    }

    spdlog::info("Vulkan version: {}.{}.{}", major, minor, patch);

    return {};
}

}  // namespace gpu::impl
