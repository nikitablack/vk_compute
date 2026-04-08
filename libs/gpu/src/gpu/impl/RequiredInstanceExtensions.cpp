#include <fmt/core.h>
#include <vulkan/vulkan.h>

#include <gpu/impl/RequiredInstanceExtensions.hpp>

namespace gpu::impl {

auto RequiredInstanceExtensions::get() noexcept -> std::vector<std::string> {
    std::vector<std::string> extensions{};
    extensions.reserve(2);

    extensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);

#ifdef ENABLE_VULKAN_DEBUG_UTILS
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    return extensions;
}

auto RequiredInstanceExtensions::print() noexcept -> void {
    fmt::println("required instance extensions:");

    for (auto const& ext : get()) {
        fmt::println("\t{}", ext);
    }
}

}  // namespace gpu::impl
