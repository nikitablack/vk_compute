#include <fmt/core.h>
#include <vulkan/vulkan.h>

#include <gpu/impl/RequiredDeviceExtensions.hpp>

namespace gpu::impl {

auto RequiredDeviceExtensions::get() noexcept -> std::vector<std::string> {
    std::vector<std::string> extensions{};

    extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);  // for debugPrintfEXT
    extensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);

    return extensions;
}

auto RequiredDeviceExtensions::print() noexcept -> void {
    fmt::println("required device extensions:");

    for (auto const& ext : get()) {
        fmt::println("\t{}", ext);
    }
}

}  // namespace gpu::impl
