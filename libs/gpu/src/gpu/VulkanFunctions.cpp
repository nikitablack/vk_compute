#include <fmt/core.h>

#include <gpu/VulkanFunctions.hpp>
#include <utils/try_expected.hpp>

namespace {

template <typename F>
auto get_function(VkInstance instance, char const* name) noexcept -> std::expected<F, std::string> {
    auto const f{reinterpret_cast<F>(vkGetInstanceProcAddr(instance, name))};

    if (!f) {
        return std::unexpected{fmt::format("failed to get function {}", name)};
    }

    return f;
}

}  // namespace

namespace gpu {

// debug utils
PFN_vkSetDebugUtilsObjectNameEXT VulkanFunctions::vkSetDebugUtilsObjectNameEXT{VK_NULL_HANDLE};

auto VulkanFunctions::initialize(VkInstance instance) noexcept -> std::expected<void, std::string> {
    fmt::println("initializing Vulkan functions");

    // debug utils
    TRY_EXPECTED(vkSetDebugUtilsObjectNameEXT,
                 get_function<PFN_vkSetDebugUtilsObjectNameEXT>(instance, "vkSetDebugUtilsObjectNameEXT"));

    return {};
}

}  // namespace gpu
