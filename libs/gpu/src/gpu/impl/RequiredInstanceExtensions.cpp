#include <spdlog/spdlog.h>
#include <vulkan/vulkan.h>

#include <gpu/impl/RequiredInstanceExtensions.hpp>

namespace gpu::impl {

auto RequiredInstanceExtensions::get() noexcept -> std::vector<std::string> {
    std::vector<std::string> extensions{};
    extensions.reserve(1);

#ifdef ENABLE_VULKAN_DEBUG_UTILS
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

    return extensions;
}

auto RequiredInstanceExtensions::print() noexcept -> void {
    spdlog::info("required instance extensions:");
    for (auto const& ext : get()) {
        spdlog::info("\t{}", ext);
    }
}

}  // namespace gpu::impl
