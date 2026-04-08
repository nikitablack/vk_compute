#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto create_instance() noexcept -> std::expected<VkInstance, std::string>;

}  // namespace gpu::impl
