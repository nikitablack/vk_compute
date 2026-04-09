#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <span>
#include <string>

namespace gpu::utils {

[[nodiscard]] auto create_shader_module(VkDevice device,  //
                                        std::span<std::byte const> shader  //
                                        ) noexcept -> std::expected<VkShaderModule, std::string>;

}  // namespace gpu::utils
