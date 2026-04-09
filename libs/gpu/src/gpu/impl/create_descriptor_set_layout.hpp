#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto create_descriptor_set_layout(VkDevice device,  //
                                                VkDescriptorType descriptorType,  //
                                                uint32_t requiredDescriptorCount  //
                                                ) noexcept -> std::expected<VkDescriptorSetLayout, std::string>;

}  // namespace gpu::impl
