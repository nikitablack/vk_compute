#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto create_pipeline_layout(VkDevice device,  //
                                          VkDescriptorSetLayout storageDescriptorSetLayout  //
                                          ) noexcept -> std::expected<VkPipelineLayout, std::string>;

}  // namespace gpu::impl
