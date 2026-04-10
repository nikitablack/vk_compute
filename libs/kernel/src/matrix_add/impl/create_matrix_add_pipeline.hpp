#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace matrix_add::impl {

[[nodiscard]] auto create_matrix_add_pipeline(VkDevice device,  //
                                              VkPipelineLayout pipelineLayout  //
                                              ) noexcept -> std::expected<VkPipeline, std::string>;

}  // namespace matrix_add::impl
