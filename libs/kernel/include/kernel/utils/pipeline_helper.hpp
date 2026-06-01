#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace kernel::utils {

[[nodiscard]] auto create_pipeline(std::string const& shaderName,  //
                                   uint32_t workgroupSizeX,  //
                                   uint32_t workgroupSizeY,  //
                                   uint32_t workgroupSizeZ  //
                                   ) noexcept -> std::expected<VkPipeline, std::string>;

[[nodiscard]] auto get_pipeline(std::string const& shaderName,  //
                                uint32_t workgroupSizeX,  //
                                uint32_t workgroupSizeY,  //
                                uint32_t workgroupSizeZ  //
                                ) noexcept -> std::expected<VkPipeline, std::string>;

}  // namespace kernel::utils
