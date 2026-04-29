#include <spdlog/spdlog.h>

#include <array>
#include <gpu/impl/create_pipeline_layout.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto create_pipeline_layout(VkDevice device,  //
                            VkDescriptorSetLayout storageDescriptorSetLayout  //
                            ) noexcept -> std::expected<VkPipelineLayout, std::string> {
    spdlog::trace("creating pipeline layout");

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 256;  // guaranteed by the Spec for Vulkan >= 1.4

    VkPipelineLayoutCreateInfo info = vku::InitStructHelper{};
    info.flags = 0;
    info.setLayoutCount = 1;
    info.pSetLayouts = &storageDescriptorSetLayout;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &pushConstantRange;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &info, nullptr, &pipelineLayout) != VK_SUCCESS) {
        return std::unexpected{"failed to create pipeline layout"};
    }

    return pipelineLayout;
}

}  // namespace gpu::impl
