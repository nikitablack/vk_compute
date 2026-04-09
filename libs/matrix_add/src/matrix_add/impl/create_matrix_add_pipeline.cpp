#include <cmrc/cmrc.hpp>
#include <gpu/utils/create_shader_module.hpp>
#include <matrix_add/impl/create_matrix_add_pipeline.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

CMRC_DECLARE(matrix_add_shaders);

namespace matrix_add::impl {

auto create_matrix_add_pipeline(VkDevice device,  //
                                VkPipelineLayout pipelineLayout  //
                                ) noexcept -> std::expected<VkPipeline, std::string> {
    auto const fs{cmrc::matrix_add_shaders::get_filesystem()};
    auto const shader{fs.open("matrix_add.comp.spv")};

    TRY_EXPECTED(auto const shaderModule,
                 gpu::utils::create_shader_module(device, utils::to_byte_span(shader.cbegin(), shader.size())));

    VkPipelineShaderStageCreateInfo shaderStageInfo = vku::InitStructHelper{};
    shaderStageInfo.flags = 0;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";
    shaderStageInfo.pSpecializationInfo = nullptr;

    VkComputePipelineCreateInfo pipelineInfo = vku::InitStructHelper{};
    pipelineInfo.flags = 0;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkPipeline pipeline{VK_NULL_HANDLE};
    if (vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        return std::unexpected{"failed to create matrix add pipeline"};
    }

    vkDestroyShaderModule(device, shaderModule, nullptr);

    return pipeline;
}

}  // namespace matrix_add::impl