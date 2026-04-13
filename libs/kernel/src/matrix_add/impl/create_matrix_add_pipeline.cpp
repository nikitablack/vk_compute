#include <cmrc/cmrc.hpp>
#include <gpu/utils/create_shader_module.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <matrix_add/impl/create_matrix_add_pipeline.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

CMRC_DECLARE(kernel_shaders);

namespace matrix_add::impl {

auto create_matrix_add_pipeline(VkDevice device,  //
                                VkPipelineLayout pipelineLayout,  //
                                uint32_t workgroupSizeX,  //
                                uint32_t workgroupSizeY,  //
                                uint32_t workgroupSizeZ  //
                                ) noexcept -> std::expected<VkPipeline, std::string> {
    std::array<VkSpecializationMapEntry, 3> cpecEntries{};
    cpecEntries[0].constantID = 0;
    cpecEntries[0].offset = 0;
    cpecEntries[0].size = sizeof(uint32_t);
    cpecEntries[1].constantID = 1;
    cpecEntries[1].offset = sizeof(uint32_t);
    cpecEntries[1].size = sizeof(uint32_t);
    cpecEntries[2].constantID = 2;
    cpecEntries[2].offset = 2 * sizeof(uint32_t);
    cpecEntries[2].size = sizeof(uint32_t);

    auto const specData{gpu::utils::get_push_constant_data(workgroupSizeX, workgroupSizeY, workgroupSizeZ)};

    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = cpecEntries.size();
    specInfo.pMapEntries = cpecEntries.data();
    specInfo.dataSize = specData.size();
    specInfo.pData = specData.data();

    auto const fs{cmrc::kernel_shaders::get_filesystem()};
    auto const shader{fs.open("matrix_add.comp.spv")};

    TRY_EXPECTED(auto const shaderModule,
                 gpu::utils::create_shader_module(device, utils::to_byte_span(shader.cbegin(), shader.size())));

    VkPipelineShaderStageCreateInfo shaderStageInfo = vku::InitStructHelper{};
    shaderStageInfo.flags = 0;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";
    shaderStageInfo.pSpecializationInfo = &specInfo;

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