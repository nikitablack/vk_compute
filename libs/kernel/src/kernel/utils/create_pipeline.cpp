#include <cmrc/cmrc.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/create_shader_module.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <kernel/utils/create_pipeline.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

CMRC_DECLARE(kernel_shaders);

namespace kernel::utils {

auto create_pipeline(std::string const& shaderName,
                     uint32_t workgroupSizeX,  //
                     uint32_t workgroupSizeY,  //
                     uint32_t workgroupSizeZ  //
                     ) noexcept -> std::expected<VkPipeline, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

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
    auto const shader{fs.open(shaderName + ".comp.spv")};

    TRY_EXPECTED(auto const shaderModule,
                 gpu::utils::create_shader_module(gpuManager.device(),  //
                                                  ::utils::to_byte_span(shader.cbegin(), shader.size())));

    VkPipelineShaderStageCreateInfo shaderStageInfo = vku::InitStructHelper{};
    shaderStageInfo.flags = 0;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";
    shaderStageInfo.pSpecializationInfo = &specInfo;

    VkComputePipelineCreateInfo pipelineInfo = vku::InitStructHelper{};
    pipelineInfo.flags = 0;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = gpuManager.pipelineLayout();
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VkPipeline pipeline{VK_NULL_HANDLE};
    if (vkCreateComputePipelines(gpuManager.device(), nullptr, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        return std::unexpected{"failed to create pipeline"};
    }

    vkDestroyShaderModule(gpuManager.device(), shaderModule, nullptr);

    return pipeline;
}

}  // namespace kernel::utils