#include <gpu/GpuManager.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/add.hpp>
#include <kernel/utils/create_pipeline.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace {

std::string constexpr KERNEL_NAME{"add"};

}

namespace kernel {

auto add(std::span<float const> a,  //
         std::span<float const> b,  //
         std::span<float> result  //
         ) noexcept -> std::expected<void, std::string> {
    // special case
    if (a.size() == 0) {
        return {};
    }

    // input validation
    if ((a.size() != b.size()) || (a.size() != result.size())) {
        return std::unexpected("input buffers size mismatch");
    }

    uint32_t const dataSizeBytes{static_cast<uint32_t>(a.size_bytes())};

    // create buffers
    gpu::DeviceBuffer aDevice{};
    gpu::DeviceBuffer bDevice{};
    gpu::DeviceBuffer resultDevice{};

    TRY_EXPECTED_VOID(aDevice.init(dataSizeBytes));
    TRY_EXPECTED_VOID(bDevice.init(dataSizeBytes));
    TRY_EXPECTED_VOID(resultDevice.init(dataSizeBytes));

    // copy data
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(aDevice,  //
                                                       ::utils::to_byte_span(a)));

        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(bDevice,  //
                                                       ::utils::to_byte_span(b)));
    }

    // run compute
    TRY_EXPECTED_VOID(add(aDevice, bDevice, resultDevice, dataSizeBytes));

    // read back
    gpu::HostVisibleBuffer stagingBuffer{};

    {
        TRY_EXPECTED_VOID(stagingBuffer.init(dataSizeBytes, true));

        TRY_EXPECTED_VOID(gpu::utils::read_data_sync(resultDevice, stagingBuffer, dataSizeBytes));
        TRY_EXPECTED_VOID(stagingBuffer.copyFrom(result.data(), dataSizeBytes));
    }

    aDevice.destroy();
    bDevice.destroy();
    resultDevice.destroy();
    stagingBuffer.destroy();

    return {};
}

auto add(gpu::DeviceBuffer const& a,  //
         gpu::DeviceBuffer const& b,  //
         gpu::DeviceBuffer const& result,  //
         std::optional<uint64_t> sizeBytes  //
         ) noexcept -> std::expected<void, std::string> {
    using T = float;

    uint64_t dataSizeBytes{0};
    uint32_t dataCount{0};

    // input validation
    {
        if (sizeBytes) {
            dataSizeBytes = *sizeBytes;

            if (dataSizeBytes == 0) {
                return {};
            }

            if ((a.size() < dataSizeBytes) || (b.size() < dataSizeBytes) || (result.size() < dataSizeBytes)) {
                return std::unexpected("input buffers are too small");
            }

        } else {
            dataSizeBytes = a.size();

            if (dataSizeBytes == 0) {
                return {};
            }

            if ((dataSizeBytes != b.size()) || (dataSizeBytes != result.size())) {
                return std::unexpected("input buffers size mismatch");
            }
        }

        if ((dataSizeBytes % sizeof(T)) != 0) {
            return std::unexpected("input size should be multiple of T");
        }

        dataCount = static_cast<uint32_t>(dataSizeBytes / sizeof(T));
    }

    uint32_t constexpr WORKGROUP_SIZE_X{1024};
    uint32_t constexpr WORKGROUP_SIZE_Y{1};
    uint32_t constexpr WORKGROUP_SIZE_Z{1};
    uint32_t constexpr WORKGROUP_SIZE{WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y * WORKGROUP_SIZE_Z};

    gpu::GpuManager& gpuManager{gpu::GpuManager::get()};
    VkPipeline vkPipeline{VK_NULL_HANDLE};

    if (auto p{gpuManager.getPipeline(KERNEL_NAME,  //
                                      WORKGROUP_SIZE_X,  //
                                      WORKGROUP_SIZE_Y,  //
                                      WORKGROUP_SIZE_Z)}) {
        vkPipeline = *p;
    } else {
        TRY_EXPECTED(vkPipeline, utils::create_pipeline(KERNEL_NAME,  //
                                                        WORKGROUP_SIZE_X,  //
                                                        WORKGROUP_SIZE_Y,  //
                                                        WORKGROUP_SIZE_Z));

        gpuManager.addPipeline(vkPipeline,  //
                               KERNEL_NAME,  //
                               WORKGROUP_SIZE_X,  //
                               WORKGROUP_SIZE_Y,  //
                               WORKGROUP_SIZE_Z);
    }

    // compute
    {
        TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = a.buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = dataSizeBytes;

            TRY_EXPECTED(uint32_t const descriptorIndexA,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = b.buffer();

            TRY_EXPECTED(uint32_t const descriptorIndexB,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = result.buffer();

            TRY_EXPECTED(uint32_t const descriptorIndexOut,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // see add.cpmp
            auto const pushConstData{gpu::utils::get_push_constant_data(dataCount,  //
                                                                        descriptorIndexA,  //
                                                                        descriptorIndexB,  //
                                                                        descriptorIndexOut)};

            vkCmdPushConstants(commandBuffer,  //
                               gpuManager.pipelineLayout(),  //
                               VK_SHADER_STAGE_COMPUTE_BIT,  //
                               0,  //
                               static_cast<uint32_t>(pushConstData.size()),  //
                               pushConstData.data());
        }

        // dispatch and wait
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipeline);

        uint32_t const numGroupsX{(dataCount + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE};
        vkCmdDispatch(commandBuffer, numGroupsX, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            return std::unexpected{"failed to end copy command buffer"};
        }

        TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer, gpuManager.computeQueue().queue));

        if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
            return std::unexpected{"failed to wait queue"};
        }

        gpuManager.storageDescriptorSetManager().reset();
        TRY_EXPECTED_VOID(gpuManager.commandManager().resetCommandBuffer(commandBuffer));
    }

    return {};
}

}  // namespace kernel
