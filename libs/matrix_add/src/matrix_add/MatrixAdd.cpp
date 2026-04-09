#include <gpu/GpuManager.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <matrix_add/MatrixAdd.hpp>
#include <matrix_add/impl/create_matrix_add_pipeline.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace matrix_add {

auto MatrixAdd::destroy() noexcept -> void {
    if (!m_device) {
        return;
    }

    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_bufferA.destroy();
    m_bufferB.destroy();
    m_bufferOut.destroy();
    m_stagingBuffer.destroy();
}

auto MatrixAdd::run(gpu::GpuManager& gpuManager,  //
                    std::span<float const> a,  //
                    std::span<float const> b,  //
                    std::vector<float>& out  //
                    ) noexcept -> std::expected<void, std::string> {
    if (a.size() != b.size()) {
        return std::unexpected("input buffers size mismatch");
    }

    if (!m_device) {
        m_device = gpuManager.device();
        TRY_EXPECTED(m_pipeline, impl::create_matrix_add_pipeline(m_device, gpuManager.pipelineLayout()));
    }

    uint32_t dataSize{static_cast<uint32_t>(a.size_bytes())};

    // create buffers
    {
        if (dataSize > m_bufferA.size()) {
            m_bufferA.destroy();
            TRY_EXPECTED_VOID(
                m_bufferA.init(gpuManager.allocator(),  //
                               VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,  //
                               dataSize));
        }

        if (dataSize > m_bufferB.size()) {
            m_bufferB.destroy();
            TRY_EXPECTED_VOID(
                m_bufferB.init(gpuManager.allocator(),  //
                               VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,  //
                               dataSize));
        }

        if (dataSize > m_bufferOut.size()) {
            m_bufferOut.destroy();
            TRY_EXPECTED_VOID(
                m_bufferOut.init(gpuManager.allocator(),  //
                                 VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,  //
                                 dataSize));
        }
    }

    // copy data
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(gpuManager,  //
                                                       m_bufferA,  //
                                                       utils::to_byte_span(a)));

        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(gpuManager,  //
                                                       m_bufferB,  //
                                                       utils::to_byte_span(b)));
    }

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // update descriptors
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_bufferA.buffer();
        bufferInfo.offset = 0;
        bufferInfo.range = dataSize;

        TRY_EXPECTED(uint32_t const descriptorIndexA,
                     gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

        bufferInfo.buffer = m_bufferB.buffer();

        TRY_EXPECTED(uint32_t const descriptorIndexB,
                     gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

        bufferInfo.buffer = m_bufferOut.buffer();

        TRY_EXPECTED(uint32_t const descriptorIndexOut,
                     gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

        // see matrix_add.cpmp
        auto const pushConstData{gpu::utils::get_push_constant_data(static_cast<uint32_t>(a.size()),  //
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
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdDispatch(commandBuffer, 1, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            return std::unexpected{"failed to end copy command buffer"};
        }

        TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer, gpuManager.computeQueue().queue));

        if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
            return std::unexpected{"failed to wait queue"};
        }
    }

    // read back
    {
        if (dataSize > m_stagingBuffer.size()) {
            m_stagingBuffer.destroy();
            TRY_EXPECTED_VOID(m_stagingBuffer.init(gpuManager.allocator(),  //
                                                   VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,  //
                                                   dataSize,  //
                                                   true));
        }

        TRY_EXPECTED_VOID(gpu::utils::read_data_sync(gpuManager, m_bufferOut, m_stagingBuffer, dataSize));

        if ((out.size() * sizeof(float)) < dataSize) {
            out.resize(dataSize / sizeof(float));
        }

        TRY_EXPECTED_VOID(m_stagingBuffer.copyFrom(out.data(), dataSize, 0));
    }

    return {};
}

}  // namespace matrix_add
