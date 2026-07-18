#include <spdlog/spdlog.h>

#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/Span.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/try_expected.hpp>

namespace gpu::utils {

[[nodiscard]] auto read_data_sync(Span const& dst,  //
                                  Span const& src  //
                                  ) -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    size_t const sizeBytes{src.size()};

    if (dst.size() < sizeBytes) {
        return std::unexpected{"dst buffer is too small"};
    }

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();
        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }
    })};

    set_buffer_barrier(commandBuffer,  //
                       src.buffer(),  //
                       src.offset(),  //
                       src.size(),  //
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_READ_BIT);

    set_buffer_barrier(commandBuffer,  //
                       dst.buffer(),  //
                       dst.offset(),  //
                       dst.size(),  //
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferCopy region{};
    region.srcOffset = src.offset();
    region.dstOffset = dst.offset();
    region.size = sizeBytes;

    vkCmdCopyBuffer(commandBuffer, src.buffer(), dst.buffer(), 1, &region);

    set_buffer_barrier(commandBuffer,  //
                       src.buffer(),  //
                       src.offset(),  //
                       src.size(),  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_READ_BIT,  //
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    set_buffer_barrier(commandBuffer,  //
                       dst.buffer(),  //
                       dst.offset(),  //
                       dst.size(),  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_WRITE_BIT,  //
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return std::unexpected{"failed to end copy command buffer"};
    }

    TRY_EXPECTED_VOID(submit(commandBuffer, gpuManager.computeQueue().queue, VK_NULL_HANDLE));

    if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
        return std::unexpected{"failed to wait queue"};
    }

    return {};
}

auto read_data_sync(DeviceBuffer const& src,  //
                    HostVisibleBuffer const& dst,  //
                    size_t sizeBytes,  //
                    size_t srcOffset,  //
                    size_t dstOffset  //
                    ) -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    if ((src.size() < (srcOffset + sizeBytes))) {
        return std::unexpected{"src buffer is too small"};
    }

    if ((dst.size() < (dstOffset + sizeBytes))) {
        return std::unexpected{"dst buffer is too small"};
    }

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();
        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }
    })};

    before_read(commandBuffer, src);

    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = dstOffset;
    region.size = sizeBytes;

    before_write(commandBuffer, dst);

    vkCmdCopyBuffer(commandBuffer, src.buffer(), dst.buffer(), 1, &region);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return std::unexpected{"failed to end copy command buffer"};
    }

    TRY_EXPECTED_VOID(submit(commandBuffer, gpuManager.computeQueue().queue, VK_NULL_HANDLE));

    if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
        throw std::unexpected{"failed to wait queue"};
    }

    return {};
}

}  // namespace gpu::utils
