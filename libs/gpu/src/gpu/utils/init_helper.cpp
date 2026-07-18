#include <spdlog/spdlog.h>

#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/VulkanQueue.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/init_helper.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/try_expected.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::utils {

auto copy_to(VkCommandBuffer commandBuffer,  //
             DeviceBuffer const& dst,  //
             HostVisibleBuffer const& src,  //
             VkDeviceSize sizeBytes,  //
             VkDeviceSize dstOffset,  //
             VkDeviceSize srcOffset  //
             ) noexcept -> std::expected<void, std::string> {
    if ((src.size() < (srcOffset + sizeBytes)) ||  //
        (dst.size() < (dstOffset + sizeBytes))) {
        return std::unexpected{"input buffers are too small"};
    }

    VkBufferCopy2 region = vku::InitStructHelper{};
    region.srcOffset = srcOffset;
    region.dstOffset = dstOffset;
    region.size = sizeBytes;

    VkCopyBufferInfo2 copyInfo = vku::InitStructHelper{};
    copyInfo.srcBuffer = src.buffer();
    copyInfo.dstBuffer = dst.buffer();
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;

    vkCmdCopyBuffer2(commandBuffer, &copyInfo);

    return {};
}

[[nodiscard]] auto init_buffer(VkCommandBuffer commandBuffer,  //
                               DeviceBuffer const& dst,  //
                               HostVisibleBuffer src,  //
                               VkDeviceSize sizeBytes,  //
                               VkDeviceSize dstOffset,  //
                               VkDeviceSize srcOffset  //
                               ) noexcept -> std::expected<InitData, std::string> {
    utils::before_write(commandBuffer, dst);
    TRY_EXPECTED_VOID(copy_to(commandBuffer, dst, src, sizeBytes, dstOffset, srcOffset));
    utils::after_write(commandBuffer, dst);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return std::unexpected{"failed to end copy command buffer"};
    }

    InitData initData{};
    initData.commandBuffer = commandBuffer;
    initData.stagingBuffer = std::move(src);

    return initData;
}

auto init_buffer_sync(DeviceBuffer const& dst,  //
                      std::span<std::byte const> src,  //
                      VkDeviceSize dstOffset  //
                      ) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    TRY_EXPECTED(auto stagingBuffer, HostVisibleBuffer::create(src));

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();

        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }

        stagingBuffer.destroy();
    })};

    TRY_EXPECTED(InitData initData,
                 init_buffer(commandBuffer,  //
                             dst,  //
                             std::move(stagingBuffer),  //
                             src.size_bytes(),  //
                             dstOffset,  //
                             0));

    std::vector<InitData> v{};
    v.push_back(std::move(initData));

    TRY_EXPECTED_VOID(submit_init_data_sync(std::move(v), gpuManager.computeQueue()));

    return {};
}

auto submit_init_data_sync(std::vector<InitData>&& initData,  //
                           VulkanQueue const& queue  //
                           ) noexcept -> std::expected<void, std::string> {
    std::vector<VkCommandBufferSubmitInfo> submitInfos{};
    submitInfos.reserve(initData.size());

    for (auto const& data : initData) {
        VkCommandBufferSubmitInfo commandBufferSubmitInfo = vku::InitStructHelper{};
        commandBufferSubmitInfo.commandBuffer = data.commandBuffer;
        commandBufferSubmitInfo.deviceMask = 0;

        submitInfos.push_back(commandBufferSubmitInfo);
    }

    VkSubmitInfo2 submitInfo = vku::InitStructHelper{};
    submitInfo.flags = 0;
    submitInfo.waitSemaphoreInfoCount = 0;
    submitInfo.pWaitSemaphoreInfos = nullptr;
    submitInfo.commandBufferInfoCount = static_cast<uint32_t>(submitInfos.size());
    submitInfo.pCommandBufferInfos = submitInfos.data();
    submitInfo.signalSemaphoreInfoCount = 0;
    submitInfo.pSignalSemaphoreInfos = nullptr;

    if (vkQueueSubmit2(queue.queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        return std::unexpected{"failed to submit staging command buffer"};
    }

    if (vkQueueWaitIdle(queue.queue) != VK_SUCCESS) {
        return std::unexpected{"failed to wait staging queue"};
    }

    for (auto& data : initData) {
        data.stagingBuffer.destroy();
    }

    return {};
}

}  // namespace gpu::utils
