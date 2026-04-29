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
             DeviceBuffer const& deviceBuffer,  //
             HostVisibleBuffer const& stagingBuffer,  //
             VkDeviceSize dstBufferOffset,  //
             VkDeviceSize stagingBufferOffset  //
             ) noexcept -> std::expected<void, std::string> {
    if ((stagingBufferOffset > stagingBuffer.size()) ||
        ((stagingBuffer.size() - stagingBufferOffset) != deviceBuffer.size())) {
        return std::unexpected{"staging buffer size does not match buffer size"};
    }

    VkBufferCopy2 region = vku::InitStructHelper{};
    region.srcOffset = stagingBufferOffset;
    region.dstOffset = dstBufferOffset;
    region.size = stagingBuffer.size();

    VkCopyBufferInfo2 copyInfo = vku::InitStructHelper{};
    copyInfo.srcBuffer = stagingBuffer.buffer();
    copyInfo.dstBuffer = deviceBuffer.buffer();
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;

    vkCmdCopyBuffer2(commandBuffer, &copyInfo);

    return {};
}

[[nodiscard]] auto init_buffer(VkCommandBuffer commandBuffer,  //
                               DeviceBuffer const& deviceBuffer,  //
                               HostVisibleBuffer const& stagingBuffer,  //
                               VkDeviceSize dstBufferOffset,  //
                               VkDeviceSize stagingBufferOffset  //
                               ) noexcept -> std::expected<InitData, std::string> {
    utils::before_write(commandBuffer, deviceBuffer);
    TRY_EXPECTED_VOID(copy_to(commandBuffer, deviceBuffer, stagingBuffer, dstBufferOffset, stagingBufferOffset));
    utils::after_write(commandBuffer, deviceBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return std::unexpected{"failed to end copy command buffer"};
    }

    InitData initData{};
    initData.commandBuffer = commandBuffer;
    initData.stagingBuffer = stagingBuffer;

    return initData;
}

auto init_buffer_sync(DeviceBuffer const& deviceBuffer,  //
                      std::span<std::byte const> data  //
                      ) noexcept -> std::expected<void, std::string> {
    if (!GpuManager::initialized()) {
        return std::unexpected{"GpuManager is not initialized. Did you forget to call GpuManager::init()?"};
    }

    GpuManager& gpuManager{GpuManager::get()};

    HostVisibleBuffer stagingBuffer{};
    TRY_EXPECTED_VOID(stagingBuffer.init(data.size()));
    TRY_EXPECTED_VOID(stagingBuffer.copyTo(data));

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();
        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }
    })};

    TRY_EXPECTED(InitData initData,
                 init_buffer(commandBuffer,  //
                             deviceBuffer,  //
                             stagingBuffer));

    TRY_EXPECTED_VOID(submit_init_data_sync(std::vector<InitData>{std::move(initData)}, gpuManager.computeQueue()));

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
