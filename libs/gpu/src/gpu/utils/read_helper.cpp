#include <spdlog/spdlog.h>

#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/try_expected.hpp>

namespace gpu::utils {

auto read_data_sync(DeviceBuffer const& src,  //
                    HostVisibleBuffer const& dst,  //
                    uint32_t size  //
                    ) -> std::expected<void, std::string> {
    if (!GpuManager::initialized()) {
        return std::unexpected{"GpuManager is not initialized. Did you forget to call GpuManager::init()?"};
    }

    if (dst.size() < src.size()) {
        return std::unexpected{"not enough space in destination buffer"};
    }

    GpuManager& gpuManager{GpuManager::get()};

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
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = size;

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
