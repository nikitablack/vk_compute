#include <gpu/utils/submit.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::utils {

auto submit(VkCommandBuffer commandBuffer, VkQueue queue, VkFence fence) noexcept -> std::expected<void, std::string> {
    VkCommandBufferSubmitInfo commandBufferSubmitInfo = vku::InitStructHelper{};
    commandBufferSubmitInfo.commandBuffer = commandBuffer;
    commandBufferSubmitInfo.deviceMask = 0;

    VkSubmitInfo2 submitInfo = vku::InitStructHelper{};
    submitInfo.flags = 0;
    submitInfo.waitSemaphoreInfoCount = 0;
    submitInfo.pWaitSemaphoreInfos = nullptr;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferSubmitInfo;
    submitInfo.signalSemaphoreInfoCount = 0;
    submitInfo.pSignalSemaphoreInfos = nullptr;

    if (vkQueueSubmit2(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
        return std::unexpected{"failed to submit staging command buffer"};
    }

    return {};
}

}  // namespace gpu::utils
