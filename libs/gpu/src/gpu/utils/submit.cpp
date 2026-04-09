#include <gpu/utils/submit.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::utils {

auto submit(VkCommandBuffer commandBuffer, VkQueue queue, VkFence fence) noexcept -> std::expected<void, std::string> {
    VkSubmitInfo submitInfo = vku::InitStructHelper{};
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS) {
        return std::unexpected{"failed to submit"};
    }

    return {};
}

}  // namespace gpu::utils
