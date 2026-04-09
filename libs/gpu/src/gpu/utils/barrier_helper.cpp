#include <gpu/DeviceBuffer.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::utils {

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        VkBuffer buffer,  //
                        VkDeviceSize offset,  //
                        VkDeviceSize size,  //
                        VkPipelineStageFlags2 srcStageMask,  //
                        VkAccessFlags2 srcAccessMask,  //
                        VkPipelineStageFlags2 dstStageMask,  //
                        VkAccessFlags2 dstAccessMask  //
                        ) noexcept -> void {
    VkBufferMemoryBarrier2 barrier = vku::InitStructHelper{};
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = offset;
    barrier.size = size;

    VkDependencyInfo dependencyInfo = vku::InitStructHelper{};
    dependencyInfo.dependencyFlags = 0;
    dependencyInfo.memoryBarrierCount = 0;
    dependencyInfo.pMemoryBarriers = nullptr;
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers = &barrier;
    dependencyInfo.imageMemoryBarrierCount = 0;
    dependencyInfo.pImageMemoryBarriers = nullptr;

    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        DeviceBuffer const& deviceBuffer,  //
                        VkPipelineStageFlags2 srcStageMask,  //
                        VkAccessFlags2 srcAccessMask,  //
                        VkPipelineStageFlags2 dstStageMask,  //
                        VkAccessFlags2 dstAccessMask  //
                        ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer.buffer(),  //
                       0,  //
                       deviceBuffer.size(),  //
                       srcStageMask,  //
                       srcAccessMask,  //
                       dstStageMask,  //
                       dstAccessMask);
}

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        HostVisibleBuffer const& stagingBuffer,  //
                        VkPipelineStageFlags srcStageMask,  //
                        VkAccessFlags srcAccessMask,  //
                        VkPipelineStageFlags dstStageMask,  //
                        VkAccessFlags dstAccessMask  //
                        ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       stagingBuffer.buffer(),  //
                       0,  //
                       stagingBuffer.size(),  //
                       srcStageMask,  //
                       srcAccessMask,  //
                       dstStageMask,  //
                       dstAccessMask);
}

auto before_write(VkCommandBuffer commandBuffer,  //
                  DeviceBuffer const& deviceBuffer  //
                  ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer,  //
                       VK_PIPELINE_STAGE_2_NONE,  //
                       VK_ACCESS_2_NONE,  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

auto after_write(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer const& deviceBuffer  //
                 ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer,  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_WRITE_BIT,  //
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
}

auto before_read(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer const& deviceBuffer  //
                 ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       deviceBuffer,  //
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

auto before_write(VkCommandBuffer commandBuffer,  //
                  HostVisibleBuffer const& stagingBuffer  //
                  ) noexcept -> void {
    set_buffer_barrier(commandBuffer,  //
                       stagingBuffer,  //
                       VK_PIPELINE_STAGE_2_NONE,  //
                       VK_ACCESS_2_NONE,  //
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                       VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

}  // namespace gpu::utils
