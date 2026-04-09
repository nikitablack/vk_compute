#pragma once

#include <vulkan/vulkan.h>

// forward declaration
namespace gpu {

class DeviceBuffer;
class HostVisibleBuffer;

}  // namespace gpu

namespace gpu::utils {

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        VkBuffer buffer,  //
                        VkDeviceSize offset,  //
                        VkDeviceSize size,  //
                        VkPipelineStageFlags2 srcStageMask,  //
                        VkAccessFlags2 srcAccessMask,  //
                        VkPipelineStageFlags2 dstStageMask,  //
                        VkAccessFlags2 dstAccessMask  //
                        ) noexcept -> void;

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        DeviceBuffer const& deviceBuffer,  //
                        VkPipelineStageFlags2 srcStageMask,  //
                        VkAccessFlags2 srcAccessMask,  //
                        VkPipelineStageFlags2 dstStageMask,  //
                        VkAccessFlags2 dstAccessMask  //
                        ) noexcept -> void;

auto set_buffer_barrier(VkCommandBuffer commandBuffer,  //
                        HostVisibleBuffer const& stagingBuffer,  //
                        VkPipelineStageFlags srcStageMask,  //
                        VkAccessFlags srcAccessMask,  //
                        VkPipelineStageFlags dstStageMask,  //
                        VkAccessFlags dstAccessMask  //
                        ) noexcept -> void;

auto before_write(VkCommandBuffer commandBuffer,  //
                  DeviceBuffer const& deviceBuffer  //
                  ) noexcept -> void;

auto after_write(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer const& deviceBuffer  //
                 ) noexcept -> void;

auto before_read(VkCommandBuffer commandBuffer,  //
                 DeviceBuffer const& deviceBuffer  //
                 ) noexcept -> void;

auto before_write(VkCommandBuffer commandBuffer,  //
                  HostVisibleBuffer const& stagingBuffer  //
                  ) noexcept -> void;

}  // namespace gpu::utils
