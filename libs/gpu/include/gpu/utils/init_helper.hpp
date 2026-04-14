#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <gpu/HostVisibleBuffer.hpp>
#include <span>
#include <string>
#include <vector>

// forward declarations
namespace gpu {

class DeviceBuffer;
struct VulkanQueue;

}  // namespace gpu

namespace gpu::utils {

struct InitData {
    VkCommandBuffer commandBuffer;
    HostVisibleBuffer stagingBuffer;
};

[[nodiscard]] auto copy_to(VkCommandBuffer commandBuffer,  //
                           DeviceBuffer const& deviceBuffer,  //
                           HostVisibleBuffer const& stagingBuffer,  //
                           VkDeviceSize dstBufferOffset = 0,  //
                           VkDeviceSize stagingBufferOffset = 0  //
                           ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto init_buffer(VkCommandBuffer commandBuffer,  //
                               DeviceBuffer const& deviceBuffer,  //
                               HostVisibleBuffer const& stagingBuffer,  //
                               VkDeviceSize dstBufferOffset = 0,  //
                               VkDeviceSize stagingBufferOffset = 0  //
                               ) noexcept -> std::expected<InitData, std::string>;

[[nodiscard]] auto init_buffer_sync(DeviceBuffer const& deviceBuffer,  //
                                    std::span<std::byte const> data  //
                                    ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto submit_init_data_sync(std::vector<InitData>&& initData,  //
                                         VulkanQueue const& queue  //
                                         ) noexcept -> std::expected<void, std::string>;

}  // namespace gpu::utils
