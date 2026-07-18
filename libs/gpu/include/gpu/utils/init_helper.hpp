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
    InitData() = default;
    InitData(InitData const&) = delete;
    auto operator=(InitData const&) -> InitData& = delete;

    InitData(InitData&& other) noexcept
        : commandBuffer{other.commandBuffer},  //
          stagingBuffer{std::move(other.stagingBuffer)}  //
    {
        other.commandBuffer = VK_NULL_HANDLE;
    }

    auto operator=(InitData&& other) noexcept -> InitData& {
        commandBuffer = other.commandBuffer;
        stagingBuffer = std::move(other.stagingBuffer);

        other.commandBuffer = VK_NULL_HANDLE;

        return *this;
    }

    VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
    HostVisibleBuffer stagingBuffer{};
};

[[nodiscard]] auto copy_to(VkCommandBuffer commandBuffer,  //
                           DeviceBuffer const& dst,  //
                           HostVisibleBuffer const& src,  //
                           VkDeviceSize sizeBytes,  //
                           VkDeviceSize dstOffset = 0,  //
                           VkDeviceSize srcOffset = 0  //
                           ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto init_buffer(VkCommandBuffer commandBuffer,  //
                               DeviceBuffer const& dst,  //
                               HostVisibleBuffer src,  //
                               VkDeviceSize sizeBytes,  //
                               VkDeviceSize dstOffset = 0,  //
                               VkDeviceSize srcOffset = 0  //
                               ) noexcept -> std::expected<InitData, std::string>;

[[nodiscard]] auto init_buffer_sync(DeviceBuffer const& dst,  //
                                    std::span<std::byte const> src,  //
                                    VkDeviceSize dstOffset = 0  //
                                    ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto submit_init_data_sync(std::vector<InitData>&& initData,  //
                                         VulkanQueue const& queue  //
                                         ) noexcept -> std::expected<void, std::string>;

}  // namespace gpu::utils
