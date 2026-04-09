#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>
#include <vector>

namespace gpu {

class CommandManager {
public:
    CommandManager() = default;

public:
    [[nodiscard]] auto init(VkDevice device, uint32_t queueFamily) noexcept -> std::expected<void, std::string>;
    auto destroy() noexcept -> void;

    [[nodiscard]] auto commandBuffer() noexcept -> std::expected<VkCommandBuffer, std::string>;
    [[nodiscard]] auto commandBufferBegin() noexcept -> std::expected<VkCommandBuffer, std::string>;
    [[nodiscard]] auto resetCommandBuffer(VkCommandBuffer const commandBuffer) noexcept
        -> std::expected<void, std::string>;

private:
    VkDevice m_device{VK_NULL_HANDLE};
    VkCommandPool m_commandPool{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> m_availableCommandBuffers{};
};

}  // namespace gpu
