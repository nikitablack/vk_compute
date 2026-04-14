#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu {

class DeviceBuffer {
public:
    DeviceBuffer() = default;

public:
    [[nodiscard]] auto init(size_t size,  //
                            VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |
                                                             VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
                                                             VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT  //
                            ) noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;

    auto size() const noexcept -> size_t;
    auto buffer() const noexcept -> VkBuffer;

private:
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
};

}  // namespace gpu
