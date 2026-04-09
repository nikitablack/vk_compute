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
    [[nodiscard]] auto init(VmaAllocator allocator,  //
                            VkBufferUsageFlags2 usageFlags,  //
                            size_t size  //
                            ) noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;

    auto size() const noexcept -> size_t;
    auto buffer() const noexcept -> VkBuffer;

private:
    VmaAllocator m_allocator{VK_NULL_HANDLE};
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
};

}  // namespace gpu
