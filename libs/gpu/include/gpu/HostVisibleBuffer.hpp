#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <span>
#include <string>

namespace gpu {

class HostVisibleBuffer {
public:
    HostVisibleBuffer() = default;

public:
    [[nodiscard]] auto init(VmaAllocator allocator,  //
                            VkBufferUsageFlags usageFlags,  //
                            size_t size,  //
                            bool readback = false  //
                            ) noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;

    [[nodiscard]] auto copyTo(std::span<std::byte const> data,  //
                              size_t offset = 0  //
                              ) noexcept -> std::expected<void, std::string>;

    [[nodiscard]] auto copyFrom(void* dst, size_t size, size_t offset) noexcept -> std::expected<void, std::string>;
    auto size() const noexcept -> size_t;
    auto buffer() const noexcept -> VkBuffer;

private:
    VmaAllocator m_allocator{VK_NULL_HANDLE};
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
};

}  // namespace gpu
