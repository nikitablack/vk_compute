#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <span>
#include <string>
#include <utils/try_expected.hpp>

namespace gpu {

class HostVisibleBuffer {
public:
    HostVisibleBuffer() = default;
    HostVisibleBuffer(HostVisibleBuffer const&) = delete;
    HostVisibleBuffer& operator=(HostVisibleBuffer const&) = delete;

    HostVisibleBuffer(HostVisibleBuffer&& other) noexcept;
    auto operator=(HostVisibleBuffer&& other) noexcept -> HostVisibleBuffer&;

public:
    template <typename T>
    [[nodiscard]] auto static create(std::span<T const> initialData,  //
                                     bool readback = false,  //
                                     VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |  //
                                                                      VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT  //
                                     ) noexcept -> std::expected<HostVisibleBuffer, std::string> {
        TRY_EXPECTED(auto buffer, create(initialData.size_bytes(), readback, usageFlags));

        TRY_EXPECTED_VOID(buffer.copyTo(initialData));

        return buffer;
    }

    [[nodiscard]] static auto create(size_t sizeBytes,  //
                                     bool readback = false,  //
                                     VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
                                                                      VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT  //
                                     ) noexcept -> std::expected<HostVisibleBuffer, std::string>;

    template <typename T>
    [[nodiscard]] auto copyTo(std::span<T const> src,  //
                              size_t dstOffset = 0  //
                              ) noexcept -> std::expected<void, std::string> {
        return copyTo(std::as_bytes(src), dstOffset);
    }

    [[nodiscard]] auto copyTo(std::span<std::byte const> src,  //
                              size_t dstOffset = 0  //
                              ) noexcept -> std::expected<void, std::string>;

    template <typename T>
    [[nodiscard]] auto copyFrom(std::span<T> dst,  //
                                size_t srcOffset = 0  //
                                ) noexcept -> std::expected<void, std::string> {
        return copyFrom(std::as_writable_bytes(dst), srcOffset);
    }

    [[nodiscard]] auto copyFrom(std::span<std::byte> dst,  //
                                size_t srcOffset = 0  //
                                ) noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;
    auto size() const noexcept -> size_t;
    auto buffer() const noexcept -> VkBuffer;

private:
    [[nodiscard]] auto init(size_t sizeBytes,  //
                            bool readback = false,  //
                            VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
                                                             VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT  //
                            ) noexcept -> std::expected<void, std::string>;

private:
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
};

}  // namespace gpu
