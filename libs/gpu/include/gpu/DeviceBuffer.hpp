#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <gpu/utils/init_helper.hpp>
#include <span>
#include <string>
#include <utils/try_expected.hpp>

namespace gpu {

class DeviceBuffer {
public:
    DeviceBuffer() = default;

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept;
    auto operator=(DeviceBuffer&& other) noexcept -> DeviceBuffer&;

public:
    template <typename T>
    [[nodiscard]] auto static create(std::span<T const> initialData,  //
                                     VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |  //
                                                                      VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |  //
                                                                      VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT  //
                                     ) noexcept -> std::expected<DeviceBuffer, std::string> {
        TRY_EXPECTED(auto buffer, create(initialData.size_bytes(), usageFlags));

        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(buffer, std::as_bytes(initialData)));

        return buffer;
    }

    [[nodiscard]] auto static create(size_t sizeBytes,  //
                                     VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |  //
                                                                      VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |  //
                                                                      VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT  //
                                     ) noexcept -> std::expected<DeviceBuffer, std::string>;

    auto destroy() noexcept -> void;
    auto size() const noexcept -> size_t;
    auto buffer() const noexcept -> VkBuffer;

private:
    [[nodiscard]] auto init(size_t sizeBytes,  //
                            VkBufferUsageFlags2 usageFlags = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |
                                                             VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
                                                             VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT  //
                            ) noexcept -> std::expected<void, std::string>;

private:
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
};

}  // namespace gpu
