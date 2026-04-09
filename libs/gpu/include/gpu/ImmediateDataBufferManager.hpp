#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/StorageDescriptorSetManager.hpp>
#include <span>
#include <string>
#include <vector>

namespace gpu {

class ImmediateDataBufferManager {
public:
    struct PushDataResult {
        uint64_t startOffset;
        VkDeviceSize size;
        VkBuffer buffer;
    };

private:
    struct OccupancyInfo {
        HostVisibleBuffer vulkanBuffer;
        size_t occupied;
    };

public:
    ImmediateDataBufferManager() = default;

public:
    [[nodiscard]] auto init(VmaAllocator allocator,  //
                            VkPhysicalDeviceProperties2 const& deviceProperties  //
                            ) noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;
    auto reset() noexcept -> void;

    [[nodiscard]] auto pushData(VkCommandBuffer const commandBuffer,  //
                                std::span<std::byte const> data,  //
                                StorageDescriptorSetManager& storageDescriptorSetManager  //
                                ) noexcept -> std::expected<uint32_t, std::string>;

private:
    [[nodiscard]] auto pushDataImpl(std::span<std::byte const> data) noexcept
        -> std::expected<PushDataResult, std::string>;

private:
    VmaAllocator m_allocator{VK_NULL_HANDLE};
    std::vector<OccupancyInfo> m_occupancyInfos{};
    uint64_t m_minStorageBufferOffsetAlignment{0};
};

}  // namespace gpu
