#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <gpu/CommandManager.hpp>
#include <gpu/ImmediateDataBufferManager.hpp>
#include <gpu/StorageDescriptorSetManager.hpp>
#include <gpu/VulkanDebugUtils.hpp>
#include <gpu/VulkanQueue.hpp>
#include <string>
#include <vector>

namespace gpu {

class GpuManager {
public:
    GpuManager() = default;

public:
    [[nodiscard]] auto initialize() noexcept -> std::expected<void, std::string>;
    auto destroy() noexcept -> void;
    auto flush() const noexcept -> void;

    auto allocator() const noexcept -> VmaAllocator;
    auto commandManager() noexcept -> CommandManager&;
    auto computeQueue() const noexcept -> VulkanQueue;
    // auto debugUtils() const noexcept -> VulkanDebugUtils const&;
    auto device() const noexcept -> VkDevice;
    // auto immediateDataBufferManager() noexcept -> ImmediateDataBufferManager&;
    auto pipelineLayout() const noexcept -> VkPipelineLayout;
    auto storageDescriptorSetManager() noexcept -> StorageDescriptorSetManager&;

private:
    VkInstance m_instance{VK_NULL_HANDLE};
    VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
    VkPhysicalDeviceProperties2 m_physicalDeviceProperties{};
    VkDevice m_device{VK_NULL_HANDLE};
    VulkanDebugUtils m_debugUtils{};
    VulkanQueue m_computeQueue{};
    VmaAllocator m_allocator{VK_NULL_HANDLE};
    CommandManager m_commandManager{};
    VkDescriptorSetLayout m_storageDescriptorSetLayout{VK_NULL_HANDLE};
    VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
    StorageDescriptorSetManager m_storageDescriptorSetManager{};
    ImmediateDataBufferManager m_immediateDataBufferManager{};
};

}  // namespace gpu
