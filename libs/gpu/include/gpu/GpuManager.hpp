#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <gpu/CommandManager.hpp>
#include <gpu/StorageDescriptorSetManager.hpp>
#include <gpu/VulkanDebugUtils.hpp>
#include <gpu/VulkanQueue.hpp>
#include <optional>
#include <string>
#include <vector>

namespace gpu {

class GpuManager {
public:
    [[nodiscard]] static auto get() noexcept -> std::expected<std::reference_wrapper<GpuManager>, std::string>;
    static auto destroy() noexcept -> void;

    GpuManager(const GpuManager&) = delete;
    GpuManager& operator=(const GpuManager&) = delete;
    GpuManager(GpuManager&&) = delete;
    GpuManager& operator=(GpuManager&&) = delete;

private:
    GpuManager() = default;
    [[nodiscard]] auto initialize() noexcept -> std::expected<void, std::string>;

public:
    auto destroyImpl() noexcept -> void;
    auto flush() const noexcept -> void;
    auto allocator() const noexcept -> VmaAllocator;
    auto commandManager() noexcept -> CommandManager&;
    auto computeQueue() const noexcept -> VulkanQueue;
    // auto debugUtils() const noexcept -> VulkanDebugUtils const&;
    auto device() const noexcept -> VkDevice;
    auto pipelineLayout() const noexcept -> VkPipelineLayout;
    auto physicalDeviceProperties() const noexcept -> VkPhysicalDeviceProperties2 const&;
    auto physicalDeviceSubgroupProperties() const noexcept -> VkPhysicalDeviceSubgroupProperties const&;
    auto storageDescriptorSetManager() noexcept -> StorageDescriptorSetManager&;

    static bool m_initialized;

private:
    VkInstance m_instance{VK_NULL_HANDLE};
    VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
    VkPhysicalDeviceProperties2 m_physicalDeviceProperties{};
    VkPhysicalDeviceSubgroupProperties m_physicalDeviceSubgroupProperties{};
    VkDevice m_device{VK_NULL_HANDLE};
    VulkanDebugUtils m_debugUtils{};
    VulkanQueue m_computeQueue{};
    VmaAllocator m_allocator{VK_NULL_HANDLE};
    CommandManager m_commandManager{};
    VkDescriptorSetLayout m_storageDescriptorSetLayout{VK_NULL_HANDLE};
    VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
    StorageDescriptorSetManager m_storageDescriptorSetManager{};
};

}  // namespace gpu
