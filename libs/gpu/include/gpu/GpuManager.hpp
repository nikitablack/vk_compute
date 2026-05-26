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
#include <unordered_map>
#include <vector>

namespace gpu {

class GpuManager {
private:
    struct PipelineData {
        std::string name{};
        uint32_t workgroupSizeX{0};
        uint32_t workgroupSizeY{0};
        uint32_t workgroupSizeZ{0};

        auto operator==(PipelineData const& other) const -> bool {
            return name == other.name &&  //
                   workgroupSizeX == other.workgroupSizeX &&  //
                   workgroupSizeY == other.workgroupSizeY &&  //
                   workgroupSizeZ == other.workgroupSizeZ;
        }
    };

    struct PipelineDataHash {
        auto operator()(PipelineData const& p) const -> size_t {
            size_t h{std::hash<std::string>{}(p.name)};

            // Combine hashes (classic hash combine)
            auto hashCombine{
                [](size_t& seed, size_t value) { seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }};

            hashCombine(h, std::hash<uint32_t>{}(p.workgroupSizeX));
            hashCombine(h, std::hash<uint32_t>{}(p.workgroupSizeY));
            hashCombine(h, std::hash<uint32_t>{}(p.workgroupSizeZ));

            return h;
        }
    };

public:
    static auto get() noexcept -> GpuManager&;
    [[nodiscard]] static auto init() noexcept -> std::expected<void, std::string>;
    static auto initialized() noexcept -> bool;
    static auto destroy() noexcept -> void;

    GpuManager(const GpuManager&) = delete;
    GpuManager& operator=(const GpuManager&) = delete;
    GpuManager(GpuManager&&) = delete;
    GpuManager& operator=(GpuManager&&) = delete;

private:
    GpuManager() = default;
    [[nodiscard]] auto initialize() noexcept -> std::expected<void, std::string>;

public:
    auto addPipeline(VkPipeline pipeline,  //
                     std::string const& name,  //
                     uint32_t workgroupSizeX,  //
                     uint32_t workgroupSizeY,  //
                     uint32_t workgroupSizeZ  //
                     ) noexcept -> void;

    auto getPipeline(std::string const& name,  //
                     uint32_t workgroupSizeX,  //
                     uint32_t workgroupSizeY,  //
                     uint32_t workgroupSizeZ  //
                     ) noexcept -> std::optional<VkPipeline>;

    auto destroyImpl() noexcept -> void;
    auto flush() const noexcept -> void;
    auto allocator() const noexcept -> VmaAllocator;
    auto commandManager() noexcept -> CommandManager&;
    auto computeQueue() const noexcept -> VulkanQueue;
    // auto debugUtils() const noexcept -> VulkanDebugUtils const&;
    auto device() const noexcept -> VkDevice;
    auto pipelineLayout() const noexcept -> VkPipelineLayout;
    auto physicalDeviceProperties() const noexcept -> VkPhysicalDeviceProperties2 const&;
    auto storageDescriptorSetManager() noexcept -> StorageDescriptorSetManager&;

private:
    static bool m_initialized;

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
    std::unordered_map<PipelineData, VkPipeline, PipelineDataHash> m_dataToPipeline{};
};

}  // namespace gpu
