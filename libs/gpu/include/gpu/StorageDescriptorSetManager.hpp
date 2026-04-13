#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>
#include <vector>

namespace gpu {

namespace impl {

[[nodiscard]] auto create_descriptor_pool(VkDevice device,  //
                                          VkDescriptorType descriptorType,  //
                                          uint32_t descriptorCount  //
                                          ) noexcept -> std::expected<VkDescriptorPool, std::string>;

[[nodiscard]] auto allocate_descriptor_set(VkDevice device, VkDescriptorPool descriptorPool,
                                           VkDescriptorSetLayout descriptorSetLayout, uint32_t descriptorCount) noexcept
    -> std::expected<VkDescriptorSet, std::string>;

}  // namespace impl

class StorageDescriptorSetManager {
private:
    struct DescriptorData {
        VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
        VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
        uint32_t descriptorCounter{0};
    };

public:
    StorageDescriptorSetManager() = default;

public:
    [[nodiscard]] auto init(VkDevice device,  //
                            VkPipelineLayout pipelineLayout,  //
                            VkDescriptorSetLayout descriptorSetLayout,  //
                            uint32_t requiredDescriptorCount  //
                            ) noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;

    [[nodiscard]] auto push(VkCommandBuffer commandBuffer,  //
                            VkDescriptorBufferInfo const& bufferInfo  //
                            ) noexcept -> std::expected<uint32_t, std::string>;

    auto reset() noexcept -> void;

private:
    [[nodiscard]] auto createDescriptorData() noexcept -> std::expected<DescriptorData, std::string>;

public:
    static uint32_t constexpr SET_INDEX{0};

private:
    VkDevice m_device{VK_NULL_HANDLE};
    VkPipelineLayout m_pipelineLayout{VK_NULL_HANDLE};
    VkDescriptorSetLayout m_descriptorSetLayout{VK_NULL_HANDLE};
    uint32_t m_maxDescriptorCount{0};
    std::vector<DescriptorData> m_activeDescriptorData{};
    uint32_t m_currDescriptorDataIndex{0};
    VkCommandBuffer m_currBoundCommandBuffer{VK_NULL_HANDLE};
};

}  // namespace gpu
