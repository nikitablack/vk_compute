#include <fmt/core.h>

#include <gpu/StorageDescriptorSetManager.hpp>
#include <gpu/impl/allocate_descriptor_set.hpp>
#include <gpu/impl/create_descriptor_pool.hpp>
#include <utils/try_expected.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu {

auto StorageDescriptorSetManager::init(VkDevice device,  //
                                       VkPipelineLayout pipelineLayout,  //
                                       VkDescriptorSetLayout descriptorSetLayout,  //
                                       uint32_t requiredDescriptorCount  //
                                       ) noexcept -> std::expected<void, std::string> {
    m_device = device;
    m_pipelineLayout = pipelineLayout;

    m_descriptorSetLayout = descriptorSetLayout;
    m_maxDescriptorCount = requiredDescriptorCount;

    TRY_EXPECTED(DescriptorData descriptorData, createDescriptorData());
    m_activeDescriptorData.push_back(std::move(descriptorData));

    m_currDescriptorDataIndex = 0;

    return {};
}

auto StorageDescriptorSetManager::destroy() noexcept -> void {
    if (!m_device) {
        return;
    }

    for (auto& descriptorData : m_activeDescriptorData) {
        vkDestroyDescriptorPool(m_device, descriptorData.descriptorPool, nullptr);
        descriptorData.descriptorPool = VK_NULL_HANDLE;
    }
    m_activeDescriptorData.clear();

    m_device = VK_NULL_HANDLE;
    m_pipelineLayout = VK_NULL_HANDLE;
    m_descriptorSetLayout = VK_NULL_HANDLE;
    m_maxDescriptorCount = 0;
    m_currDescriptorDataIndex = 0;
}

auto StorageDescriptorSetManager::push(VkCommandBuffer commandBuffer,  //
                                       VkDescriptorBufferInfo const& bufferInfo  //
                                       ) noexcept -> std::expected<uint32_t, std::string> {
    if (!m_device) {
        return std::unexpected{"DescriptorSetManager is not initialized"};
    }

    // check if there is enough space in the current descriptor set
    // if not, move to the next, creating a new one if necessary
    if (m_currDescriptorDataIndex < m_activeDescriptorData.size()) {
        auto& dd{m_activeDescriptorData[m_currDescriptorDataIndex]};

        if ((++dd.descriptorCounter) >= m_maxDescriptorCount) {
            ++m_currDescriptorDataIndex;
        }
    }

    // create new descriptor set if necessary
    if (m_currDescriptorDataIndex >= m_activeDescriptorData.size()) {
        fmt::println("binding descriptor set {}", m_currDescriptorDataIndex);

        TRY_EXPECTED(DescriptorData newDescriptorData, createDescriptorData());
        m_activeDescriptorData.push_back(std::move(newDescriptorData));

        vkCmdBindDescriptorSets(commandBuffer,  //
                                VK_PIPELINE_BIND_POINT_GRAPHICS,  //
                                m_pipelineLayout,  //
                                SET_INDEX,  //
                                1,  //
                                &newDescriptorData.descriptorSet,  //
                                0,  //
                                nullptr);
    }

    auto& descriptorData{m_activeDescriptorData[m_currDescriptorDataIndex]};

    VkWriteDescriptorSet writeDescriptorSet = vku::InitStructHelper{};
    writeDescriptorSet.dstSet = descriptorData.descriptorSet;
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.dstArrayElement = descriptorData.descriptorCounter;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSet.pImageInfo = nullptr;
    writeDescriptorSet.pBufferInfo = &bufferInfo;
    writeDescriptorSet.pTexelBufferView = nullptr;

    vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSet, 0, nullptr);

    auto const descriptorIndex{descriptorData.descriptorCounter};
    ++descriptorData.descriptorCounter;

    return descriptorIndex;
}

auto StorageDescriptorSetManager::createDescriptorData() noexcept -> std::expected<DescriptorData, std::string> {
    DescriptorData descriptorData{};
    descriptorData.descriptorCounter = 0;

    TRY_EXPECTED(descriptorData.descriptorPool,
                 impl::create_descriptor_pool(m_device, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_maxDescriptorCount));

    TRY_EXPECTED(descriptorData.descriptorSet,
                 impl::allocate_descriptor_set(m_device,  //
                                               descriptorData.descriptorPool,  //
                                               m_descriptorSetLayout,  //
                                               m_maxDescriptorCount));

    return descriptorData;
}

}  // namespace gpu
