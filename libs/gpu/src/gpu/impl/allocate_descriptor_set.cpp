#include <fmt/core.h>

#include <gpu/impl/allocate_descriptor_set.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto allocate_descriptor_set(VkDevice device,  //
                             VkDescriptorPool descriptorPool,  //
                             VkDescriptorSetLayout descriptorSetLayout,  //
                             uint32_t descriptorCount  //
                             ) noexcept -> std::expected<VkDescriptorSet, std::string> {
    fmt::println("allocating descriptor set");

    VkDescriptorSetVariableDescriptorCountAllocateInfo allocateInfo = vku::InitStructHelper{};
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pDescriptorCounts = &descriptorCount;

    VkDescriptorSetAllocateInfo info = vku::InitStructHelper{&allocateInfo};
    info.descriptorPool = descriptorPool;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet{VK_NULL_HANDLE};
    if (vkAllocateDescriptorSets(device, &info, &descriptorSet) != VK_SUCCESS) {
        return std::unexpected{"failed to allocate descriptor set"};
    }

    return descriptorSet;
}

}  // namespace gpu::impl
