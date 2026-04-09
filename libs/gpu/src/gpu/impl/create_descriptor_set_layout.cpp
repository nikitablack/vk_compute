#include <fmt/core.h>

#include <gpu/impl/create_descriptor_set_layout.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto create_descriptor_set_layout(VkDevice device,  //
                                  VkDescriptorType descriptorType,  //
                                  uint32_t requiredDescriptorCount  //
                                  ) noexcept -> std::expected<VkDescriptorSetLayout, std::string> {
    fmt::println("creating descriptor set layout");

    VkDescriptorBindingFlags bindingFlags{VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
                                          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                          VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT};

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo = vku::InitStructHelper{};
    flagsInfo.bindingCount = 1;
    flagsInfo.pBindingFlags = &bindingFlags;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = descriptorType;
    binding.descriptorCount = 1;  // set it to 1 before finding out the real supported number of descriptors
    binding.stageFlags = VK_SHADER_STAGE_ALL;
    binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo info = vku::InitStructHelper{&flagsInfo};
    info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    info.bindingCount = 1;
    info.pBindings = &binding;

    // check layout support and the maximum number of variable-sized descriptors that can be bound
    VkDescriptorSetVariableDescriptorCountLayoutSupport variableDescriptorSupport = vku::InitStructHelper{};
    variableDescriptorSupport.maxVariableDescriptorCount = 0;

    VkDescriptorSetLayoutSupport support = vku::InitStructHelper{&variableDescriptorSupport};
    support.supported = VK_FALSE;

    vkGetDescriptorSetLayoutSupport(device, &info, &support);

    if (support.supported == VK_FALSE) {
        return std::unexpected{"descriptor set layout is not supported"};
    }

    if (variableDescriptorSupport.maxVariableDescriptorCount < requiredDescriptorCount) {
        return std::unexpected{
            fmt::format("descriptor set layout max descriptor count is too low (max: {}, required: {})",
                        variableDescriptorSupport.maxVariableDescriptorCount, requiredDescriptorCount)};
    }

    // now update the binding with the known descriptor count
    binding.descriptorCount = requiredDescriptorCount;

    VkDescriptorSetLayout layout{};
    if (vkCreateDescriptorSetLayout(device, &info, nullptr, &layout) != VK_SUCCESS) {
        return std::unexpected{"failed to create descriptor set layout"};
    }

    return layout;
}

}  // namespace gpu::impl
