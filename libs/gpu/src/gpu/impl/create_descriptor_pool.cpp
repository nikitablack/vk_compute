#include <fmt/core.h>

#include <gpu/impl/allocate_descriptor_set.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto create_descriptor_pool(VkDevice device,  //
                            VkDescriptorType descriptorType,  //
                            uint32_t descriptorCount  //
                            ) noexcept -> std::expected<VkDescriptorPool, std::string> {
    fmt::println("creating descriptor pool");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = descriptorType;
    poolSize.descriptorCount = descriptorCount;

    VkDescriptorPoolCreateInfo info = vku::InitStructHelper{};
    info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    info.maxSets = 1;
    info.poolSizeCount = 1;
    info.pPoolSizes = &poolSize;

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &info, nullptr, &descriptorPool) != VK_SUCCESS) {
        return std::unexpected{"failed to create descriptor pool"};
    }

    return descriptorPool;
}

}  // namespace gpu::impl
