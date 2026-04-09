#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto allocate_descriptor_set(VkDevice device,  //
                                           VkDescriptorPool descriptorPool,  //
                                           VkDescriptorSetLayout descriptorSetLayout,  //
                                           uint32_t descriptorCount  //
                                           ) noexcept -> std::expected<VkDescriptorSet, std::string>;

}  // namespace gpu::impl
