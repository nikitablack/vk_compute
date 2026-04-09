#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto create_descriptor_pool(VkDevice device,  //
                                          VkDescriptorType descriptorType,  //
                                          uint32_t descriptorCount  //
                                          ) noexcept -> std::expected<VkDescriptorPool, std::string>;

}  // namespace gpu::impl
