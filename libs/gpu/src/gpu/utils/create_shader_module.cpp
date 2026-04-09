#include <cstring>
#include <gpu/utils/create_shader_module.hpp>
#include <vector>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::utils {

auto create_shader_module(VkDevice device,  //
                          std::span<std::byte const> shader  //
                          ) noexcept -> std::expected<VkShaderModule, std::string> {
    if (shader.size() == 0 || (shader.size() % 4 != 0)) {
        return std::unexpected{"shader data size should be a nonzero multiple of 4"};
    }

    std::vector<uint32_t> spirv(shader.size() / 4);
    memcpy(spirv.data(), shader.data(), shader.size());

    VkShaderModuleCreateInfo info = vku::InitStructHelper{};
    info.flags = 0;
    info.codeSize = shader.size();
    info.pCode = spirv.data();

    VkShaderModule shaderModule{VK_NULL_HANDLE};
    if (vkCreateShaderModule(device, &info, nullptr, &shaderModule) != VK_SUCCESS) {
        return std::unexpected{"failed to create shader module"};
    }

    return shaderModule;
}

}  // namespace gpu::utils
