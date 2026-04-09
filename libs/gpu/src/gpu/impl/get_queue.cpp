#include <fmt/core.h>

#include <gpu/impl/get_queue.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu::impl {

auto get_queue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex) noexcept -> VkQueue {
    fmt::println("getting queue");

    VkDeviceQueueInfo2 info = vku::InitStructHelper{};
    info.flags = 0;
    info.queueFamilyIndex = queueFamilyIndex;
    info.queueIndex = queueIndex;

    VkQueue queue{VK_NULL_HANDLE};
    vkGetDeviceQueue2(device, &info, &queue);

    return queue;
}

}  // namespace gpu::impl
