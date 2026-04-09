
#include <fmt/core.h>

#include <gpu/impl/RequiredApiVersion.hpp>
#include <gpu/impl/create_allocator.hpp>

namespace gpu::impl {

auto create_allocator(VkInstance instance,  //
                      VkPhysicalDevice physicalDevice,  //
                      VkDevice device  //
                      ) noexcept -> std::expected<VmaAllocator, std::string> {
    fmt::println("creating vma allocator");

    VmaAllocatorCreateInfo info{};
    info.flags = 0;
    info.physicalDevice = physicalDevice;
    info.device = device;
    info.preferredLargeHeapBlockSize = 0;
    info.pAllocationCallbacks = nullptr;
    info.pDeviceMemoryCallbacks = nullptr;
    info.pHeapSizeLimit = nullptr;
    info.pVulkanFunctions = nullptr;
    info.instance = instance;
    info.vulkanApiVersion = VK_MAKE_API_VERSION(0, RequiredApiVersion::MAJOR, RequiredApiVersion::MINOR, 0);

    VmaAllocator allocator;
    if (vmaCreateAllocator(&info, &allocator) != VK_SUCCESS) {
        return std::unexpected{"failed to create vma allocator"};
    }

    return allocator;
}

}  // namespace gpu::impl
