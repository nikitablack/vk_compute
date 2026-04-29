#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu {

auto DeviceBuffer::init(size_t size,  //
                        VkBufferUsageFlags2 usageFlags  //
                        ) noexcept -> std::expected<void, std::string> {
    if (!GpuManager::initialized()) {
        return std::unexpected{"GpuManager is not initialized. Did you forget to call GpuManager::init()?"};
    }

    m_size = size;

    VkBufferUsageFlags2CreateInfo flagsCreateInfo = vku::InitStructHelper{};
    flagsCreateInfo.usage = usageFlags;

    VkBufferCreateInfo bufferCreateInfo = vku::InitStructHelper{&flagsCreateInfo};
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = 0;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = nullptr;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.requiredFlags = 0;
    allocationCreateInfo.preferredFlags = 0;
    allocationCreateInfo.memoryTypeBits = 0;
    allocationCreateInfo.pool = VK_NULL_HANDLE;
    allocationCreateInfo.pUserData = nullptr;

    VmaAllocationInfo allocationInfo{};
    if (vmaCreateBuffer(GpuManager::get().allocator(),  //
                        &bufferCreateInfo,  //
                        &allocationCreateInfo,  //
                        &m_buffer,  //
                        &m_allocation,  //
                        &allocationInfo) != VK_SUCCESS) {
        return std::unexpected{"failed to create device buffer"};
    }

    return {};
}

auto DeviceBuffer::destroy() noexcept -> void {
    if (!m_buffer) {
        return;
    }

    vmaDestroyBuffer(GpuManager::get().allocator(), m_buffer, m_allocation);

    m_allocation = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_size = 0;
}

auto DeviceBuffer::size() const noexcept -> size_t {
    return m_size;
}

auto DeviceBuffer::buffer() const noexcept -> VkBuffer {
    return m_buffer;
}

}  // namespace gpu
