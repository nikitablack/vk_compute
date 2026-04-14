#include <cstring>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>

namespace gpu {

auto HostVisibleBuffer::init(size_t size,  //
                             bool readback,  //
                             VkBufferUsageFlags usageFlags  //
                             ) noexcept -> std::expected<void, std::string> {
    m_size = size;

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = nullptr;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = nullptr;

    VmaAllocationCreateFlags allocationFlags{};
    if (readback) {
        allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.flags = allocationFlags;
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
        return std::unexpected{"failed to create host-visible buffer"};
    }

    return {};
}

auto HostVisibleBuffer::copyTo(std::span<std::byte const> data,  //
                               size_t offset  //
                               ) noexcept -> std::expected<void, std::string> {
    if (vmaCopyMemoryToAllocation(GpuManager::get().allocator(),  //
                                  data.data(),  //
                                  m_allocation,  //
                                  offset,  //
                                  data.size()) != VK_SUCCESS) {
        return std::unexpected{"failed to copy data to host-visible buffer"};
    }

    return {};
}

auto HostVisibleBuffer::copyFrom(void* dst, size_t size, size_t offset) noexcept -> std::expected<void, std::string> {
    if (vmaCopyAllocationToMemory(GpuManager::get().allocator(), m_allocation, offset, dst, size) != VK_SUCCESS) {
        return std::unexpected{"failed to copy data from host-visible buffer"};
    }

    return {};
}

auto HostVisibleBuffer::destroy() noexcept -> void {
    if (!m_buffer) {
        return;
    }

    vmaDestroyBuffer(GpuManager::get().allocator(), m_buffer, m_allocation);

    m_allocation = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_size = 0;
}

auto HostVisibleBuffer::size() const noexcept -> size_t {
    return m_size;
}
auto HostVisibleBuffer::buffer() const noexcept -> VkBuffer {
    return m_buffer;
}

}  // namespace gpu
