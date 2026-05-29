#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <utils/try_expected.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace gpu {

auto DeviceBuffer::init(size_t size,  //
                        VkBufferUsageFlags2 usageFlags  //
                        ) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

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
    if (vmaCreateBuffer(gpuManager.allocator(),  //
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

    auto result{GpuManager::get()};
    // should never happen, since if m_buffer exists meanss GpuManager is initizlized
    if (!result) {
        return;
    }

    auto& gpuManager{result.value().get()};

    vmaDestroyBuffer(gpuManager.allocator(), m_buffer, m_allocation);

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
