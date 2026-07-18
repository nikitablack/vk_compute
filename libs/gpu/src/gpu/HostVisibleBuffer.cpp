#include <cstring>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <utils/try_expected.hpp>

namespace gpu {

HostVisibleBuffer::HostVisibleBuffer(HostVisibleBuffer&& other) noexcept
    : m_buffer{other.m_buffer},  //
      m_size{other.m_size},  //
      m_allocation{other.m_allocation}  //
{
    other.m_buffer = VK_NULL_HANDLE;
    other.m_size = 0;
    other.m_allocation = VK_NULL_HANDLE;
}

auto HostVisibleBuffer::operator=(HostVisibleBuffer&& other) noexcept -> HostVisibleBuffer& {
    m_buffer = other.m_buffer;
    m_size = other.m_size;
    m_allocation = other.m_allocation;

    other.m_buffer = VK_NULL_HANDLE;
    other.m_size = 0;
    other.m_allocation = VK_NULL_HANDLE;

    return *this;
}

auto HostVisibleBuffer::create(size_t sizeBytes,  //
                               bool readback,  //
                               VkBufferUsageFlags2 usageFlags  //
                               ) noexcept -> std::expected<HostVisibleBuffer, std::string> {
    HostVisibleBuffer buffer{};
    TRY_EXPECTED_VOID(buffer.init(sizeBytes, readback, usageFlags));

    return buffer;
}

auto HostVisibleBuffer::init(size_t sizeBytes,  //
                             bool readback,  //
                             VkBufferUsageFlags2 usageFlags  //
                             ) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    m_size = sizeBytes;

    VkBufferUsageFlags2CreateInfo flagsCreateInfo = vku::InitStructHelper{};
    flagsCreateInfo.usage = usageFlags;

    VkBufferCreateInfo bufferCreateInfo = vku::InitStructHelper{&flagsCreateInfo};
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = m_size;
    bufferCreateInfo.usage = 0;
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
    if (vmaCreateBuffer(gpuManager.allocator(),  //
                        &bufferCreateInfo,  //
                        &allocationCreateInfo,  //
                        &m_buffer,  //
                        &m_allocation,  //
                        &allocationInfo) != VK_SUCCESS) {
        return std::unexpected{"failed to create host-visible buffer"};
    }

    return {};
}

auto HostVisibleBuffer::copyTo(std::span<std::byte const> src,  //
                               size_t dstOffset  //
                               ) noexcept -> std::expected<void, std::string> {
    if (!m_allocation) {
        return std::unexpected{"HostVisibleBuffer is not initialized. Did you forget to call init()?"};
    }

    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    if (vmaCopyMemoryToAllocation(gpuManager.allocator(),  //
                                  src.data(),  //
                                  m_allocation,  //
                                  dstOffset,  //
                                  src.size_bytes()) != VK_SUCCESS) {
        return std::unexpected{"failed to copy data to host-visible buffer"};
    }

    return {};
}

auto HostVisibleBuffer::copyFrom(std::span<std::byte> dst,  //
                                 size_t srcOffset  //
                                 ) noexcept -> std::expected<void, std::string> {
    if (!m_allocation) {
        return std::unexpected{"HostVisibleBuffer is not initialized. Did you forget to call init()?"};
    }

    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    if (vmaCopyAllocationToMemory(gpuManager.allocator(), m_allocation, srcOffset, dst.data(), dst.size_bytes()) !=
        VK_SUCCESS) {
        return std::unexpected{"failed to copy data from host-visible buffer"};
    }

    return {};
}

auto HostVisibleBuffer::destroy() noexcept -> void {
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

auto HostVisibleBuffer::size() const noexcept -> size_t {
    return m_size;
}
auto HostVisibleBuffer::buffer() const noexcept -> VkBuffer {
    return m_buffer;
}

}  // namespace gpu
