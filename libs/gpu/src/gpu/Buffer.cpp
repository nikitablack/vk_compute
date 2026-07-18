#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <gpu/Buffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <utils/ScopeGuard.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace {

auto copy_memory_to_device_buffer(gpu::Buffer const& dst,  //
                                  std::span<std::byte const> src,  //
                                  VkDeviceSize dstOffset  //
                                  ) noexcept -> std::expected<void, std::string> {
    if (dst.type() != gpu::Buffer::Type::Device) {
        return std::unexpected{fmt::format("wrong buffer type, expected {}, provided {}",  //
                                           static_cast<uint32_t>(gpu::Buffer::Type::Device),  //
                                           static_cast<uint32_t>(dst.type()))};
    }

    if (dst.size() < (dstOffset + src.size_bytes())) {
        return std::unexpected{"buffer is too small"};
    }

    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    TRY_EXPECTED(auto stagingBuffer, gpu::Buffer::create(src, gpu::Buffer::Type::Upload));

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();

        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }

        stagingBuffer.destroy();
    })};

    // barriers before copy
    {
        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       stagingBuffer.buffer(),  //
                                       0,  //
                                       stagingBuffer.size(),  //
                                       VK_PIPELINE_STAGE_2_NONE,  //
                                       VK_ACCESS_2_NONE,  //
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                       VK_ACCESS_2_TRANSFER_READ_BIT);

        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       dst.buffer(),  //
                                       dstOffset,  //
                                       src.size(),  //
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,  //
                                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,  //
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                       VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }

    // copy
    {
        VkBufferCopy2 region = vku::InitStructHelper{};
        region.srcOffset = 0;
        region.dstOffset = dstOffset;
        region.size = stagingBuffer.size();

        VkCopyBufferInfo2 copyInfo = vku::InitStructHelper{};
        copyInfo.srcBuffer = stagingBuffer.buffer();
        copyInfo.dstBuffer = dst.buffer();
        copyInfo.regionCount = 1;
        copyInfo.pRegions = &region;

        vkCmdCopyBuffer2(commandBuffer, &copyInfo);
    }

    // barrier after copy
    {
        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       dst.buffer(),  //
                                       dstOffset,  //
                                       src.size(),  //
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                       VK_ACCESS_2_TRANSFER_WRITE_BIT,  //
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,  //
                                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    // submit, wait and cleanup
    {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            return std::unexpected{"failed to end copy command buffer"};
        }

        auto const queue{gpuManager.computeQueue()};

        TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer,  //
                                             queue.queue,  //
                                             VK_NULL_HANDLE));

        if (vkQueueWaitIdle(queue.queue) != VK_SUCCESS) {
            return std::unexpected{"failed to wait staging queue"};
        }

        stagingBuffer.destroy();
    }

    return {};
}

auto copy_device_buffer_to_memory(std::span<std::byte const> dst,  //
                                  gpu::Buffer const& src,  //
                                  VkDeviceSize srcOffset  //
                                  ) noexcept -> std::expected<void, std::string> {
    if (src.type() != gpu::Buffer::Type::Device) {
        return std::unexpected{fmt::format("wrong buffer type, expected {}, provided {}",  //
                                           static_cast<uint32_t>(gpu::Buffer::Type::Device),  //
                                           static_cast<uint32_t>(src.type()))};
    }

    if (src.size() < (srcOffset + dst.size_bytes())) {
        return std::unexpected{"buffer is too small"};
    }

    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    TRY_EXPECTED(auto stagingBuffer, gpu::Buffer::create(dst.size(), gpu::Buffer::Type::Readback));

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();
        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }
    })};

    // barriers before copy
    {
        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       stagingBuffer.buffer(),  //
                                       0,  //
                                       stagingBuffer.size(),  //
                                       VK_PIPELINE_STAGE_2_NONE,  //
                                       VK_ACCESS_2_NONE,  //
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                       VK_ACCESS_2_TRANSFER_WRITE_BIT);

        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       src.buffer(),  //
                                       srcOffset,  //
                                       dst.size_bytes(),  //
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,  //
                                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,  //
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                       VK_ACCESS_2_TRANSFER_READ_BIT);
    }

    // copy
    {
        VkBufferCopy2 region = vku::InitStructHelper{};
        region.srcOffset = srcOffset;
        region.dstOffset = 0;
        region.size = stagingBuffer.size();

        VkCopyBufferInfo2 copyInfo = vku::InitStructHelper{};
        copyInfo.srcBuffer = src.buffer();
        copyInfo.dstBuffer = stagingBuffer.buffer();
        copyInfo.regionCount = 1;
        copyInfo.pRegions = &region;

        vkCmdCopyBuffer2(commandBuffer, &copyInfo);
    }

    // barrier after copy
    {
        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       src.buffer(),  //
                                       srcOffset,  //
                                       dst.size_bytes(),  //
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                       VK_ACCESS_2_TRANSFER_WRITE_BIT,  //
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR,  //
                                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    }

    // submit, wait and cleanup
    {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            return std::unexpected{"failed to end copy command buffer"};
        }

        auto const queue{gpuManager.computeQueue()};

        TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer,  //
                                             queue.queue,  //
                                             VK_NULL_HANDLE));

        if (vkQueueWaitIdle(queue.queue) != VK_SUCCESS) {
            return std::unexpected{"failed to wait staging queue"};
        }

        stagingBuffer.destroy();
    }

    return {};
}

}  // namespace

namespace gpu {

Buffer::Buffer(Buffer&& other) noexcept
    : m_buffer{other.m_buffer},  //
      m_size{other.m_size},  //
      m_allocation{other.m_allocation},  //
      m_type{other.m_type}  //
{
    other.m_buffer = VK_NULL_HANDLE;
    other.m_size = 0;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_type = Type::None;
}

auto Buffer::operator=(Buffer&& other) noexcept -> Buffer& {
    m_buffer = other.m_buffer;
    m_size = other.m_size;
    m_allocation = other.m_allocation;
    m_type = other.m_type;

    other.m_buffer = VK_NULL_HANDLE;
    other.m_size = 0;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_type = Type::None;

    return *this;
}

auto Buffer::create(size_t sizeBytes,  //
                    Type type  //
                    ) noexcept -> std::expected<Buffer, std::string> {
    Buffer buffer{};
    TRY_EXPECTED_VOID(buffer.init(sizeBytes, type));

    return buffer;
}

auto Buffer::copyToBuffer(gpu::BufferSpan const& src,  //
                          size_t dstOffset  //
                          ) noexcept -> std::expected<void, std::string> {
    size_t const sizeBytes{src.size()};

    if (m_size < (dstOffset + sizeBytes)) {
        return std::unexpected{"buffer size is too small"};
    }

    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();
        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }
    })};

    gpu::utils::set_buffer_barrier(commandBuffer,  //
                                   src.buffer().buffer(),  //
                                   src.offset(),  //
                                   sizeBytes,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,  //
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                   VK_ACCESS_2_TRANSFER_READ_BIT);

    gpu::utils::set_buffer_barrier(commandBuffer,  //
                                   m_buffer,  //
                                   dstOffset,  //
                                   sizeBytes,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,  //
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferCopy region{};
    region.srcOffset = src.offset();
    region.dstOffset = dstOffset;
    region.size = sizeBytes;

    vkCmdCopyBuffer(commandBuffer,  //
                    src.buffer().buffer(),  //
                    m_buffer,  //
                    1,  //
                    &region);

    gpu::utils::set_buffer_barrier(commandBuffer,  //
                                   src.buffer().buffer(),  //
                                   src.offset(),  //
                                   sizeBytes,  //
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                   VK_ACCESS_2_TRANSFER_READ_BIT,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    gpu::utils::set_buffer_barrier(commandBuffer,  //
                                   m_buffer,  //
                                   dstOffset,  //
                                   sizeBytes,  //
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,  //
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return std::unexpected{"failed to end copy command buffer"};
    }

    TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer, gpuManager.computeQueue().queue, VK_NULL_HANDLE));

    if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
        return std::unexpected{"failed to wait queue"};
    }

    return {};
}

auto Buffer::copyToBuffer(std::span<std::byte const> src,  //
                          size_t dstOffset  //
                          ) noexcept -> std::expected<void, std::string> {
    if (!m_allocation) {
        return std::unexpected{"buffer is not initialized"};
    }

    if (m_size < (dstOffset + src.size_bytes())) {
        return std::unexpected{"buffer size is too small"};
    }

    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    switch (m_type) {
        case Type::Device: {
            TRY_EXPECTED_VOID(copy_memory_to_device_buffer(*this, src, dstOffset));
        } break;
        case Type::Upload: {
            if (vmaCopyMemoryToAllocation(gpuManager.allocator(),  //
                                          src.data(),  //
                                          m_allocation,  //
                                          dstOffset,  //
                                          src.size_bytes()) != VK_SUCCESS) {
                return std::unexpected{"failed to copy data to buffer"};
            }
        } break;
        case Type::Readback: {
            // makes no sense, but still valid
            if (vmaCopyMemoryToAllocation(gpuManager.allocator(),  //
                                          src.data(),  //
                                          m_allocation,  //
                                          dstOffset,  //
                                          src.size_bytes()) != VK_SUCCESS) {
                return std::unexpected{"failed to copy data to buffer"};
            }
        } break;

        default:
            return std::unexpected{"unknown buffer type"};
    }

    return {};
}

auto Buffer::copyFromBuffer(std::span<std::byte> dst,  //
                            size_t srcOffset  //
                            ) noexcept -> std::expected<void, std::string> {
    if (!m_allocation) {
        return std::unexpected{"buffer is not initialized"};
    }

    if (m_size < (srcOffset + dst.size_bytes())) {
        return std::unexpected{"buffer size is too small"};
    }

    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    switch (m_type) {
        case Type::Device: {
            TRY_EXPECTED_VOID(copy_device_buffer_to_memory(dst, *this, srcOffset));
        } break;
        case Type::Upload: {
            // makes no sense, but still valid
            if (vmaCopyAllocationToMemory(gpuManager.allocator(),  //
                                          m_allocation,  //
                                          srcOffset,  //
                                          dst.data(),  //
                                          dst.size_bytes()) != VK_SUCCESS) {
                return std::unexpected{"failed to copy data from host-visible buffer"};
            }
        } break;
        case Type::Readback: {
            if (vmaCopyAllocationToMemory(gpuManager.allocator(),  //
                                          m_allocation,  //
                                          srcOffset,  //
                                          dst.data(),  //
                                          dst.size_bytes()) != VK_SUCCESS) {
                return std::unexpected{"failed to copy data from host-visible buffer"};
            }
        } break;

        default:
            return std::unexpected{"unknown buffer type"};
    }

    return {};
}

auto Buffer::init(size_t size, Type type) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, GpuManager::get());

    m_size = size;
    m_type = type;

    VkBufferUsageFlags2CreateInfo flagsCreateInfo = vku::InitStructHelper{};
    flagsCreateInfo.usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |  //
                            VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |  //
                            VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;

    VkBufferCreateInfo bufferCreateInfo = vku::InitStructHelper{&flagsCreateInfo};
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = 0;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = nullptr;

    VmaAllocationCreateFlags allocationFlags{};

    switch (m_type) {
        case Type::Device:
            allocationFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
            break;
        case Type::Upload:
            allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;
        case Type::Readback:
            allocationFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            break;

        default:
            return std::unexpected{"unknown buffer type"};
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
        return std::unexpected{"failed to create buffer"};
    }

    return {};
}

auto Buffer::destroy() noexcept -> void {
    if (!m_buffer) {
        return;
    }

    auto result{GpuManager::get()};
    // should never happen, since if m_buffer exists means GpuManager is initizlized
    if (!result) {
        return;
    }

    auto& gpuManager{result.value().get()};

    vmaDestroyBuffer(gpuManager.allocator(), m_buffer, m_allocation);

    m_allocation = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_size = 0;
}

auto Buffer::size() const noexcept -> size_t {
    return m_size;
}

auto Buffer::buffer() const noexcept -> VkBuffer {
    return m_buffer;
}

auto Buffer::type() const noexcept -> Type {
    return m_type;
}

BufferSpan::BufferSpan(Buffer const& buffer) noexcept
    : m_buffer{buffer},  //
      m_offset{0},  //
      m_size{buffer.size()}  //
{}

auto BufferSpan::create(Buffer const& buffer,  //
                        size_t offset,  //
                        size_t size  //
                        ) noexcept -> std::expected<BufferSpan, std::string> {
    if (buffer.size() < (offset + size)) {
        return std::unexpected{"buffer size is too small"};
    }

    BufferSpan s{buffer};
    s.m_offset = offset;
    s.m_size = size;

    return s;
}

auto BufferSpan::buffer() const noexcept -> Buffer const& {
    return m_buffer;
}
auto BufferSpan::size() const noexcept -> size_t {
    return m_size;
};
auto BufferSpan::offset() const noexcept -> size_t {
    return m_offset;
};

}  // namespace gpu
