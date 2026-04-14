#include <fmt/core.h>

#include <cstring>
#include <glm/gtc/type_ptr.hpp>
#include <gpu/ImmediateDataBufferManager.hpp>
#include <utils/try_expected.hpp>

namespace {

constexpr VkDeviceSize DEFAULT_BUFFER_SIZE{1024 * 1024};  // 1MB

auto padding(uint64_t offset, uint64_t alignment) -> uint64_t {
    auto misalignment = offset % alignment;
    if (misalignment > 0) {
        return alignment - misalignment;
    }

    return 0;
}

}  // namespace

namespace gpu {

[[nodiscard]] auto ImmediateDataBufferManager::init(VmaAllocator allocator,  //
                                                    VkPhysicalDeviceProperties2 const& deviceProperties  //
                                                    ) noexcept -> std::expected<void, std::string> {
    m_allocator = allocator;

    HostVisibleBuffer vulkanBuffer{};
    TRY_EXPECTED_VOID(vulkanBuffer.init(m_allocator,  //
                                        DEFAULT_BUFFER_SIZE,  //
                                        false,  //
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));

    m_occupancyInfos.push_back(OccupancyInfo{vulkanBuffer, 0});

    m_minStorageBufferOffsetAlignment = deviceProperties.properties.limits.minStorageBufferOffsetAlignment;

    return {};
}

auto ImmediateDataBufferManager::destroy() noexcept -> void {
    for (auto& info : m_occupancyInfos) {
        info.vulkanBuffer.destroy();
    }
}

auto ImmediateDataBufferManager::reset() noexcept -> void {
    for (auto& info : m_occupancyInfos) {
        info.occupied = 0;
    }
}

auto ImmediateDataBufferManager::pushData(VkCommandBuffer const commandBuffer,  //
                                          std::span<std::byte const> data,  //
                                          StorageDescriptorSetManager& storageDescriptorSetManager  //
                                          ) noexcept -> std::expected<uint32_t, std::string> {
    TRY_EXPECTED(PushDataResult const pushDataResult, pushDataImpl(data));

    // update descriptor
    VkDescriptorBufferInfo descriptorBufferInfo{};
    descriptorBufferInfo.buffer = pushDataResult.buffer;
    descriptorBufferInfo.offset = pushDataResult.startOffset;
    descriptorBufferInfo.range = pushDataResult.size;

    return storageDescriptorSetManager.push(commandBuffer, descriptorBufferInfo);
}

auto ImmediateDataBufferManager::pushDataImpl(std::span<std::byte const> data  //
                                              ) noexcept -> std::expected<PushDataResult, std::string> {
    OccupancyInfo* occupancyInfo{nullptr};

    // find the first buffer with enough space for the new data
    for (auto& info : m_occupancyInfos) {
        if ((info.occupied + data.size()) > info.vulkanBuffer.size()) {
            continue;
        }

        occupancyInfo = &info;
    }

    // if no buffer was found with enough space, create a new one
    if (occupancyInfo == nullptr) {
        VkDeviceSize newSize{std::max(data.size(), DEFAULT_BUFFER_SIZE)};

        fmt::println("allocating new immediate buffer with the size {}", newSize);

        HostVisibleBuffer buffer{};
        TRY_EXPECTED_VOID(buffer.init(m_allocator,  //
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  //
                                      newSize));

        m_occupancyInfos.push_back(OccupancyInfo{buffer, 0});

        occupancyInfo = &m_occupancyInfos.back();
    }

    // copy data to a buffer at offset
    size_t startOffset{occupancyInfo->occupied};
    TRY_EXPECTED_VOID(occupancyInfo->vulkanBuffer.copyTo(data, startOffset));

    // update offset taking alingment into account
    occupancyInfo->occupied += data.size();
    occupancyInfo->occupied += padding(occupancyInfo->occupied, m_minStorageBufferOffsetAlignment);

    PushDataResult pushDataResult{};
    pushDataResult.startOffset = startOffset;
    pushDataResult.size = data.size();
    pushDataResult.buffer = occupancyInfo->vulkanBuffer.buffer();

    return pushDataResult;
}

}  // namespace gpu
