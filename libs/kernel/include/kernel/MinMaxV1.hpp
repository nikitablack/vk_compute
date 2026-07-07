#pragma once

#include <expected>
#include <gpu/HostVisibleBuffer.hpp>
#include <optional>
#include <span>
#include <string>

namespace gpu {

class DeviceBuffer;

};

namespace kernel {

/**
 * MinMaxV1 searches minimum and maximum value in range using a uint representation of a float.
 */
class MinMaxV1 {
public:
    struct Result {
        float min;
        float max;
    };

public:
    MinMaxV1(MinMaxV1&& other) noexcept;

    MinMaxV1(const MinMaxV1&) = delete;
    MinMaxV1& operator=(const MinMaxV1&) = delete;
    MinMaxV1& operator=(MinMaxV1&&) = delete;

private:
    MinMaxV1() = default;

    [[nodiscard]] auto init(uint32_t workgroupSizeX) noexcept -> std::expected<void, std::string>;

public:
    [[nodiscard]] static auto create(uint32_t workgroupSizeX = 128) noexcept -> std::expected<MinMaxV1, std::string>;

    [[nodiscard]] auto operator()(std::span<float const> a  //
    ) const noexcept -> std::expected<Result, std::string>;

    [[nodiscard]] auto operator()(gpu::DeviceBuffer const& a,  //
                                  gpu::DeviceBuffer const& b,  //
                                  size_t sizeBytes  //
    ) const noexcept -> std::expected<void, std::string>;

    [[nodiscard]] auto read(gpu::DeviceBuffer const& bDevice,  //
                            std::optional<gpu::HostVisibleBuffer> stagingBuffer = std::nullopt  //
    ) const noexcept -> std::expected<Result, std::string>;

    auto destroy() noexcept -> void;

private:
    VkPipeline m_minmaxAtomicUintPipeline{VK_NULL_HANDLE};
    VkPipeline m_atomicUintInitPipeline{VK_NULL_HANDLE};
    uint32_t m_workgroupSizeX{0};
};

}  // namespace kernel
