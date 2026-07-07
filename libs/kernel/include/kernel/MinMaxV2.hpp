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
 * MinMaxV2 searches minimum and maximum value in range using a reduction.
 */
class MinMaxV2 {
public:
    struct Result {
        float min;
        float max;
    };

public:
    MinMaxV2(MinMaxV2&& other) noexcept;

    MinMaxV2(const MinMaxV2&) = delete;
    MinMaxV2& operator=(const MinMaxV2&) = delete;
    MinMaxV2& operator=(MinMaxV2&&) = delete;

private:
    MinMaxV2() = default;

    [[nodiscard]] auto init(uint32_t workgroupSizeX) noexcept -> std::expected<void, std::string>;

public:
    [[nodiscard]] static auto create(uint32_t workgroupSizeX = 128) noexcept -> std::expected<MinMaxV2, std::string>;

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
    VkPipeline m_minmaxPipeline{VK_NULL_HANDLE};
    uint32_t m_workgroupSizeX{0};
};

}  // namespace kernel
