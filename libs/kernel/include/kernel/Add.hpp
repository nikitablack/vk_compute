#pragma once

#include <expected>
#include <gpu/HostVisibleBuffer.hpp>
#include <optional>
#include <span>
#include <string>

namespace gpu {

class DeviceBuffer;

}

namespace kernel {

class Add {
public:
    Add(const Add&) = delete;
    Add(Add&& other) noexcept;
    Add& operator=(const Add&) = delete;
    Add& operator=(Add&&) = delete;

private:
    Add() = default;

    [[nodiscard]] auto init(uint32_t workgroupSizeX) noexcept -> std::expected<void, std::string>;

public:
    [[nodiscard]] static auto create(uint32_t workgroupSizeX = 128) noexcept -> std::expected<Add, std::string>;

    [[nodiscard]] auto operator()(std::span<float const> a,  //
                                  std::span<float const> b,  //
                                  std::span<float> c  //
    ) const noexcept -> std::expected<void, std::string>;

    [[nodiscard]] auto operator()(gpu::DeviceBuffer const& a,  //
                                  gpu::DeviceBuffer const& b,  //
                                  gpu::DeviceBuffer const& c,  //
                                  size_t sizeBytes  //
    ) const noexcept -> std::expected<void, std::string>;

    [[nodiscard]] auto read(std::span<float> cHost,  //
                            gpu::DeviceBuffer const& cDevice,  //
                            size_t sizeBytes,  //
                            std::optional<gpu::HostVisibleBuffer> stagingBuffer = std::nullopt  //
    ) const noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;

private:
    VkPipeline m_pipeline{VK_NULL_HANDLE};
    uint32_t m_workgroupSizeX{0};
};

}  // namespace kernel
