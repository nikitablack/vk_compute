#pragma once

#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <optional>
#include <span>
#include <string>

namespace kernel::minmax {

struct Result {
    float min;
    float max;
};

[[nodiscard]] auto run(std::span<float const> in,  //
                       uint32_t workgroupSizeX = 128  //
                       ) noexcept -> std::expected<Result, std::string>;

[[nodiscard]] auto run(gpu::DeviceBuffer const& in,  //
                       gpu::DeviceBuffer const& result,  //
                       uint32_t workgroupSizeX = 128,  //
                       std::optional<uint64_t> sizeBytesIn = std::nullopt  //
                       ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto read(gpu::DeviceBuffer const& resultDevice,  //
                        std::optional<gpu::HostVisibleBuffer> stagingBuffer = std::nullopt  //
                        ) -> std::expected<Result, std::string>;

}  // namespace kernel::minmax
