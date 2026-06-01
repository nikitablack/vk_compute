#pragma once

#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <optional>
#include <span>
#include <string>

namespace kernel {

[[nodiscard]] auto add(std::span<float const> a,  //
                       std::span<float const> b,  //
                       std::span<float> result,  //
                       uint32_t workgroupSizeX = 128  //
                       ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto add(gpu::DeviceBuffer const& a,  //
                       gpu::DeviceBuffer const& b,  //
                       gpu::DeviceBuffer const& result,  //
                       uint32_t workgroupSizeX = 128,  //
                       std::optional<uint64_t> sizeBytes = std::nullopt  //
                       ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto read_add_result(std::span<float> resultHost,  //
                                   gpu::DeviceBuffer const& resultDevice,  //
                                   std::optional<size_t> bytesToRead = std::nullopt,  //
                                   std::optional<gpu::HostVisibleBuffer> stagingBuffer = std::nullopt  //
                                   ) -> std::expected<void, std::string>;

}  // namespace kernel
