#pragma once

#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <span>
#include <string>

namespace kernel {

[[nodiscard]] auto add(std::span<float const> a,  //
                       std::span<float const> b,  //
                       std::span<float> result  //
                       ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto add(gpu::DeviceBuffer const& a,  //
                       gpu::DeviceBuffer const& b,  //
                       gpu::DeviceBuffer const& result,  //
                       std::optional<uint64_t> sizeBytes = std::nullopt  //
                       ) noexcept -> std::expected<void, std::string>;

}  // namespace kernel
