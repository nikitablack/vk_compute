#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace cuda::impl {

[[nodiscard]] auto add_v4(std::span<float const> aHost,  //
                          std::span<float const> bHost,  //
                          std::span<float> resultHost,  //
                          uint32_t blockSizeX = 128  //
                          ) noexcept -> std::optional<std::string>;

[[nodiscard]] auto add_v4(float const* aDevice,  //
                          float const* bDevice,  //
                          float* resultDevice,  //
                          size_t sizeBytes,  //
                          uint32_t blockSizeX = 128  //
                          ) noexcept -> std::optional<std::string>;

}  // namespace cuda::impl
