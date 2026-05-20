#pragma once

#include <cuda/utils/constants.hpp>
#include <optional>
#include <span>
#include <string>

namespace cuda::impl {

[[nodiscard]] auto add(std::span<float const> aHost,  //
                       std::span<float const> bHost,  //
                       std::span<float> resultHost,  //
                       uint32_t blockSizeX = 128,  //
                       utils::ItCount const itCount = utils::ItCount::IT_64  //
                       ) noexcept -> std::optional<std::string>;

[[nodiscard]] auto add(float const* aDevice,  //
                       float const* bDevice,  //
                       float* resultDevice,  //
                       size_t sizeBytes,  //
                       uint32_t blockSizeX = 128,  //
                       utils::ItCount const itCount = utils::ItCount::IT_64  //
                       ) noexcept -> std::optional<std::string>;

}  // namespace cuda::impl
