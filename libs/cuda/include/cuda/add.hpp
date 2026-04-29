#pragma once

#include <optional>
#include <span>
#include <string>

namespace cuda {

[[nodiscard]] auto add(std::span<float const> aHost,  //
                       std::span<float const> bHost,  //
                       std::span<float> resultHost  //
                       ) noexcept -> std::optional<std::string>;

[[nodiscard]] auto add(float const* aDevice,  //
                       float const* bDevice,  //
                       float* resultDevice,  //
                       size_t sizeBytes  //
                       ) noexcept -> std::optional<std::string>;

}  // namespace cuda
