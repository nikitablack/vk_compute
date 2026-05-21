#pragma once

#include <cuda/utils/constants.hpp>
#include <expected>
#include <span>
#include <string>

namespace cuda {

[[nodiscard]] auto add(std::span<float const> aHost,  //
                       std::span<float const> bHost,  //
                       std::span<float> resultHost,  //
                       uint32_t blockSizeX = 128  //
                       ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto add(float const* aDevice,  //
                       float const* bDevice,  //
                       float* resultDevice,  //
                       size_t sizeBytes,  //
                       uint32_t blockSizeX = 128  //
                       ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto add_v4(std::span<float const> aHost,  //
                          std::span<float const> bHost,  //
                          std::span<float> resultHost,  //
                          uint32_t blockSizeX = 128  //
                          ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto add_v4(float const* aDevice,  //
                          float const* bDevice,  //
                          float* resultDevice,  //
                          size_t sizeBytes,  //
                          uint32_t blockSizeX = 128  //
                          ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto add_chunked(std::span<float const> aHost,  //
                               std::span<float const> bHost,  //
                               std::span<float> resultHost,  //
                               uint32_t blockSizeX = 128,  //
                               utils::ChunkSize const chunkSize = utils::ChunkSize::C_64  //
                               ) noexcept -> std::expected<void, std::string>;

[[nodiscard]] auto add_chunked(float const* aDevice,  //
                               float const* bDevice,  //
                               float* resultDevice,  //
                               size_t sizeBytes,  //
                               uint32_t blockSizeX = 128,  //
                               utils::ChunkSize const chunkSize = utils::ChunkSize::C_64  //
                               ) noexcept -> std::expected<void, std::string>;

}  // namespace cuda
