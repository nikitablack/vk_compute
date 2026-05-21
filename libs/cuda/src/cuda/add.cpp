#include <cuda/add.hpp>
#include <cuda/impl/add.hpp>
#include <cuda/impl/add_chunked.hpp>
#include <cuda/impl/add_v4.hpp>

namespace cuda {

auto add(std::span<float const> aHost,  //
         std::span<float const> bHost,  //
         std::span<float> resultHost,  //
         uint32_t blockSizeX  //
         ) noexcept -> std::expected<void, std::string> {
    if (auto const result{impl::add(aHost, bHost, resultHost, blockSizeX)}) {
        return std::unexpected{result.value()};
    }

    return {};
}

auto add(float const* aDevice,  //
         float const* bDevice,  //
         float* resultDevice,  //
         size_t sizeBytes,  //
         uint32_t blockSizeX  //
         ) noexcept -> std::expected<void, std::string> {
    if (auto const result{impl::add(aDevice, bDevice, resultDevice, sizeBytes, blockSizeX)}) {
        return std::unexpected{result.value()};
    }

    return {};
}

auto add_v4(std::span<float const> aHost,  //
            std::span<float const> bHost,  //
            std::span<float> resultHost,  //
            uint32_t blockSizeX  //
            ) noexcept -> std::expected<void, std::string> {
    if (auto const result{impl::add_v4(aHost, bHost, resultHost, blockSizeX)}) {
        return std::unexpected{result.value()};
    }

    return {};
}

auto add_v4(float const* aDevice,  //
            float const* bDevice,  //
            float* resultDevice,  //
            size_t sizeBytes,  //
            uint32_t blockSizeX  //
            ) noexcept -> std::expected<void, std::string> {
    if (auto const result{impl::add_v4(aDevice, bDevice, resultDevice, sizeBytes, blockSizeX)}) {
        return std::unexpected{result.value()};
    }

    return {};
}

auto add_chunked(std::span<float const> aHost,  //
                 std::span<float const> bHost,  //
                 std::span<float> resultHost,  //
                 uint32_t blockSizeX,  //
                 utils::ChunkSize const chunkSize  //
                 ) noexcept -> std::expected<void, std::string> {
    if (auto const result{impl::add_chunked(aHost, bHost, resultHost, blockSizeX, chunkSize)}) {
        return std::unexpected{result.value()};
    }

    return {};
}

auto add_chunked(float const* aDevice,  //
                 float const* bDevice,  //
                 float* resultDevice,  //
                 size_t sizeBytes,  //
                 uint32_t blockSizeX,  //
                 utils::ChunkSize const chunkSize  //
                 ) noexcept -> std::expected<void, std::string> {
    if (auto const result{impl::add_chunked(aDevice, bDevice, resultDevice, sizeBytes, blockSizeX, chunkSize)}) {
        return std::unexpected{result.value()};
    }

    return {};
}

}  // namespace cuda
