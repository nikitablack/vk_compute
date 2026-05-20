#include <cuda/add.hpp>
#include <cuda/impl/add.hpp>

namespace cuda {

auto add(std::span<float const> aHost,  //
         std::span<float const> bHost,  //
         std::span<float> resultHost,  //
         uint32_t blockSizeX,  //
         utils::ItCount const itCount  //
         ) noexcept -> std::expected<void, std::string> {
    auto const result{impl::add(aHost, bHost, resultHost, blockSizeX, itCount)};

    if (result) {
        return std::unexpected{result.value()};
    }

    return {};
}

auto add(float const* aDevice,  //
         float const* bDevice,  //
         float* resultDevice,  //
         size_t sizeBytes,  //
         uint32_t blockSizeX,  //
         utils::ItCount const itCount  //
         ) noexcept -> std::expected<void, std::string> {
    auto const result{impl::add(aDevice, bDevice, resultDevice, sizeBytes, blockSizeX, itCount)};

    if (result) {
        return std::unexpected{result.value()};
    }

    return {};
}

}  // namespace cuda
