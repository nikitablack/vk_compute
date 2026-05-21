#include <cuda/impl/add_chunked.hpp>
#include <cuda/impl/add_data.hpp>
#include <cuda/utils/check_error.hpp>
#include <utility>
#include <utils/try_optional.hpp>

namespace {

template <uint32_t N>
__global__ auto gpu_add_chunked(float const* __restrict__ a,  //
                                float const* __restrict__ b,  //
                                float* __restrict__ result,  //
                                size_t n  //
                                ) -> void {
    uint32_t const globalIdX{blockIdx.x * blockDim.x + threadIdx.x};

#pragma unroll
    for (uint32_t i{0}; i < N; ++i) {
        size_t const idx{globalIdX + i * gridDim.x * blockDim.x};

        if (idx < n) {
            result[idx] = a[idx] + b[idx];
        }
    }
}

template <cuda::utils::ChunkSize CHUNK_SIZE>
auto call_add(float const* aDevice,  //
              float const* bDevice,  //
              float* resultDevice,  //
              uint32_t n,  //
              uint32_t blockSizeX  //
              ) noexcept -> void {
    uint32_t constexpr S{static_cast<uint32_t>(CHUNK_SIZE)};

    // threadCapacity = gridSizeX * blockSizeX * S, where
    // threadCapacity >= n =>
    // => gridSizeX = n / (blockSizeX * S) =>
    // => gridSizeX = (n + blockSizeX * S - 1) / (blockSizeX * N)

    dim3 const blockSize{blockSizeX, 1, 1};
    dim3 const gridSize{(n + blockSize.x * S - 1) / (blockSize.x * S), 1, 1};

    gpu_add_chunked<S><<<gridSize, blockSize>>>(aDevice, bDevice, resultDevice, n);
}

}  // namespace

namespace cuda::impl {

auto add_chunked(std::span<float const> aHost,  //
                 std::span<float const> bHost,  //
                 std::span<float> resultHost,  //
                 uint32_t blockSizeX,  //
                 utils::ChunkSize const chunkSize  //
                 ) noexcept -> std::optional<std::string> {
    // special case
    if (aHost.size() == 0) {
        return {};
    }

    // input validation
    if ((aHost.size() != bHost.size()) || (aHost.size() != resultHost.size())) {
        return std::optional("input data size mismatch");
    }

    uint32_t const dataSizeBytes{static_cast<uint32_t>(aHost.size_bytes())};

    AddData addData{};

    CHECK_ERROR_OPT(cudaMalloc(&addData.a, dataSizeBytes));
    CHECK_ERROR_OPT(cudaMalloc(&addData.b, dataSizeBytes));
    CHECK_ERROR_OPT(cudaMalloc(&addData.result, dataSizeBytes));

    CHECK_ERROR_OPT(cudaMemcpy(addData.a, aHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));
    CHECK_ERROR_OPT(cudaMemcpy(addData.b, bHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));

    TRY_OPTIONAL(add_chunked(addData.a, addData.b, addData.result, dataSizeBytes, blockSizeX, chunkSize));

    CHECK_ERROR_OPT(cudaMemcpy(resultHost.data(), addData.result, dataSizeBytes, cudaMemcpyDeviceToHost));

    return std::nullopt;
}

auto add_chunked(float const* aDevice,  //
                 float const* bDevice,  //
                 float* resultDevice,  //
                 size_t sizeBytes,  //
                 uint32_t blockSizeX,  //
                 utils::ChunkSize const chunkSize  //
                 ) noexcept -> std::optional<std::string> {
    if (sizeBytes == 0) {
        return {};
    }

    {
        uint32_t const n{static_cast<uint32_t>(sizeBytes / sizeof(float))};

        switch (chunkSize) {
            case utils::ChunkSize::C_1: {
                call_add<utils::ChunkSize::C_1>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_2: {
                call_add<utils::ChunkSize::C_2>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_4: {
                call_add<utils::ChunkSize::C_4>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_8: {
                call_add<utils::ChunkSize::C_8>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_16: {
                call_add<utils::ChunkSize::C_16>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_32: {
                call_add<utils::ChunkSize::C_32>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_64: {
                call_add<utils::ChunkSize::C_64>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_128: {
                call_add<utils::ChunkSize::C_128>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_256: {
                call_add<utils::ChunkSize::C_256>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_512: {
                call_add<utils::ChunkSize::C_512>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ChunkSize::C_1024: {
                call_add<utils::ChunkSize::C_1024>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;

            default:
                return std::make_optional<std::string>(
                    "unknown chunk size, use a value from the ChunkSize enumeration");
        }
    }

    CHECK_ERROR_OPT(cudaPeekAtLastError());
    CHECK_ERROR_OPT(cudaDeviceSynchronize());

    return std::nullopt;
}

}  // namespace cuda::impl
