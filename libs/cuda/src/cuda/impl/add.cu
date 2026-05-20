#include <spdlog/spdlog.h>

#include <cuda/impl/add.hpp>
#include <cuda/utils/check_error.hpp>
#include <utility>
#include <utils/try_optional.hpp>

namespace {

// __global__ auto gpu_add(float const* __restrict__ a,  //
//                         float const* __restrict__ b,  //
//                         float* __restrict__ result,  //
//                         size_t n  //
//                         ) -> void {
//     uint32_t const globalIdX{blockIdx.x * blockDim.x + threadIdx.x};

//     if (globalIdX >= n) {
//         return;
//     }

//     float const va{a[globalIdX]};
//     float const vb{b[globalIdX]};
//     float const vc{va + vb};

//     result[globalIdX] = vc;
// }

// __global__ auto gpu_add2(float4 const* __restrict__ a,  //
//                          float4 const* __restrict__ b,  //
//                          float4* __restrict__ result,  //
//                          size_t n  //
//                          ) -> void {
//     uint32_t const globalIdX{blockIdx.x * blockDim.x + threadIdx.x};

//     if (globalIdX >= n) {
//         return;
//     }

//     float4 const va{a[globalIdX]};
//     float4 const vb{b[globalIdX]};
//     float4 const vc{va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};

//     result[globalIdX] = vc;
// }

// template <uint32_t N>
// __global__ auto gpu_add3(float const* __restrict__ a,  //
//                          float const* __restrict__ b,  //
//                          float* __restrict__ result,  //
//                          size_t n  //
//                          ) -> void {
//     uint32_t const globalIdX{blockIdx.x * blockDim.x + threadIdx.x};

// #pragma unroll
//     for (uint32_t i{0}; i < N; ++i) {
//         size_t const idx{globalIdX + i * gridDim.x * blockDim.x};

//         if (idx < n) {
//             result[idx] = a[idx] + b[idx];
//         }
//     }
// }

template <uint32_t N>
__global__ auto gpu_add4(float4 const* __restrict__ a,  //
                         float4 const* __restrict__ b,  //
                         float4* __restrict__ result,  //
                         size_t n  //
                         ) -> void {
    uint32_t const globalIdX{blockIdx.x * blockDim.x + threadIdx.x};
    uint32_t const totalThreads{gridDim.x * blockDim.x};

#pragma unroll
    for (uint32_t i{0}; i < N; ++i) {
        size_t const idx{globalIdX + i * totalThreads};
        // printf("%u %llu %llu\n", globalIdX, idx, n);

        if (idx < n) {
            float4 const va{a[idx]};
            float4 const vb{b[idx]};
            float4 const vc{va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};

            result[idx] = vc;
        }
    }
}

struct DeviceData {
    float* a{nullptr};
    float* b{nullptr};
    float* result{nullptr};

    ~DeviceData() {
        if (a) {
            if (cudaFree(a) != cudaSuccess) {
                spdlog::warn("failed to free cuda memory");
            }
        }

        if (b) {
            if (cudaFree(b) != cudaSuccess) {
                spdlog::warn("failed to free cuda memory");
            }
        }

        if (result) {
            if (cudaFree(result) != cudaSuccess) {
                spdlog::warn("failed to free cuda memory");
            }
        }
    }
};

// template <cuda::ItCount IT_COUNT>
// auto call_add(float const* aDevice,  //
//               float const* bDevice,  //
//               float* resultDevice,  //
//               uint32_t n,  //
//               uint32_t blockSizeX  //
//               ) noexcept -> void {
//     uint32_t constexpr N{static_cast<uint32_t>(IT_COUNT)};

//     dim3 const BLOCK_SIZE{blockSizeX, 1, 1};
//     dim3 const GRID_SIZE{(n + BLOCK_SIZE.x * N - 1) / (BLOCK_SIZE.x * N), 1, 1};

//     gpu_add3<N><<<GRID_SIZE, BLOCK_SIZE>>>(aDevice, bDevice, resultDevice, n);
// }

template <cuda::utils::ItCount IT_COUNT>
auto call_add4(float const* aDevice,  //
               float const* bDevice,  //
               float* resultDevice,  //
               uint32_t n,  //
               uint32_t blockSizeX  //
               ) noexcept -> void {
    n /= 4;
    uint32_t constexpr N{static_cast<uint32_t>(IT_COUNT)};

    // threadCapacity = gridSizeX * blockSizeX * N, where
    // threadCapacity >= n =>
    // => gridSizeX = n / (blockSizeX * N) =>
    // => gridSizeX = (n + blockSizeX * N - 1) / (blockSizeX * N)

    dim3 const BLOCK_SIZE{blockSizeX, 1, 1};
    dim3 const GRID_SIZE{(n + BLOCK_SIZE.x * N - 1) / (BLOCK_SIZE.x * N), 1, 1};

    gpu_add4<N><<<GRID_SIZE, BLOCK_SIZE>>>(reinterpret_cast<float4 const*>(aDevice),  //
                                           reinterpret_cast<float4 const*>(bDevice),  //
                                           reinterpret_cast<float4*>(resultDevice),  //
                                           n);
}

}  // namespace

namespace cuda::impl {

auto add(std::span<float const> aHost,  //
         std::span<float const> bHost,  //
         std::span<float> resultHost,  //
         uint32_t blockSizeX,  //
         utils::ItCount const itCount  //
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

    DeviceData deviceData{};

    CHECK_ERROR_OPT(cudaMalloc(&deviceData.a, dataSizeBytes));
    CHECK_ERROR_OPT(cudaMalloc(&deviceData.b, dataSizeBytes));
    CHECK_ERROR_OPT(cudaMalloc(&deviceData.result, dataSizeBytes));

    CHECK_ERROR_OPT(cudaMemcpy(deviceData.a, aHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));
    CHECK_ERROR_OPT(cudaMemcpy(deviceData.b, bHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));

    TRY_OPTIONAL(add(deviceData.a, deviceData.b, deviceData.result, dataSizeBytes, blockSizeX, itCount));

    CHECK_ERROR_OPT(cudaMemcpy(resultHost.data(), deviceData.result, dataSizeBytes, cudaMemcpyDeviceToHost));

    return std::nullopt;
}

auto add(float const* aDevice,  //
         float const* bDevice,  //
         float* resultDevice,  //
         size_t sizeBytes,  //
         uint32_t blockSizeX,  //
         utils::ItCount const itCount  //
         ) noexcept -> std::optional<std::string> {
    using T = float;

    if (sizeBytes == 0) {
        return {};
    }

    // {
    //     uint32_t const dataCount{static_cast<uint32_t>(sizeBytes / sizeof(T))};

    //     dim3 constexpr BLOCK_SIZE{1024, 1, 1};
    //     dim3 const GRID_SIZE{(dataCount + (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z) - 1) /
    //                              (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z),
    //                          1, 1};

    //     gpu_add<<<GRID_SIZE, BLOCK_SIZE>>>(aDevice, bDevice, resultDevice, dataCount);
    // }

    // {
    //     uint32_t const dataCount{static_cast<uint32_t>(sizeBytes / sizeof(T)) / 4};

    //     dim3 constexpr BLOCK_SIZE{1024, 1, 1};
    //     dim3 const GRID_SIZE{(dataCount + (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z) - 1) /
    //                              (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z),
    //                          1, 1};
    //     gpu_add2<<<GRID_SIZE, BLOCK_SIZE>>>(reinterpret_cast<float4 const*>(aDevice),
    //                                         reinterpret_cast<float4 const*>(bDevice),
    //                                         reinterpret_cast<float4*>(resultDevice), dataCount);
    // }

    // {
    //     // uint32_t constexpr N{16};

    //     uint32_t const dataCount{static_cast<uint32_t>(sizeBytes / sizeof(T)) / N};

    //     dim3 BLOCK_SIZE{blockSizeX, 1, 1};
    //     dim3 const GRID_SIZE{(dataCount + (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z) - 1) /
    //                              (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z),
    //                          1, 1};

    //     gpu_add3<N><<<GRID_SIZE, BLOCK_SIZE>>>(aDevice, bDevice, resultDevice, dataCount);
    //     CHECK_ERROR_OPT(cudaPeekAtLastError());
    //     CHECK_ERROR_OPT(cudaDeviceSynchronize());
    // }

    // {
    //     uint32_t const n{static_cast<uint32_t>(sizeBytes / sizeof(T))};

    //     TRY_OPTIONAL(add_impl_all<1>(aDevice, bDevice, resultDevice, n));
    // }

    // {
    //     uint32_t const n{static_cast<uint32_t>(sizeBytes / sizeof(T))};

    //     switch (itCount) {
    //         case ItCount::IT_1: {
    //             call_add<ItCount::IT_1>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_2: {
    //             call_add<ItCount::IT_2>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_4: {
    //             call_add<ItCount::IT_4>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_8: {
    //             call_add<ItCount::IT_8>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_16: {
    //             call_add<ItCount::IT_16>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_32: {
    //             call_add<ItCount::IT_32>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_64: {
    //             call_add<ItCount::IT_64>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_128: {
    //             call_add<ItCount::IT_128>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_256: {
    //             call_add<ItCount::IT_256>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_512: {
    //             call_add<ItCount::IT_512>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;
    //         case ItCount::IT_1024: {
    //             call_add<ItCount::IT_1024>(aDevice, bDevice, resultDevice, n, blockSizeX);
    //         } break;

    //         default:
    //             return std::make_optional<std::string>(
    //                 "unknown number of iterations, use a value from ItCount enumeration");
    //     }
    // }

    {
        uint32_t const n{static_cast<uint32_t>(sizeBytes / sizeof(T))};

        switch (itCount) {
            case utils::ItCount::IT_1: {
                call_add4<utils::ItCount::IT_1>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_2: {
                call_add4<utils::ItCount::IT_2>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_4: {
                call_add4<utils::ItCount::IT_4>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_8: {
                call_add4<utils::ItCount::IT_8>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_16: {
                call_add4<utils::ItCount::IT_16>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_32: {
                call_add4<utils::ItCount::IT_32>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_64: {
                call_add4<utils::ItCount::IT_64>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_128: {
                call_add4<utils::ItCount::IT_128>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_256: {
                call_add4<utils::ItCount::IT_256>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_512: {
                call_add4<utils::ItCount::IT_512>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;
            case utils::ItCount::IT_1024: {
                call_add4<utils::ItCount::IT_1024>(aDevice, bDevice, resultDevice, n, blockSizeX);
            } break;

            default:
                return std::make_optional<std::string>(
                    "unknown number of iterations, use a value from ItCount enumeration");
        }
    }

    CHECK_ERROR_OPT(cudaPeekAtLastError());
    CHECK_ERROR_OPT(cudaDeviceSynchronize());

    return std::nullopt;
}

}  // namespace cuda::impl
