#include <spdlog/spdlog.h>

#include <cuda/add.hpp>
#include <cuda/utils/check_error.hpp>
#include <utils/try_optional.hpp>

namespace {

__global__ auto gpu_add(float const* __restrict__ a,  //
                        float const* __restrict__ b,  //
                        float* __restrict__ result,  //
                        size_t n  //
                        ) -> void {
    uint32_t const globalIdX{blockIdx.x * blockDim.x + threadIdx.x};

    if (globalIdX >= n) {
        return;
    }

    float const va{a[globalIdX]};
    float const vb{b[globalIdX]};
    float const vc{va + vb};

    result[globalIdX] = vc;
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

}  // namespace

namespace cuda {

// auto add(std::span<float const> aHost,  //
//          std::span<float const> bHost,  //
//          std::span<float> resultHost  //
//          ) noexcept -> std::optional<std::string> {
//     // special case
//     if (aHost.size() == 0) {
//         return {};
//     }

//     // input validation
//     if ((aHost.size() != bHost.size()) || (aHost.size() != resultHost.size())) {
//         return std::optional("input data size mismatch");
//     }

//     uint32_t const dataSizeBytes{static_cast<uint32_t>(aHost.size_bytes())};

//     DeviceData deviceData{};

//     CHECK_ERROR_OPT(cudaMalloc(&deviceData.a, dataSizeBytes));
//     CHECK_ERROR_OPT(cudaMalloc(&deviceData.b, dataSizeBytes));
//     CHECK_ERROR_OPT(cudaMalloc(&deviceData.result, dataSizeBytes));

//     CHECK_ERROR_OPT(cudaMemcpy(deviceData.a, aHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));
//     CHECK_ERROR_OPT(cudaMemcpy(deviceData.b, bHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));

//     TRY_OPTIONAL(add(deviceData.a, deviceData.b, deviceData.result, dataSizeBytes));

//     CHECK_ERROR_OPT(cudaMemcpy(resultHost.data(), deviceData.result, dataSizeBytes, cudaMemcpyDeviceToHost));

//     return std::nullopt;
// }

auto matmul(float const* aDevice,  //
            float const* bDevice,  //
            float* resultDevice,  //
            uint32_t M,  //
            uint32_t N,  //
            uint32_t K  //
            ) noexcept -> std::optional<std::string> {
    using T = float;

    if ((M == 0) || (N == 0) || (K == 0)) {
        return {};
    }

    uint32_t const dataCount{static_cast<uint32_t>(sizeBytes / sizeof(T))};

    dim3 constexpr BLOCK_SIZE{1024, 1, 1};
    dim3 const GRID_SIZE{
        (dataCount + (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z) - 1) / (BLOCK_SIZE.x * BLOCK_SIZE.y * BLOCK_SIZE.z),
        1, 1};

    gpu_add<<<GRID_SIZE, BLOCK_SIZE>>>(aDevice, bDevice, resultDevice, dataCount);
    CHECK_ERROR_OPT(cudaPeekAtLastError());
    CHECK_ERROR_OPT(cudaDeviceSynchronize());

    return std::nullopt;
}

}  // namespace cuda
