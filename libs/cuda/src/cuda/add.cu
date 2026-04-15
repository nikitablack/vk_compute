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

}  // namespace

namespace cuda {

auto add(std::span<float const> aHost,  //
         std::span<float const> bHost,  //
         std::span<float> resultHost  //
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

    float* aDevice{nullptr};
    float* bDevice{nullptr};
    float* resultDevice{nullptr};

    CHECK_ERROR_OPT(cudaMalloc(&aDevice, dataSizeBytes));
    CHECK_ERROR_OPT(cudaMalloc(&bDevice, dataSizeBytes));
    CHECK_ERROR_OPT(cudaMalloc(&resultDevice, dataSizeBytes));

    CHECK_ERROR_OPT(cudaMemcpy(aDevice, aHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));
    CHECK_ERROR_OPT(cudaMemcpy(bDevice, bHost.data(), dataSizeBytes, cudaMemcpyHostToDevice));

    TRY_OPTIONAL(add(aDevice, bDevice, resultDevice, dataSizeBytes));

    CHECK_ERROR_OPT(cudaMemcpy(resultHost.data(), resultDevice, dataSizeBytes, cudaMemcpyDeviceToHost));

    CHECK_ERROR_OPT(cudaFree(aDevice));
    CHECK_ERROR_OPT(cudaFree(bDevice));
    CHECK_ERROR_OPT(cudaFree(resultDevice));

    return std::nullopt;
}

auto add(float const* aDevice,  //
         float const* bDevice,  //
         float* resultDevice,  //
         size_t sizeBytes  //
         ) noexcept -> std::optional<std::string> {
    using T = float;

    if (sizeBytes == 0) {
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
