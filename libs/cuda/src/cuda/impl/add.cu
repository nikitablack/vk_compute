#include <cuda/impl/add.hpp>
#include <cuda/impl/add_data.hpp>
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

namespace cuda::impl {

auto add(std::span<float const> aHost,  //
         std::span<float const> bHost,  //
         std::span<float> resultHost,  //
         uint32_t blockSizeX) noexcept -> std::optional<std::string> {
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

    TRY_OPTIONAL(add(addData.a, addData.b, addData.result, dataSizeBytes, blockSizeX));

    CHECK_ERROR_OPT(cudaMemcpy(resultHost.data(), addData.result, dataSizeBytes, cudaMemcpyDeviceToHost));

    return std::nullopt;
}

auto add(float const* aDevice,  //
         float const* bDevice,  //
         float* resultDevice,  //
         size_t sizeBytes,  //
         uint32_t blockSizeX  //
         ) noexcept -> std::optional<std::string> {
    if (sizeBytes == 0) {
        return {};
    }

    uint32_t const n{static_cast<uint32_t>(sizeBytes / sizeof(float))};

    dim3 const blockSize{blockSizeX, 1, 1};
    dim3 const gridSize{(n + blockSize.x - 1) / blockSize.x, 1, 1};

    gpu_add<<<gridSize, blockSize>>>(aDevice, bDevice, resultDevice, n);

    CHECK_ERROR_OPT(cudaPeekAtLastError());
    CHECK_ERROR_OPT(cudaDeviceSynchronize());

    return std::nullopt;
}

}  // namespace cuda::impl
