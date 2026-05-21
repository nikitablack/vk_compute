#include <cuda/impl/add_data.hpp>
#include <cuda/impl/add_v4.hpp>
#include <cuda/utils/check_error.hpp>
#include <utils/try_optional.hpp>

namespace {

__global__ auto gpu_add_v4(float4 const* __restrict__ a,  //
                           float4 const* __restrict__ b,  //
                           float4* __restrict__ result,  //
                           size_t n  //
                           ) -> void {
    uint32_t const globalIdX{blockIdx.x * blockDim.x + threadIdx.x};

    if (globalIdX >= n) {
        return;
    }

    float4 const va{a[globalIdX]};
    float4 const vb{b[globalIdX]};
    float4 const vc{va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};

    result[globalIdX] = vc;
}

}  // namespace

namespace cuda::impl {

auto add_v4(std::span<float const> aHost,  //
            std::span<float const> bHost,  //
            std::span<float> resultHost,  //
            uint32_t blockSizeX) noexcept -> std::optional<std::string> {
    // special case
    if (aHost.size() == 0) {
        return {};
    }

    // multiple of 4
    if ((aHost.size() % 4) != 0) {
        return std::optional("input size should be a multiple of 4");
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

    TRY_OPTIONAL(add_v4(addData.a, addData.b, addData.result, dataSizeBytes, blockSizeX));

    CHECK_ERROR_OPT(cudaMemcpy(resultHost.data(), addData.result, dataSizeBytes, cudaMemcpyDeviceToHost));

    return std::nullopt;
}

auto add_v4(float const* aDevice,  //
            float const* bDevice,  //
            float* resultDevice,  //
            size_t sizeBytes,  //
            uint32_t blockSizeX  //
            ) noexcept -> std::optional<std::string> {
    if (sizeBytes == 0) {
        return {};
    }

    if ((sizeBytes % 16) != 0) {
        return std::optional("input size should be a multiple of 16");
    }

    uint32_t const n{static_cast<uint32_t>(sizeBytes / sizeof(float)) / 4};

    dim3 const blockSize{blockSizeX, 1, 1};
    dim3 const gridSize{(n + blockSize.x - 1) / blockSize.x, 1, 1};

    gpu_add_v4<<<gridSize, blockSize>>>(reinterpret_cast<float4 const*>(aDevice),  //
                                        reinterpret_cast<float4 const*>(bDevice),  //
                                        reinterpret_cast<float4*>(resultDevice),  //
                                        n);

    CHECK_ERROR_OPT(cudaPeekAtLastError());
    CHECK_ERROR_OPT(cudaDeviceSynchronize());

    return std::nullopt;
}

}  // namespace cuda::impl
