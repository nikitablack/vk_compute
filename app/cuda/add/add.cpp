#include <cuda_runtime.h>
#include <fmt/core.h>

#include <algorithm>
#include <cstdlib>
#include <cuda/add.hpp>
#include <cuda/utils/check_error.hpp>
#include <expected>
#include <ranges>
#include <string>
#include <utils/benchmark_with_percentiles.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace {

auto main_impl() -> std::expected<void, std::string> {
    uint32_t constexpr N{1024 * 1024 * 100};
    uint32_t constexpr S{N * sizeof(float)};

    // initialize host memory
    std::vector<float> aHost(N);
    std::vector<float> bHost(N);

    // initizlize host data
    {
        std::ranges::copy(
            std::views::iota(uint32_t{0}, N) | std::views::transform([](auto i) { return static_cast<float>(i); }),
            aHost.begin());

        std::ranges::copy(
            std::views::iota(uint32_t{0}, N) | std::views::transform([](auto i) { return static_cast<float>(i); }),
            bHost.begin());
    }

    // initialize device memory
    float* aDevice{nullptr};
    float* bDevice{nullptr};
    float* resultDevice{nullptr};

    CHECK_ERROR_EXP(cudaMalloc(&aDevice, S));
    CHECK_ERROR_EXP(cudaMalloc(&bDevice, S));
    CHECK_ERROR_EXP(cudaMalloc(&resultDevice, S));

    // copy host data to device
    {
        CHECK_ERROR_EXP(cudaMemcpy(aDevice, aHost.data(), S, cudaMemcpyHostToDevice));
        CHECK_ERROR_EXP(cudaMemcpy(bDevice, bHost.data(), S, cudaMemcpyHostToDevice));
    }

    // compute
    TRY_EXPECTED_VOID(utils::benchmark_with_percentiles(
        [&]() -> std::expected<void, std::string> {
            if (auto res{cuda::add(aDevice, bDevice, resultDevice, S)}) {
                return std::unexpected(res.value());
            }
            return {};
        },
        10));

    // read result
    std::vector<float> resultHost(N);
    CHECK_ERROR_EXP(cudaMemcpy(resultHost.data(), resultDevice, S, cudaMemcpyDeviceToHost));

    // clear
    CHECK_ERROR_EXP(cudaFree(aDevice));
    CHECK_ERROR_EXP(cudaFree(bDevice));
    CHECK_ERROR_EXP(cudaFree(resultDevice));

    return {};
}

}  // namespace

auto main(int /* argc */, char* /* argv */[]) -> int {
    if (auto result{main_impl()}; !result) {
        fmt::println("fatal error: {}", result.error());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
