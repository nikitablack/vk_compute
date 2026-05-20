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

template <cuda::utils::ItCount IT_COUNT>
[[nodiscard]] auto run_benchmark(float const* aDevice,  //
                                 float const* bDevice,  //
                                 float* resultDevice,  //
                                 size_t sizeBytes  //
                                 ) noexcept -> std::expected<void, std::string> {
    for (uint32_t blockSizeX{64}; blockSizeX <= 1024; blockSizeX *= 2) {
        fmt::println("IT_COUNT: {}, BLOCK_SIZE_X: {}", static_cast<uint32_t>(IT_COUNT), blockSizeX);

        TRY_EXPECTED(auto const percentiles, utils::benchmark_with_percentiles(
                                                 [&]() -> std::expected<void, std::string> {
                                                     TRY_EXPECTED_VOID(cuda::add(aDevice,  //
                                                                                 bDevice,  //
                                                                                 resultDevice,  //
                                                                                 sizeBytes,  //
                                                                                 blockSizeX,  //
                                                                                 IT_COUNT));
                                                     return {};
                                                 },
                                                 100));

        for (auto const& p : percentiles) {
            fmt::print("p{:5.1f}: {:.2f} us\n", p.first, p.second);
        }
    }

    return {};
}

auto main_impl() noexcept -> std::expected<void, std::string> {
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
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_1>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_2>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_4>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_8>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_16>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_32>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_64>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_128>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_256>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_512>(aDevice, bDevice, resultDevice, S));
    TRY_EXPECTED_VOID(run_benchmark<cuda::utils::ItCount::IT_1024>(aDevice, bDevice, resultDevice, S));

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
