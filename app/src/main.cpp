#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <gpu/GpuManager.hpp>
#include <kernel/Add.hpp>
#include <ranges>
#include <string>
#include <utils/try_expected.hpp>

namespace {

auto main_impl() -> std::expected<void, std::string> {
    gpu::GpuManager gpuManager{};
    TRY_EXPECTED_VOID(gpuManager.initialize());

    uint32_t constexpr N{1024 * 1024 * 100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    std::ranges::copy(
        std::views::iota(uint32_t{0}, N) | std::views::transform([](auto i) { return static_cast<float>(i); }),
        a.begin());

    std::ranges::copy(
        std::views::iota(uint32_t{0}, N) | std::views::transform([](auto i) { return static_cast<float>(i); }),
        b.begin());

    kernel::Add add{};

    TRY_EXPECTED_VOID(add.run(gpuManager, a, b, result));

    // fmt::print("[{}]\n", fmt::join(result | std::views::take(10), ", "));

    gpuManager.flush();
    add.destroy();
    gpuManager.destroy();

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