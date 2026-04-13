#pragma once

#include <fmt/core.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <utils/try_expected.hpp>

namespace kernel::utils {

template <uint32_t RUNS = 1000, uint32_t PERCENTILE_COUNT = 10>
[[nodiscard]] auto benchmark_with_percentiles(std::function<std::expected<void, std::string>()> const& fn)
    -> std::expected<void, std::string> {
    static_assert(RUNS > 0, "incorrect number of runs");
    static_assert(PERCENTILE_COUNT > 0, "incorrect number of percentiles");

    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::micro>;

    std::array<double, RUNS> samples{};

    // Run benchmark
    for (size_t i{0}; i < samples.size(); ++i) {
        auto const start{Clock::now()};
        TRY_EXPECTED_VOID(fn());
        auto const end{Clock::now()};

        samples[i] = std::chrono::duration_cast<Duration>(end - start).count();
    }

    // Sort samples
    std::sort(samples.begin(), samples.end());

    // Print results
    fmt::println("runs: {}", RUNS);

    for (size_t i{1}; i <= PERCENTILE_COUNT; ++i) {
        double const p{(100.0 * static_cast<double>(i)) / static_cast<double>(PERCENTILE_COUNT)};
        auto const index{static_cast<size_t>((p / 100.0) * static_cast<double>(RUNS - size_t{1}))};
        double const value{samples[index]};

        fmt::print("p{:5.1f}: {:.2f} us\n", p, value);
    }

    // Optional: min / max
    fmt::println("min: {:.2f} us", samples.front());
    fmt::println("max: {:.2f} us", samples.back());

    return {};
}

}  // namespace kernel::utils
