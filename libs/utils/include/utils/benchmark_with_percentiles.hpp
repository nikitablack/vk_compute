#pragma once

#include <fmt/core.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <utility>
#include <utils/try_expected.hpp>

namespace utils {

[[nodiscard]] auto benchmark_with_percentiles(std::function<std::expected<void, std::string>()> const& fn,  //
                                              uint32_t runs = 100,  //
                                              uint32_t percentileCount = 10  //
                                              ) -> std::expected<std::vector<std::pair<float, double>>, std::string> {
    if (runs == 0) {
        return std::unexpected{"incorrect number of runs"};
    }

    if (percentileCount == 0) {
        return std::unexpected{"incorrect number of percentiles"};
    }

    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::micro>;

    std::vector<double> samples{};
    samples.reserve(runs);

    // Run benchmark
    for (size_t i{0}; i < runs; ++i) {
        auto const start{Clock::now()};
        TRY_EXPECTED_VOID(fn());
        auto const end{Clock::now()};

        samples.push_back(std::chrono::duration_cast<Duration>(end - start).count());
    }

    // Sort samples
    std::sort(samples.begin(), samples.end());

    std::vector<std::pair<float, double>> percentiles{};
    percentiles.reserve(percentileCount);
    for (size_t i{1}; i <= percentileCount; ++i) {
        double const p{(100.0 * static_cast<double>(i)) / static_cast<double>(percentileCount)};
        auto const index{static_cast<size_t>((p / 100.0) * static_cast<double>(runs - size_t{1}))};
        double const value{samples[index]};

        percentiles.emplace_back(static_cast<float>(p), value);
    }

    return percentiles;
}

}  // namespace utils
