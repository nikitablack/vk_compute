#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <gpu/GpuManager.hpp>
#include <kernel/minmax/minmax_atomic.hpp>
#include <limits>
#include <random>

namespace {

void check(std::span<float const> in, float min, float max) {
    auto const [minIn, maxIn]{std::minmax_element(in.begin(), in.end())};

    REQUIRE(*minIn == min);
    REQUIRE(*maxIn == max);
}

}  // namespace

TEST_CASE("minmax_atomic_uint - 100 elements in range [-100.0f, 100.0f], default workgroup size",
          "[minmax_atomic_uint]") {
    size_t constexpr N{100};

    std::vector<float> in(N);

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::uniform_real_distribution<float> dist{-100.0f, 100.0f};

    for (float& v : in) {
        v = dist(rng);
    }

    auto const result{kernel::minmax::run_atomic(in)};

    REQUIRE(result.has_value());

    check(in, result->min, result->max);

    gpu::GpuManager::destroy();
}

TEST_CASE("minmax_atomic_uint - 100 only negative elements in range [-1000.0f, -100.0f], default workgroup size",
          "[minmax_atomic_uint]") {
    size_t constexpr N{100};

    std::vector<float> in(N);

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::uniform_real_distribution<float> dist{-1000.0f, -100.0f};

    for (float& v : in) {
        v = dist(rng);
    }

    auto const result{kernel::minmax::run_atomic(in)};

    REQUIRE(result.has_value());

    check(in, result->min, result->max);

    gpu::GpuManager::destroy();
}

TEST_CASE("minmax_atomic_uint - 100 only positive elements in range [100.0f, 1000.0f], default workgroup size",
          "[minmax_atomic_uint]") {
    size_t constexpr N{100};

    std::vector<float> in(N);

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::uniform_real_distribution<float> dist{100.0f, 1000.0f};

    for (float& v : in) {
        v = dist(rng);
    }

    auto const result{kernel::minmax::run_atomic(in)};

    REQUIRE(result.has_value());

    check(in, result->min, result->max);

    gpu::GpuManager::destroy();
}

TEST_CASE("minmax_atomic_uint - 10 elements with single negative infinity, default workgroup size",
          "[minmax_atomic_uint]") {
    std::vector<float> in(10);
    in[5] = -std::numeric_limits<float>::infinity();

    auto const result{kernel::minmax::run_atomic(in)};

    REQUIRE(result.has_value());

    check(in, result->min, result->max);

    gpu::GpuManager::destroy();
}

TEST_CASE("minmax_atomic_uint - 10 elements with single positive infinity, default workgroup size",
          "[minmax_atomic_uint]") {
    std::vector<float> in(10);
    in[5] = std::numeric_limits<float>::infinity();

    auto const result{kernel::minmax::run_atomic(in)};

    REQUIRE(result.has_value());

    check(in, result->min, result->max);

    gpu::GpuManager::destroy();
}

TEST_CASE("minmax_atomic_uint - empty input", "[minmax_atomic_uint]") {
    std::vector<float> in{};

    auto const result{kernel::minmax::run_atomic(in)};

    REQUIRE_FALSE(result.has_value());

    gpu::GpuManager::destroy();
}
