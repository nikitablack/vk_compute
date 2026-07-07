#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <gpu/GpuManager.hpp>
#include <iostream>
#include <kernel/MinMaxV2.hpp>
#include <limits>
#include <random>

namespace {

void check(std::span<float const> in, float min, float max) {
    auto const [minIn, maxIn]{std::minmax_element(in.begin(), in.end())};

    REQUIRE(*minIn == min);
    REQUIRE(*maxIn == max);
}

}  // namespace

TEST_CASE("MinMaxV2 - 100 elements in range [-100.0f, 100.0f], default workgroup size", "[MinMaxV2]") {
    size_t constexpr N{100};

    std::vector<float> a(N);

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::uniform_real_distribution<float> dist{-100.0f, 100.0f};

    for (float& v : a) {
        v = dist(rng);
    }

    auto minmax{kernel::MinMaxV2::create()};
    REQUIRE(minmax.has_value());

    auto const result{(*minmax)(a)};
    REQUIRE(result.has_value());

    check(a, result->min, result->max);

    minmax->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("MinMaxV2 - 100 only negative elements in range [-1000.0f, -100.0f], default workgroup size", "[MinMaxV2]") {
    size_t constexpr N{100};

    std::vector<float> a(N);

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::uniform_real_distribution<float> dist{-1000.0f, -100.0f};

    for (float& v : a) {
        v = dist(rng);
    }

    auto minmax{kernel::MinMaxV2::create()};
    REQUIRE(minmax.has_value());

    auto const result{(*minmax)(a)};
    REQUIRE(result.has_value());

    check(a, result->min, result->max);

    minmax->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("MinMaxV2 - 100 only positive elements in range [100.0f, 1000.0f], default workgroup size", "[MinMaxV2]") {
    size_t constexpr N{100};

    std::vector<float> a(N);

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::uniform_real_distribution<float> dist{100.0f, 1000.0f};

    for (float& v : a) {
        v = dist(rng);
    }

    auto minmax{kernel::MinMaxV2::create()};
    REQUIRE(minmax.has_value());

    auto const result{(*minmax)(a)};
    REQUIRE(result.has_value());

    check(a, result->min, result->max);

    minmax->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("MinMaxV2 - 10 elements with single negative infinity, default workgroup size", "[MinMaxV2]") {
    std::vector<float> a(10);
    a[5] = -std::numeric_limits<float>::infinity();

    auto minmax{kernel::MinMaxV2::create()};
    REQUIRE(minmax.has_value());

    auto const result{(*minmax)(a)};
    REQUIRE(result.has_value());

    check(a, result->min, result->max);

    minmax->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("MinMaxV2 - 10 elements with single positive infinity, default workgroup size", "[MinMaxV2]") {
    std::vector<float> a(10);
    a[5] = std::numeric_limits<float>::infinity();

    auto minmax{kernel::MinMaxV2::create()};
    REQUIRE(minmax.has_value());

    auto const result{(*minmax)(a)};
    REQUIRE(result.has_value());

    check(a, result->min, result->max);

    minmax->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("MinMaxV2 - empty input", "[MinMaxV2]") {
    std::vector<float> a{};

    auto minmax{kernel::MinMaxV2::create()};
    REQUIRE(minmax.has_value());

    auto const result{(*minmax)(a)};
    REQUIRE_FALSE(result.has_value());

    minmax->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("MinMaxV2 - 10 elements with non-power-of-2 workgroup size", "[MinMaxV2]") {
    std::vector<float> a(10);

    auto minmax{kernel::MinMaxV2::create(31)};
    if (minmax.has_value()) {
        std::cout << "AAAAAAAAAAA" << std::endl;
    } else {
        std::cout << "BBBBBBBBBBBBBBBB" << std::endl;
    }
    REQUIRE_FALSE(minmax.has_value());

    gpu::GpuManager::destroy();
}
