#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda/add.hpp>
#include <iostream>

namespace {

void check_add(std::span<float const> a, std::span<float const> b, std::span<float const> out) {
    REQUIRE(a.size() == b.size());
    REQUIRE(a.size() == out.size());

    for (size_t i{0}; i < a.size(); ++i) {
        REQUIRE(out[i] == Catch::Approx(a[i] + b[i]));
    }
}

}  // namespace

TEST_CASE("add_chunked - 1024^2 elements add", "[add_chunked]") {
    size_t constexpr N{1024 * 1024};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    REQUIRE(cuda::add_chunked(a, b, result).has_value());

    check_add(a, b, result);
}

TEST_CASE("add_chunked - 100 elements add", "[add_chunked]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    REQUIRE(cuda::add_chunked(a, b, result, 128, cuda::utils::ChunkSize::C_64).has_value());

    check_add(a, b, result);
}

TEST_CASE("add_chunked - single element add", "[add_chunked]") {
    std::vector<float> a{1.5f};
    std::vector<float> b{2.5f};
    std::vector<float> result(1);

    REQUIRE(cuda::add(a, b, result).has_value());

    check_add(a, b, result);
}

TEST_CASE("add_chunked - zero elements add", "[add_chunked]") {
    std::vector<float> a{};
    std::vector<float> b{};
    std::vector<float> result{};

    REQUIRE(cuda::add(a, b, result).has_value());

    REQUIRE(result.empty());
}
