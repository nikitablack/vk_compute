#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <gpu/GpuManager.hpp>
#include <kernel/Add.hpp>

namespace {

void check_add(std::span<float const> a, std::span<float const> b, std::span<float const> out) {
    REQUIRE(a.size() == b.size());
    REQUIRE(a.size() == out.size());

    for (size_t i{0}; i < a.size(); ++i) {
        REQUIRE(out[i] == Catch::Approx(a[i] + b[i]));
    }
}

}  // namespace

TEST_CASE("add - 100 elements add", "[add]") {
    gpu::GpuManager gpuManager{};
    REQUIRE(gpuManager.initialize().has_value());

    kernel::Add add{};

    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    REQUIRE(add.run(gpuManager, a, b, result).has_value());

    check_add(a, b, result);
}

TEST_CASE("add - single element add", "[add]") {
    gpu::GpuManager gpuManager{};
    REQUIRE(gpuManager.initialize().has_value());

    kernel::Add add{};

    std::vector<float> a{1.5f};
    std::vector<float> b{2.5f};
    std::vector<float> result(1);

    REQUIRE(add.run(gpuManager, a, b, result).has_value());

    check_add(a, b, result);
}

TEST_CASE("add - zero elements add", "[add]") {
    gpu::GpuManager gpuManager{};
    REQUIRE(gpuManager.initialize().has_value());

    kernel::Add add{};

    std::vector<float> a{};
    std::vector<float> b{};
    std::vector<float> result{};

    REQUIRE(add.run(gpuManager, a, b, result).has_value());

    REQUIRE(result.empty());
}

TEST_CASE("add - non-multiple of 4 add", "[add]") {
    gpu::GpuManager gpuManager{};
    REQUIRE(gpuManager.initialize().has_value());

    kernel::Add add{};

    size_t constexpr N{10};  // not divisible by 4
    static_assert((N % 4) > 0, "number of elements should be non-multiple of 4");

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(i + 2);
    }

    REQUIRE(add.run(gpuManager, a, b, result).has_value());

    check_add(a, b, result);
}
