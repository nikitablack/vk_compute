#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <gpu/GpuManager.hpp>
#include <kernel/add.hpp>

namespace {

void check_add(std::span<float const> a, std::span<float const> b, std::span<float const> out) {
    REQUIRE(a.size() == b.size());
    REQUIRE(a.size() == out.size());

    for (size_t i{0}; i < a.size(); ++i) {
        REQUIRE(out[i] == Catch::Approx(a[i] + b[i]));
    }
}

}  // namespace

TEST_CASE("add - 100 elements add, workgroup size 32", "[add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    REQUIRE(kernel::add(a, b, result, 32).has_value());

    check_add(a, b, result);

    gpu::GpuManager::destroy();
}

TEST_CASE("add - 100 elements add, workgroup size 64", "[add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    REQUIRE(kernel::add(a, b, result, 64).has_value());

    check_add(a, b, result);

    gpu::GpuManager::destroy();
}

TEST_CASE("add - 100 elements add, workgroup size 512", "[add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    REQUIRE(kernel::add(a, b, result, 512).has_value());

    check_add(a, b, result);

    gpu::GpuManager::destroy();
}

TEST_CASE("add - 100 elements add, workgroup size 1024", "[add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    REQUIRE(kernel::add(a, b, result, 1024).has_value());

    check_add(a, b, result);

    gpu::GpuManager::destroy();
}

TEST_CASE("add - 100 elements add, workgroup size 2048", "[add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> result(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    // this should fail because likely 2048 is wrong since max is 1024
    // TODO: add limits check to the add() implementation.
    REQUIRE_FALSE(kernel::add(a, b, result, 2048).has_value());

    gpu::GpuManager::destroy();
}

TEST_CASE("add - single element add", "[add]") {
    std::vector<float> a{1.5f};
    std::vector<float> b{2.5f};
    std::vector<float> result(1);

    REQUIRE(kernel::add(a, b, result).has_value());

    check_add(a, b, result);

    gpu::GpuManager::destroy();
}

TEST_CASE("add - zero elements add", "[add]") {
    std::vector<float> a{};
    std::vector<float> b{};
    std::vector<float> result{};

    REQUIRE(kernel::add(a, b, result).has_value());

    REQUIRE(result.empty());

    gpu::GpuManager::destroy();
}
