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

TEST_CASE("Add - 100 elements add, workgroup size 32", "[Add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    auto add{kernel::Add::create(32)};
    REQUIRE(add.has_value());

    REQUIRE((*add)(a, b, c).has_value());

    check_add(a, b, c);

    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements add, workgroup size 64", "[Add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    auto add{kernel::Add::create(64)};
    REQUIRE(add.has_value());

    REQUIRE((*add)(a, b, c).has_value());

    check_add(a, b, c);

    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements add, workgroup size 512", "[Add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    auto add{kernel::Add::create(512)};
    REQUIRE(add.has_value());

    REQUIRE((*add)(a, b, c).has_value());

    check_add(a, b, c);

    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements add, workgroup size 1024", "[Add]") {
    size_t constexpr N{100};

    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);

    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    auto add{kernel::Add::create(32)};
    REQUIRE(add.has_value());

    REQUIRE((*add)(a, b, c).has_value());

    check_add(a, b, c);

    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - single element add", "[Add]") {
    std::vector<float> a{1.5f};
    std::vector<float> b{2.5f};
    std::vector<float> c(1);

    auto add{kernel::Add::create()};
    REQUIRE(add.has_value());

    REQUIRE((*add)(a, b, c).has_value());

    check_add(a, b, c);

    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - zero elements add", "[Add]") {
    std::vector<float> a{};
    std::vector<float> b{};
    std::vector<float> c{};

    auto add{kernel::Add::create(32)};
    REQUIRE(add.has_value());

    REQUIRE((*add)(a, b, c).has_value());

    REQUIRE(c.empty());

    add->destroy();
    gpu::GpuManager::destroy();
}
