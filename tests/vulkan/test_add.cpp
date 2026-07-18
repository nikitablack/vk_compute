#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <gpu/Buffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/init_helper.hpp>
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

TEST_CASE("Add - 100 elements, workgroup size 32", "[Add]") {
    size_t constexpr N{100};

    // init host buffers
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);
    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    // init kernel
    auto add{kernel::Add::create(32)};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(a, b, c).has_value());

    // check
    check_add(a, b, c);

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements, workgroup size 64", "[Add]") {
    size_t constexpr N{100};

    // init host buffers
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);
    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    // init kernel
    auto add{kernel::Add::create(64)};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(a, b, c).has_value());

    // check
    check_add(a, b, c);

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements, workgroup size 512", "[Add]") {
    size_t constexpr N{100};

    // init host buffers
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);
    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    // init kernel
    auto add{kernel::Add::create(512)};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(a, b, c).has_value());

    // check
    check_add(a, b, c);

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements, workgroup size 1024", "[Add]") {
    size_t constexpr N{100};

    // init host buffers
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);
    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    // init kernel
    auto add{kernel::Add::create(1024)};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(a, b, c).has_value());

    // check
    check_add(a, b, c);

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - single element", "[Add]") {
    // init host buffers
    std::vector<float> a{1.5f};
    std::vector<float> b{2.5f};
    std::vector<float> c(1);

    // init kernel
    auto add{kernel::Add::create()};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(a, b, c).has_value());

    // check
    check_add(a, b, c);

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - zero elements", "[Add]") {
    // init host buffers
    std::vector<float> a{};
    std::vector<float> b{};
    std::vector<float> c{};

    // init kernel
    auto add{kernel::Add::create()};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(a, b, c).has_value());

    // check
    REQUIRE(c.empty());

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements, single overlapped host buffer for input and output", "[Add]") {
    size_t constexpr N{100};

    // init host buffers
    std::vector<float> a(N);
    std::vector<float> aOriginal(N);
    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        aOriginal[i] = a[i];
    }

    // init kernel
    auto add{kernel::Add::create()};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(a, a, a).has_value());

    // check
    check_add(aOriginal, aOriginal, a);

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements, single overlapped device buffer for input and output", "[Add]") {
    size_t constexpr N{100};

    // init host buffer
    std::vector<float> aHost(N);

    // init device buffer
    auto aDevice{gpu::Buffer::create(N * sizeof(float), gpu::Buffer::Type::Device)};
    REQUIRE(aDevice.has_value());

    // init kernel
    auto add{kernel::Add::create()};
    REQUIRE(add.has_value());

    // compute - should fail since buffer overlapping is not allowed
    REQUIRE_FALSE((*add)(*aDevice, *aDevice, *aDevice).has_value());

    // clear
    aDevice->destroy();
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements, single host buffer with distinct offsets for input and output", "[Add]") {
    size_t constexpr N{100};

    // init host buffer
    std::vector<float> a(N * 3);
    for (size_t i{0}; i < N; ++i) {
        a[i] = static_cast<float>(i);
        a[i + N] = static_cast<float>(i) * 2.0f;
    }

    // init kernel
    auto add{kernel::Add::create()};
    REQUIRE(add.has_value());

    // compute
    REQUIRE((*add)(std::span{a}.subspan(0, N),  //
                   std::span{a}.subspan(N, N),  //
                   std::span{a}.subspan(2 * N, N))
                .has_value());

    // check
    check_add(std::span{a}.subspan(0, N),  //
              std::span{a}.subspan(N, N),  //
              std::span{a}.subspan(2 * N, N));

    // clear
    add->destroy();
    gpu::GpuManager::destroy();
}

TEST_CASE("Add - 100 elements, single device buffer with distinct offsets for input and output", "[Add]") {
    size_t constexpr N{100};

    // init host buffer
    std::vector<float> aHost(N);

    for (size_t i{0}; i < N; ++i) {
        aHost[i] = static_cast<float>(i);
    }

    // init device buffer
    auto aDevice{gpu::Buffer::create(3 * N * sizeof(float), gpu::Buffer::Type::Device)};
    REQUIRE(aDevice.has_value());

    REQUIRE(aDevice->copyToBuffer(std::as_bytes(std::span{aHost}), 0).has_value());
    REQUIRE(aDevice->copyToBuffer(std::as_bytes(std::span{aHost}), N * sizeof(float)).has_value());

    // init kernel
    auto add{kernel::Add::create()};
    REQUIRE(add.has_value());

    // get spans
    auto const aDeviceSpan{gpu::BufferSpan::create(*aDevice, 0, N * sizeof(float))};
    REQUIRE(aDeviceSpan.has_value());

    auto const bDeviceSpan{gpu::BufferSpan::create(*aDevice, N * sizeof(float), N * sizeof(float))};
    REQUIRE(bDeviceSpan.has_value());

    auto const dstDeviceSpan{gpu::BufferSpan::create(*aDevice, 2 * N * sizeof(float), N * sizeof(float))};
    REQUIRE(dstDeviceSpan.has_value());

    // compute
    REQUIRE((*add)(*aDeviceSpan, *bDeviceSpan, *dstDeviceSpan).has_value());

    // read result
    std::vector<float> dstHost(N);
    REQUIRE(add->read(dstHost, *dstDeviceSpan).has_value());

    // check
    check_add(aHost, aHost, dstHost);

    // clear
    aDevice->destroy();
    add->destroy();
    gpu::GpuManager::destroy();
}
