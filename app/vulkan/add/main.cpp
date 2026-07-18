#include <fmt/core.h>

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <gpu/Buffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <kernel/Add.hpp>
#include <ranges>
#include <string>
#include <utils/benchmark_with_percentiles.hpp>
#include <utils/try_expected.hpp>

#ifdef VK_ENABLE_RENDERDOC_DEBUG
#include <renderdoc_app.h>

#ifdef __linux
#include <dlfcn.h>
#endif
#endif

namespace {

auto main_impl() -> std::expected<void, std::string> {
    // uint32_t constexpr N{1024 * 1024 * 100};
    uint32_t constexpr N{5};
    uint32_t constexpr S{N * sizeof(float)};

    // initialize host memory
    std::vector<float> aHost(N);
    std::vector<float> bHost(N);

    // initizlize host data
    {
        std::ranges::copy(
            std::views::iota(uint32_t{0}, N) | std::views::transform([](auto i) { return static_cast<float>(i); }),
            aHost.begin());

        std::ranges::copy(
            std::views::iota(uint32_t{0}, N) | std::views::transform([](auto i) { return static_cast<float>(i); }),
            bHost.begin());
    }

#ifdef VK_ENABLE_RENDERDOC_DEBUG
    RENDERDOC_API_1_7_0* renderdocApi{nullptr};

    if (void* mod{dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)}) {
        auto const getAPI{reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"))};
        [[maybe_unused]] int ret{getAPI(eRENDERDOC_API_Version_1_7_0, reinterpret_cast<void**>(&renderdocApi))};
    }

    if (renderdocApi) {
        renderdocApi->StartFrameCapture(nullptr, nullptr);
    }
#endif

    // initialize device memory
    TRY_EXPECTED(auto aDevice, gpu::Buffer::create(std::span<float const>{aHost}, gpu::Buffer::Type::Device));
    TRY_EXPECTED(auto bDevice, gpu::Buffer::create(std::span<float const>{bHost}, gpu::Buffer::Type::Device));
    TRY_EXPECTED(auto cDevice, gpu::Buffer::create(S, gpu::Buffer::Type::Device));

    // create kernel
    TRY_EXPECTED(auto add, kernel::Add::create(32));

    // compute and run benchmarks
    TRY_EXPECTED(auto const percentiles, utils::benchmark_with_percentiles([&]() -> std::expected<void, std::string> {
                     TRY_EXPECTED_VOID(add(aDevice, bDevice, cDevice));
                     return {};
                 }));

    for (auto const& p : percentiles) {
        fmt::print("p{:5.1f}: {:.2f} us\n", p.first, p.second);
    }

    // read result
    std::vector<float> cHost(N);
    TRY_EXPECTED_VOID(add.read(cHost, cDevice));

#ifdef VK_ENABLE_RENDERDOC_DEBUG
    if (renderdocApi) {
        renderdocApi->EndFrameCapture(nullptr, nullptr);
    }
#endif

    // clear
    aDevice.destroy();
    bDevice.destroy();
    cDevice.destroy();
    add.destroy();
    gpu::GpuManager::destroy();

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
