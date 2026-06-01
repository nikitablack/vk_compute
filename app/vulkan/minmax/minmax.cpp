#include <fmt/core.h>

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <kernel/minmax.hpp>
#include <random>
#include <string>
#include <utils/benchmark_with_percentiles.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

#ifdef VK_ENABLE_RENDERDOC_DEBUG
#include <renderdoc_app.h>

#ifdef __linux
#include <dlfcn.h>
#endif
#endif

namespace {

auto main_impl() -> std::expected<void, std::string> {
    uint32_t constexpr N{10};
    uint32_t constexpr S{N * sizeof(float)};

    // initialize host memory
    std::vector<float> inHost(N);

    // initizlize host data
    {
        std::random_device rd{};
        std::mt19937 rng{rd()};
        std::uniform_real_distribution<float> dist{-10.0f, 10.0f};

        for (float& v : inHost) {
            v = dist(rng);
            fmt::println("{}", v);
        }
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
    gpu::DeviceBuffer inDevice{};
    gpu::DeviceBuffer resultDevice{};

    TRY_EXPECTED_VOID(inDevice.init(S));
    TRY_EXPECTED_VOID(resultDevice.init(8));

    // copy host data to device
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(inDevice,  //
                                                       ::utils::to_byte_span(inHost)));
    }

    // compute
    TRY_EXPECTED(auto const percentiles, utils::benchmark_with_percentiles(
                                             [&]() -> std::expected<void, std::string> {
                                                 TRY_EXPECTED_VOID(kernel::minmax(inDevice, resultDevice));
                                                 return {};
                                             },
                                             1));

    for (auto const& p : percentiles) {
        fmt::print("p{:5.1f}: {:.2f} us\n", p.first, p.second);
    }

    // read result
    TRY_EXPECTED([[maybe_unused]] auto const minmax, kernel::read_minmax_result(resultDevice));
    fmt::println("min: {}, max: {}", minmax.min, minmax.max);

#ifdef VK_ENABLE_RENDERDOC_DEBUG
    if (renderdocApi) {
        fmt::println("AAAAAAAAAAAAAAAAAa");
        renderdocApi->EndFrameCapture(nullptr, nullptr);
    }
#endif

    // clear
    inDevice.destroy();
    resultDevice.destroy();
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
