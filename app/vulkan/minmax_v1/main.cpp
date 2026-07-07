#include <fmt/core.h>

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <kernel/MinMaxV1.hpp>
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
    uint32_t constexpr N{1024 * 1024};
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
    gpu::DeviceBuffer aDevice{};
    gpu::DeviceBuffer bDevice{};

    TRY_EXPECTED_VOID(aDevice.init(S));
    TRY_EXPECTED_VOID(bDevice.init(8));

    // copy host data to device
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(aDevice,  //
                                                       ::utils::to_byte_span(inHost)));
    }

    TRY_EXPECTED(auto minmax, (kernel::MinMaxV1::create(128)));

    // compute
    TRY_EXPECTED(auto const percentiles, utils::benchmark_with_percentiles([&]() -> std::expected<void, std::string> {
                     TRY_EXPECTED_VOID(minmax(aDevice, bDevice, S));
                     return {};
                 }));

    for (auto const& p : percentiles) {
        fmt::print("p{:5.1f}: {:.2f} us\n", p.first, p.second);
    }

    // read result
    TRY_EXPECTED(auto const result, minmax.read(bDevice));
    fmt::println("min: {}, max: {}", result.min, result.max);

#ifdef VK_ENABLE_RENDERDOC_DEBUG
    if (renderdocApi) {
        renderdocApi->EndFrameCapture(nullptr, nullptr);
    }
#endif

    // clear
    aDevice.destroy();
    bDevice.destroy();
    minmax.destroy();
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
