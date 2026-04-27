#include <fmt/core.h>

#include <algorithm>
#include <cstdlib>
#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <kernel/add.hpp>
#include <ranges>
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
    TRY_EXPECTED_VOID(gpu::GpuManager::init());

    uint32_t constexpr N{1024 * 1024 * 100};
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
    gpu::DeviceBuffer aDevice{};
    gpu::DeviceBuffer bDevice{};
    gpu::DeviceBuffer resultDevice{};

    TRY_EXPECTED_VOID(aDevice.init(S));
    TRY_EXPECTED_VOID(bDevice.init(S));
    TRY_EXPECTED_VOID(resultDevice.init(S));

    // copy host data to device
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(aDevice,  //
                                                       ::utils::to_byte_span(aHost)));

        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(bDevice,  //
                                                       ::utils::to_byte_span(bHost)));
    }

    // compute
    TRY_EXPECTED_VOID(utils::benchmark_with_percentiles([&]() -> std::expected<void, std::string> {
        TRY_EXPECTED_VOID(kernel::add(aDevice, bDevice, resultDevice));
        return {};
    }));

    // read result
    std::vector<float> resultHost(N);
    gpu::HostVisibleBuffer stagingBuffer{};

    {
        TRY_EXPECTED_VOID(stagingBuffer.init(S, true));
        TRY_EXPECTED_VOID(gpu::utils::read_data_sync(resultDevice, stagingBuffer, S));
        TRY_EXPECTED_VOID(stagingBuffer.copyFrom(resultHost.data(), S));
    }

#ifdef VK_ENABLE_RENDERDOC_DEBUG
    if (renderdocApi) {
        renderdocApi->EndFrameCapture(nullptr, nullptr);
    }
#endif

    // clear
    aDevice.destroy();
    bDevice.destroy();
    resultDevice.destroy();
    stagingBuffer.destroy();
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
