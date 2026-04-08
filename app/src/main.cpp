#include <fmt/core.h>

#include <cstdlib>
#include <expected>
#include <gpu/GpuManager.hpp>
#include <string>
#include <utils/try_expected.hpp>

namespace {

auto main_impl() -> std::expected<void, std::string> {
    gpu::GpuManager gpuManager{};
    TRY_EXPECTED_VOID(gpuManager.initialize());

    // gpuManager.flush();
    // gpuManager.destroy();

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