#include <fmt/core.h>

#include <cstdlib>
#include <expected>
#include <gpu/GpuManager.hpp>
#include <matrix_add/MatrixAdd.hpp>
#include <string>
#include <utils/try_expected.hpp>

namespace {

auto main_impl() -> std::expected<void, std::string> {
    gpu::GpuManager gpuManager{};
    TRY_EXPECTED_VOID(gpuManager.initialize());

    std::vector<float> a{1.0f, 2.0f, 3.0f};
    std::vector<float> b{4.0f, 5.0f, 6.0f};
    std::vector<float> result{};

    matrix_add::MatrixAdd matrixAdd{};
    TRY_EXPECTED_VOID(matrixAdd.run(gpuManager, a, b, result));
    fmt::println("{} {} {}", result[0], result[1], result[2]);

    gpuManager.flush();
    matrixAdd.destroy();
    gpuManager.destroy();

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