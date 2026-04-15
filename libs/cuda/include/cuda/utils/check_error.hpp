#include <fmt/core.h>

#define __CHECK_ERROR_OPT_IMPL(code, file, line)                                          \
    if (code != cudaSuccess) {                                                            \
        return std::make_optional<std::string>(                                           \
            fmt::format("Cuda error: {} {} {}\n", cudaGetErrorString(code), file, line)); \
    }

#define CHECK_ERROR_OPT(code) __CHECK_ERROR_OPT_IMPL(code, __FILE__, __LINE__)

#define __CHECK_ERROR_EXP_IMPL(code, file, line)                                                             \
    if (code != cudaSuccess) {                                                                               \
        return std::unexpected(fmt::format("Cuda error: {} {} {}\n", cudaGetErrorString(code), file, line)); \
    }

#define CHECK_ERROR_EXP(code) __CHECK_ERROR_EXP_IMPL(code, __FILE__, __LINE__)