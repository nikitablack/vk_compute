#pragma once

#include <cstdint>

namespace gpu::impl {

struct RequiredApiVersion {
    static uint32_t constexpr MAJOR{1};
    static uint32_t constexpr MINOR{4};
};

}  // namespace gpu::impl
