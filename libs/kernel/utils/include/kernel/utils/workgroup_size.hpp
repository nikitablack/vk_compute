#pragma once

#include <cstdint>

namespace kernel::utils {

enum class WorkGroupSize : uint32_t {
    WG_1 = 1,
    WG_2 = 2,
    WG_4 = 4,
    WG_8 = 8,
    WG_16 = 16,
    WG_32 = 32,
    WG_64 = 64,
    WG_128 = 128,
    WG_256 = 256,
    WG_512 = 512,
    WG_1024 = 1024,
};

}