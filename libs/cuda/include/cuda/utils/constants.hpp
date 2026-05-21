#pragma once

#include <cstdint>

namespace cuda::utils {

enum class ChunkSize : uint32_t {
    C_1 = 1,
    C_2 = 2,
    C_4 = 4,
    C_8 = 8,
    C_16 = 16,
    C_32 = 32,
    C_64 = 64,
    C_128 = 128,
    C_256 = 256,
    C_512 = 512,
    C_1024 = 1024,
};

}
