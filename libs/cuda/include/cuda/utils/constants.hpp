#pragma once

#include <cstdint>

namespace cuda::utils {

enum class ItCount : uint32_t {
    IT_1 = 1,
    IT_2 = 2,
    IT_4 = 4,
    IT_8 = 8,
    IT_16 = 16,
    IT_32 = 32,
    IT_64 = 64,
    IT_128 = 128,
    IT_256 = 256,
    IT_512 = 512,
    IT_1024 = 1024,
};

}
