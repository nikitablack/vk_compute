#pragma once

#include <vector_types.h>

namespace cuda::impl {

struct AddData {
    float* a{nullptr};
    float* b{nullptr};
    float* result{nullptr};

    ~AddData();
};

}  // namespace cuda::impl
