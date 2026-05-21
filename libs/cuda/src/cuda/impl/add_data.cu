#include <spdlog/spdlog.h>

#include <cuda/impl/add_data.hpp>

namespace cuda::impl {

AddData::~AddData() {
    if (a) {
        if (cudaFree(a) != cudaSuccess) {
            spdlog::warn("failed to free cuda memory");
        }
    }

    if (b) {
        if (cudaFree(b) != cudaSuccess) {
            spdlog::warn("failed to free cuda memory");
        }
    }

    if (result) {
        if (cudaFree(result) != cudaSuccess) {
            spdlog::warn("failed to free cuda memory");
        }
    }
}

}  // namespace cuda::impl
