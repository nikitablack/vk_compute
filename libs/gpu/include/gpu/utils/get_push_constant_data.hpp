#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace gpu::utils {

template <typename T>
auto append_push_constant_data(T t, uint8_t* dataPtr, size_t& offsetRef) -> void {
    auto dataSize = sizeof(t);
    std::memcpy(dataPtr + offsetRef, &t, dataSize);
    offsetRef += dataSize;
}

template <typename... Args, size_t N = (sizeof(Args) + ...)>
auto get_push_constant_data(Args&&... args) -> std::array<uint8_t, N> {
    static_assert(N <= 256, "push constant data can't exceed 256 bytes");

    std::array<uint8_t, N> data{};
    size_t offset{0};

    (append_push_constant_data(std::forward<Args>(args), data.data(), offset), ...);

    return data;
}

}  // namespace gpu::utils
