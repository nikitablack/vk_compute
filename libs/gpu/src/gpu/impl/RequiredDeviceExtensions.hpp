#pragma once

#include <string>
#include <vector>

namespace gpu::impl {

struct RequiredDeviceExtensions {
    static auto get() noexcept -> std::vector<std::string>;
    static auto print() noexcept -> void;
};

}  // namespace gpu::impl
