#pragma once

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto check_instance_version() noexcept -> std::expected<void, std::string>;

}  // namespace gpu::impl
