#pragma once

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto check_required_instance_extensions() noexcept -> std::expected<void, std::string>;

}  // namespace gpu::impl
