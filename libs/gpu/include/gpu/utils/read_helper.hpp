#pragma once

#include <expected>
#include <string>

// forward declarations
namespace gpu {

class DeviceBuffer;
class HostVisibleBuffer;
class Span;

}  // namespace gpu

namespace gpu::utils {

[[nodiscard]] auto read_data_sync(Span const& dst,  //
                                  Span const& src  //
                                  ) -> std::expected<void, std::string>;

[[nodiscard]] auto read_data_sync(DeviceBuffer const& src,  //
                                  HostVisibleBuffer const& dst,  //
                                  size_t sizeBytes,  //
                                  size_t srcOffset = 0,  //
                                  size_t dstOffset = 0  //
                                  ) -> std::expected<void, std::string>;

}  // namespace gpu::utils
