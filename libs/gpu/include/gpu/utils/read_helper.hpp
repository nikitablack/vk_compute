#pragma once

#include <expected>
#include <string>

// forward declarations
namespace gpu {

class GpuManager;
class DeviceBuffer;
class HostVisibleBuffer;

}  // namespace gpu

namespace gpu::utils {

[[nodiscard]] auto read_data_sync(GpuManager& gpuManager,  //
                                  DeviceBuffer const& src,  //
                                  HostVisibleBuffer const& dst,  // dst MUST be resized to fit the src data!
                                  uint32_t size  //
                                  ) -> std::expected<void, std::string>;

}  // namespace gpu::utils
