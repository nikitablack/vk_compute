#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu::impl {

[[nodiscard]] auto get_compute_queue_family(VkPhysicalDevice device,  //
                                            uint32_t requiredQueueCount  //
                                            ) noexcept -> std::expected<uint32_t, std::string>;

}  // namespace gpu::impl
