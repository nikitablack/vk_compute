#pragma once

#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <optional>
#include <string>
#include <unordered_map>

namespace gpu {

class GpuManager;

}

namespace kernel {

class Add {
public:
    Add() = default;

public:
    static auto destroy() noexcept -> void;

    [[nodiscard]] static auto run(gpu::GpuManager& gpuManager,  //
                                  std::span<float const> a,  //
                                  std::span<float const> b,  //
                                  std::span<float> result  //
                                  ) noexcept -> std::expected<void, std::string>;

    [[nodiscard]] static auto run(gpu::GpuManager& gpuManager,  //
                                  gpu::DeviceBuffer const& a,  //
                                  gpu::DeviceBuffer const& b,  //
                                  gpu::DeviceBuffer const& result,  //
                                  std::optional<uint64_t> sizeBytes = std::nullopt  //
                                  ) noexcept -> std::expected<void, std::string>;

private:
    static std::unordered_map<VkDevice, VkPipeline> m_deviceToPipeline;
};

}  // namespace kernel
