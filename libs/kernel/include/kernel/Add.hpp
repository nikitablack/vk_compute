#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <string>
#include <vector>

namespace gpu {

class GpuManager;

}

namespace kernel {

class Add {
public:
    Add() = default;

public:
    auto destroy() noexcept -> void;

    [[nodiscard]] auto run(gpu::GpuManager& gpuManager,  //
                           std::span<float const> a,  //
                           std::span<float const> b,  //
                           std::vector<float>& out  //
                           ) noexcept -> std::expected<void, std::string>;

private:
    [[nodiscard]] auto runImpl(gpu::GpuManager& gpuManager,  //
                               uint32_t dataCount,  //
                               uint32_t dataSizeBytes,  //
                               uint32_t workgroupSizeX,  //
                               uint32_t workgroupSizeY,  //
                               uint32_t workgroupSizeZ  //
                               ) -> std::expected<void, std::string>;

private:
    VkDevice m_device{VK_NULL_HANDLE};
    gpu::DeviceBuffer m_bufferA{};
    gpu::DeviceBuffer m_bufferB{};
    gpu::DeviceBuffer m_bufferOut{};
    gpu::HostVisibleBuffer m_stagingBuffer{};
    VkPipeline m_pipeline{VK_NULL_HANDLE};
};

}  // namespace kernel
