#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/MinMaxV1.hpp>
#include <kernel/utils/pipeline_helper.hpp>
#include <kernel/utils/workgroup_helper.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace {

struct DeviceData {
    gpu::DeviceBuffer a{};
    gpu::DeviceBuffer b{};

    ~DeviceData() {
        a.destroy();
        b.destroy();
    }
};

auto ordered_uint_to_float(uint32_t u) noexcept -> float {
    uint32_t const v{(u & 0x80000000u) != 0u ? (u ^ 0x80000000u) : ~u};
    return std::bit_cast<float>(v);
}

// std::string const KERNEL_NAME_MINMAX{"minmax_uint_atomic_global"};
std::string const KERNEL_NAME_MINMAX{"minmax_uint_atomic_shared"};
std::string const KERNEL_NAME_INIT{"minmax_uint_init"};

}  // namespace

namespace kernel {

MinMaxV1::MinMaxV1(MinMaxV1&& other) noexcept {
    m_atomicUintInitPipeline = other.m_atomicUintInitPipeline;
    m_minmaxAtomicUintPipeline = other.m_minmaxAtomicUintPipeline;
    m_workgroupSizeX = other.m_workgroupSizeX;

    other.m_atomicUintInitPipeline = VK_NULL_HANDLE;
    other.m_minmaxAtomicUintPipeline = VK_NULL_HANDLE;
    other.m_workgroupSizeX = 0;
}

auto MinMaxV1::init(uint32_t workgroupSizeX) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    // check that requested workgroup size works for the selected device
    {
        if (workgroupSizeX == 0) {
            return std::unexpected{"workgroup size can't be 0"};
        }

        auto const& limits{gpuManager.physicalDeviceProperties().properties.limits};

        if (m_workgroupSizeX > limits.maxComputeWorkGroupSize[0]) {
            return std::unexpected{
                fmt::format("provided workgroup size x ({}) exceeds the maximum compute workgroup size ({})",
                            m_workgroupSizeX, limits.maxComputeWorkGroupSize[0])};
        }
    }

    m_workgroupSizeX = workgroupSizeX;

    // create pipelines
    {
        TRY_EXPECTED(m_minmaxAtomicUintPipeline, utils::create_pipeline(KERNEL_NAME_MINMAX,  //
                                                                        m_workgroupSizeX,  //
                                                                        1,  //
                                                                        1));

        TRY_EXPECTED(m_atomicUintInitPipeline,
                     utils::create_pipeline(KERNEL_NAME_INIT,  //
                                            2,  //
                                            1,  //
                                            1));
    }

    return {};
}

auto MinMaxV1::create(uint32_t workgroupSizeX) noexcept -> std::expected<MinMaxV1, std::string> {
    MinMaxV1 minmax{};

    TRY_EXPECTED_VOID(minmax.init(workgroupSizeX));

    return minmax;
}

auto MinMaxV1::operator()(std::span<float const> a  //
) const noexcept -> std::expected<Result, std::string> {
    auto const sizeBytes{a.size_bytes()};

    // create buffers
    DeviceData deviceData{};

    TRY_EXPECTED_VOID(deviceData.a.init(sizeBytes));
    TRY_EXPECTED_VOID(deviceData.b.init(2 * sizeof(float)));  // for min and max

    // copy data
    TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(deviceData.a,  //
                                                   ::utils::to_byte_span(a)));

    TRY_EXPECTED_VOID((*this)(deviceData.a, deviceData.b, sizeBytes));

    return read(deviceData.b);
}

auto MinMaxV1::operator()(gpu::DeviceBuffer const& a,  //
                          gpu::DeviceBuffer const& b,  //
                          size_t sizeBytes  //
) const noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    // input validation
    {
        if (sizeBytes == 0) {
            return std::unexpected("input buffer should have at least one element");
        }

        if ((sizeBytes % sizeof(float)) != 0) {
            return std::unexpected(fmt::format("input size should be multiple of {}", sizeof(float)));
        }

        if (b.size() < (2 * sizeof(float))) {
            return std::unexpected(
                fmt::format("result buffer should have space for at least {} bytes", 2 * sizeof(float)));
        }
    }

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] {
        gpuManager.storageDescriptorSetManager().reset();
        if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
            spdlog::warn("{}", r.error());
        }
    })};

    // init the output
    {
        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = b.buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = 8;

            TRY_EXPECTED(uint32_t const descriptorIndexIn,  //
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // see minmax_uint_init.comp
            auto const pushConstData{gpu::utils::get_push_constant_data(descriptorIndexIn)};

            vkCmdPushConstants(commandBuffer,  //
                               gpuManager.pipelineLayout(),  //
                               VK_SHADER_STAGE_COMPUTE_BIT,  //
                               0,  //
                               static_cast<uint32_t>(pushConstData.size()),  //
                               pushConstData.data());
        }

        // dispatch
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_atomicUintInitPipeline);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
    }

    // barrier
    gpu::utils::set_buffer_barrier(commandBuffer,  //
                                   b,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_WRITE_BIT,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // compute
    {
        // get number of elements
        uint32_t const n{static_cast<uint32_t>(sizeBytes / sizeof(float))};

        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = a.buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = sizeBytes;

            TRY_EXPECTED(uint32_t const descriptorIndexIn,  //
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = b.buffer();
            bufferInfo.range = 2 * sizeof(float);

            TRY_EXPECTED(uint32_t const descriptorIndexOut,  //
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // see minmax_uint_atomic_shared.comp
            auto const pushConstData{gpu::utils::get_push_constant_data(n,  //
                                                                        descriptorIndexIn,  //
                                                                        descriptorIndexOut)};

            vkCmdPushConstants(commandBuffer,  //
                               gpuManager.pipelineLayout(),  //
                               VK_SHADER_STAGE_COMPUTE_BIT,  //
                               0,  //
                               static_cast<uint32_t>(pushConstData.size()),  //
                               pushConstData.data());
        }

        // dispatch and wait
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_minmaxAtomicUintPipeline);

        uint32_t const groupCountX{utils::get_workgroupgroup_count(n, m_workgroupSizeX)};

        vkCmdDispatch(commandBuffer, groupCountX, 1, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            return std::unexpected{"failed to end copy command buffer"};
        }

        TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer, gpuManager.computeQueue().queue));

        if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
            return std::unexpected{"failed to wait queue"};
        }
    }

    return {};
}

auto MinMaxV1::read(gpu::DeviceBuffer const& bDevice,  //
                    std::optional<gpu::HostVisibleBuffer> stagingBuffer  //
) const noexcept -> std::expected<Result, std::string> {
    size_t constexpr SIZE_BYTES{2 * sizeof(float)};

    gpu::HostVisibleBuffer stagingBufferTmp{};

    // RAII cleanup, will do nothing if the temporary buffer was not initialized
    auto const guard{::utils::make_scope_guard([&] { stagingBufferTmp.destroy(); })};

    gpu::HostVisibleBuffer readbackBuffer{};

    if (stagingBuffer) {
        readbackBuffer = *stagingBuffer;
    } else {
        TRY_EXPECTED_VOID(stagingBufferTmp.init(SIZE_BYTES, true));

        readbackBuffer = stagingBufferTmp;
    }

    if (bDevice.size() < SIZE_BYTES) {
        return std::unexpected(fmt::format("result buffer should have space for at least {} bytes", SIZE_BYTES));
    }

    if (readbackBuffer.size() < SIZE_BYTES) {
        return std::unexpected(fmt::format("staging buffer should have space for at least {} bytes", SIZE_BYTES));
    }

    TRY_EXPECTED_VOID(gpu::utils::read_data_sync(bDevice, readbackBuffer, SIZE_BYTES));

    std::array<uint8_t, SIZE_BYTES> data{};
    TRY_EXPECTED_VOID(readbackBuffer.copyFrom(data.data(), SIZE_BYTES));

    uint32_t minUint{};
    std::memcpy(&minUint, data.data(), sizeof(float));

    uint32_t maxUint{};
    std::memcpy(&maxUint, data.data() + sizeof(float), sizeof(float));

    Result result{};
    result.min = ordered_uint_to_float(minUint);
    result.max = ordered_uint_to_float(maxUint);

    return result;
}

auto MinMaxV1::destroy() noexcept -> void {
    auto gpuManagerResult{gpu::GpuManager::get()};

    if (!gpuManagerResult.has_value()) {
        spdlog::warn("failed to get gpu manager");
        return;
    }

    auto& gpuManager{gpuManagerResult.value().get()};

    vkDestroyPipeline(gpuManager.device(), m_minmaxAtomicUintPipeline, nullptr);
    m_minmaxAtomicUintPipeline = VK_NULL_HANDLE;

    vkDestroyPipeline(gpuManager.device(), m_atomicUintInitPipeline, nullptr);
    m_atomicUintInitPipeline = VK_NULL_HANDLE;

    m_workgroupSizeX = 0;
}

}  // namespace kernel
