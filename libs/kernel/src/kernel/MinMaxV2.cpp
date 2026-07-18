#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <bit>
#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/MinMaxV2.hpp>
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

std::string const KERNEL_NAME_MINMAX{"minmax_reduction"};

}  // namespace

namespace kernel {

MinMaxV2::MinMaxV2(MinMaxV2&& other) noexcept {
    m_minmaxPipeline = other.m_minmaxPipeline;
    m_workgroupSizeX = other.m_workgroupSizeX;

    other.m_minmaxPipeline = VK_NULL_HANDLE;
    other.m_workgroupSizeX = 0;
}

auto MinMaxV2::init(uint32_t workgroupSizeX) noexcept -> std::expected<void, std::string> {
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

        if (!std::has_single_bit(workgroupSizeX)) {
            return std::unexpected{"workgroup size should be power of 2"};
        }
    }

    m_workgroupSizeX = workgroupSizeX;

    // create pipelines
    {
        TRY_EXPECTED(m_minmaxPipeline, utils::create_pipeline(KERNEL_NAME_MINMAX,  //
                                                              m_workgroupSizeX,  //
                                                              1,  //
                                                              1));
    }

    return {};
}

auto MinMaxV2::create(uint32_t workgroupSizeX) noexcept -> std::expected<MinMaxV2, std::string> {
    MinMaxV2 minmax{};

    TRY_EXPECTED_VOID(minmax.init(workgroupSizeX));

    return minmax;
}

auto MinMaxV2::operator()(std::span<float const> a  //
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

auto MinMaxV2::operator()(gpu::DeviceBuffer const& a,  //
                          gpu::DeviceBuffer const& b,  //
                          size_t sizeBytes  //
) const noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    // input validation
    {
        if (sizeBytes == 0) {
            return std::unexpected{"input buffer should have at least one element"};
        }

        if ((sizeBytes % sizeof(float)) != 0) {
            return std::unexpected{fmt::format("input size should be multiple of {}", sizeof(float))};
        }

        if (b.size() < (2 * sizeof(float))) {
            return std::unexpected{
                fmt::format("result buffer should have space for at least {} bytes", 2 * sizeof(float))};
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

    uint32_t n{static_cast<uint32_t>(sizeBytes / sizeof(float))};
    uint32_t groupCountX{utils::get_workgroupgroup_count(n, m_workgroupSizeX)};

    DeviceData intermediateDeviceData{};

    // allocate buffers for intermediate results
    {
        // each workgroup produces a min and a max
        size_t const bufferSize{groupCountX * 2 * sizeof(float)};

        TRY_EXPECTED_VOID(intermediateDeviceData.a.init(bufferSize));
        TRY_EXPECTED_VOID(intermediateDeviceData.b.init(bufferSize));
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_minmaxPipeline);

    gpu::DeviceBuffer const* inputBuffer{&a};
    gpu::DeviceBuffer const* outputBuffer{&intermediateDeviceData.a};

    bool firstPass{true};

    while (true) {
        // n is updated every iteration, so the group count does
        groupCountX = utils::get_workgroupgroup_count(n, m_workgroupSizeX);

        // last pass - use the provided output buffer instead of intermediate
        if (groupCountX == 1) {
            outputBuffer = &b;
        }

        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = inputBuffer->buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = n * sizeof(float);

            TRY_EXPECTED(uint32_t const descriptorIndexA,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = outputBuffer->buffer();
            bufferInfo.range = outputBuffer->size();

            TRY_EXPECTED(uint32_t const descriptorIndexB,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // see minmax_reduction.comp
            auto const pushConstData{gpu::utils::get_push_constant_data(n,  //
                                                                        descriptorIndexA,  //
                                                                        descriptorIndexB)};

            vkCmdPushConstants(commandBuffer,  //
                               gpuManager.pipelineLayout(),  //
                               VK_SHADER_STAGE_COMPUTE_BIT,  //
                               0,  //
                               static_cast<uint32_t>(pushConstData.size()),  //
                               pushConstData.data());
        }

        // compute
        vkCmdDispatch(commandBuffer, groupCountX, 1, 1);

        // barrier
        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       *outputBuffer,  //
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                       VK_ACCESS_2_SHADER_WRITE_BIT,  //
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                       VK_ACCESS_2_SHADER_READ_BIT);

        if (groupCountX == 1) {
            break;
        }

        n = groupCountX * 2;  // each group count outputs 2 floats - min and max

        if (firstPass) {
            firstPass = false;

            inputBuffer = outputBuffer;
            outputBuffer = &intermediateDeviceData.b;
        } else {
            std::swap(inputBuffer, outputBuffer);
        }
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return std::unexpected{"failed to end copy command buffer"};
    }

    TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer, gpuManager.computeQueue().queue));

    if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
        return std::unexpected{"failed to wait queue"};
    }

    return {};
}

auto MinMaxV2::read(gpu::DeviceBuffer const& bDevice,  //
                    std::optional<gpu::HostVisibleBuffer> stagingBuffer  //
) const noexcept -> std::expected<Result, std::string> {
    size_t constexpr SIZE_BYTES{2 * sizeof(float)};

    gpu::HostVisibleBuffer stagingBufferTmp{};

    // RAII cleanup, will do nothing if the temporary buffer was not initialized
    auto const guard{::utils::make_scope_guard([&] { stagingBufferTmp.destroy(); })};

    gpu::HostVisibleBuffer* readbackBuffer{};

    if (stagingBuffer) {
        readbackBuffer = &stagingBuffer.value();
    } else {
        TRY_EXPECTED_VOID(stagingBufferTmp.init(SIZE_BYTES, true));

        readbackBuffer = &stagingBufferTmp;
    }

    if (bDevice.size() < SIZE_BYTES) {
        return std::unexpected{fmt::format("result buffer should have space for at least {} bytes", SIZE_BYTES)};
    }

    if (readbackBuffer->size() < SIZE_BYTES) {
        return std::unexpected{fmt::format("staging buffer should have space for at least {} bytes", SIZE_BYTES)};
    }

    TRY_EXPECTED_VOID(gpu::utils::read_data_sync(bDevice, *readbackBuffer, SIZE_BYTES));

    std::array<uint8_t, SIZE_BYTES> data{};
    TRY_EXPECTED_VOID(readbackBuffer->copyFrom(std::as_writable_bytes(std::span{data}), SIZE_BYTES));

    Result result{};
    std::memcpy(&result.min, data.data(), 4);
    std::memcpy(&result.max, data.data() + 4, 4);

    return result;
}

auto MinMaxV2::destroy() noexcept -> void {
    auto gpuManagerResult{gpu::GpuManager::get()};

    if (!gpuManagerResult.has_value()) {
        spdlog::warn("failed to get gpu manager");
        return;
    }

    auto& gpuManager{gpuManagerResult.value().get()};

    vkDestroyPipeline(gpuManager.device(), m_minmaxPipeline, nullptr);
    m_minmaxPipeline = VK_NULL_HANDLE;

    m_workgroupSizeX = 0;
}

}  // namespace kernel
