#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/minmax/minmax_reduction.hpp>
#include <kernel/utils/pipeline_helper.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace {

std::string const KERNEL_NAME{"minmax_reduction"};

struct DeviceData {
    gpu::DeviceBuffer in{};
    gpu::DeviceBuffer result{};

    ~DeviceData() {
        in.destroy();
        result.destroy();
    }
};

struct IntermediateDeviceData {
    gpu::DeviceBuffer intermediateA{};
    gpu::DeviceBuffer intermediateB{};

    ~IntermediateDeviceData() {
        intermediateA.destroy();
        intermediateB.destroy();
    }
};

auto read_impl(gpu::DeviceBuffer const& resultDevice,  //
               gpu::HostVisibleBuffer& stagingBuffer  //
               ) -> std::expected<kernel::minmax::Result, std::string> {
    uint32_t constexpr SIZE{8};

    if (resultDevice.size() < SIZE) {
        return std::unexpected(fmt::format("result buffer should have space for at least {} bytes", SIZE));
    }

    if (stagingBuffer.size() < SIZE) {
        return std::unexpected(fmt::format("staging buffer should have space for at least {} bytes", SIZE));
    }

    TRY_EXPECTED_VOID(gpu::utils::read_data_sync(resultDevice, stagingBuffer, SIZE));

    std::vector<uint8_t> data(SIZE);
    TRY_EXPECTED_VOID(stagingBuffer.copyFrom(data.data(), SIZE));

    kernel::minmax::Result result{};
    std::memcpy(&result.min, data.data(), 4);
    std::memcpy(&result.max, data.data() + 4, 4);

    return result;
}

}  // namespace

namespace kernel::minmax {

auto run_reduction(std::span<float const> in,  //
                   utils::WorkGroupSize workgroupSizeX  //
                   ) noexcept -> std::expected<Result, std::string> {
    uint32_t const dataSizeBytes{static_cast<uint32_t>(in.size_bytes())};

    // create buffers
    DeviceData deviceData{};

    TRY_EXPECTED_VOID(deviceData.in.init(dataSizeBytes));
    TRY_EXPECTED_VOID(deviceData.result.init(8));

    // copy data
    TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(deviceData.in,  //
                                                   ::utils::to_byte_span(in)));

    // run compute
    TRY_EXPECTED_VOID(run_reduction(deviceData.in, deviceData.result, workgroupSizeX, dataSizeBytes));

    // read back
    return read_reduction(deviceData.result);
}

[[nodiscard]] auto run_reduction(gpu::DeviceBuffer const& in,  //
                                 gpu::DeviceBuffer const& result,  //
                                 utils::WorkGroupSize workgroupSizeX,  //
                                 std::optional<uint64_t> sizeBytesIn  //
                                 ) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    uint32_t const wgSizeX{static_cast<uint32_t>(workgroupSizeX)};

    auto const& limits{gpuManager.physicalDeviceProperties().properties.limits};
    if (wgSizeX > limits.maxComputeWorkGroupSize[0]) {
        return std::unexpected{fmt::format(
            "provided workgroupSizeX ({}) exceeds the maximum compute work group size ({}). Try to lower the size.",
            wgSizeX, limits.maxComputeWorkGroupSize[0])};
    }

    uint64_t dataSizeBytes{0};

    // input validation
    {
        if (sizeBytesIn) {
            dataSizeBytes = *sizeBytesIn;
        } else {
            dataSizeBytes = in.size();
        }

        if (dataSizeBytes < 4) {
            return std::unexpected("input buffer should have at least one element");
        }

        if ((dataSizeBytes % sizeof(float)) != 0) {
            return std::unexpected("input size should be multiple of sizeof(float)");
        }

        if (result.size() < 8) {
            return std::unexpected("result buffer should have space for at least 8 bytes");
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

    // get number of elements
    uint32_t n{static_cast<uint32_t>(dataSizeBytes / sizeof(float))};

    uint32_t constexpr WORKGROUP_SIZE_Y{1};
    uint32_t constexpr WORKGROUP_SIZE_Z{1};
    uint32_t constexpr GROUP_COUNT_Y{1};
    uint32_t constexpr GROUP_COUNT_Z{1};

    IntermediateDeviceData intermediateDeviceData{};

    // allocate buffers for intermediate results
    {
        uint32_t const initialGroupCountX{(n + wgSizeX - 1) / wgSizeX};

        // each workgroup produces a min and a max
        size_t const bufferSize{initialGroupCountX * 2 * sizeof(float)};

        TRY_EXPECTED_VOID(intermediateDeviceData.intermediateA.init(bufferSize));
        TRY_EXPECTED_VOID(intermediateDeviceData.intermediateB.init(bufferSize));
    }

    // using the same pipeline for all passes
    TRY_EXPECTED(auto const vkComputePipeline,
                 utils::get_pipeline(KERNEL_NAME,  //
                                     wgSizeX,  //
                                     WORKGROUP_SIZE_Y,  //
                                     WORKGROUP_SIZE_Z));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkComputePipeline);

    gpu::DeviceBuffer inputBuffer{in};
    gpu::DeviceBuffer outputBuffer{intermediateDeviceData.intermediateA};

    bool firstPass{true};

    while (true) {
        uint32_t const groupCountX{(n + wgSizeX - 1) / wgSizeX};

        // last pass - use the provided output buffer instead of intermediate
        if (groupCountX == 1) {
            outputBuffer = result;
        }

        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = inputBuffer.buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = n * sizeof(float);

            TRY_EXPECTED(uint32_t const descriptorIndexIn,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = outputBuffer.buffer();
            bufferInfo.range = outputBuffer.size();

            TRY_EXPECTED(uint32_t const descriptorIndexOut,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // see minmax_reduction.comp
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

        // compute
        vkCmdDispatch(commandBuffer, groupCountX, GROUP_COUNT_Y, GROUP_COUNT_Z);

        // barrier
        gpu::utils::set_buffer_barrier(commandBuffer,  //
                                       outputBuffer,  //
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
            outputBuffer = intermediateDeviceData.intermediateB;
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

auto read_reduction(gpu::DeviceBuffer const& resultDevice,  //
                    std::optional<gpu::HostVisibleBuffer> stagingBuffer  //
                    ) -> std::expected<Result, std::string> {
    uint32_t constexpr SIZE{8};

    if (stagingBuffer) {
        return read_impl(resultDevice, *stagingBuffer);
    }

    gpu::HostVisibleBuffer stagingBufferTmp{};
    TRY_EXPECTED_VOID(stagingBufferTmp.init(SIZE, true));

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] { stagingBufferTmp.destroy(); })};

    return read_impl(resultDevice, stagingBufferTmp);
}

}  // namespace kernel::minmax
