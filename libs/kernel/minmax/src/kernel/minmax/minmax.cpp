#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <gpu/GpuManager.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/utils/barrier_helper.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/minmax/minmax.hpp>
#include <kernel/utils/pipeline_helper.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace {

std::string const INIT_KERNEL_NAME{"minmax_uint_init"};
std::string const COMPUTE_KERNEL_NAME{"minmax_uint_atomic"};

struct DeviceData {
    gpu::DeviceBuffer in{};
    gpu::DeviceBuffer result{};
    gpu::HostVisibleBuffer staging{};

    ~DeviceData() {
        in.destroy();
        result.destroy();
        staging.destroy();
    }
};

auto ordered_uint_to_float(uint32_t u) noexcept -> float {
    uint32_t const v{(u & 0x80000000u) != 0u ? (u ^ 0x80000000u) : ~u};
    return std::bit_cast<float>(v);
}

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

    uint32_t minUint{};
    std::memcpy(&minUint, data.data(), 4);

    uint32_t maxUint{};
    std::memcpy(&maxUint, data.data() + 4, 4);

    kernel::minmax::Result result{};
    result.min = ordered_uint_to_float(minUint);
    result.max = ordered_uint_to_float(maxUint);

    return result;
}

}  // namespace

namespace kernel::minmax {

auto run(std::span<float const> in,  //
         uint32_t workgroupSizeX  //
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
    TRY_EXPECTED_VOID(run(deviceData.in, deviceData.result, workgroupSizeX, dataSizeBytes));

    // read back
    return read(deviceData.result);
}

[[nodiscard]] auto run(gpu::DeviceBuffer const& in,  //
                       gpu::DeviceBuffer const& result,  //
                       uint32_t workgroupSizeX,  //
                       std::optional<uint64_t> sizeBytesIn  //
                       ) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    auto const& limits{gpuManager.physicalDeviceProperties().properties.limits};
    if (workgroupSizeX > limits.maxComputeWorkGroupSize[0]) {
        return std::unexpected{
            fmt::format("provided workgroupSizeX ({}) exceeds the maximum compute work group size ({})", workgroupSizeX,
                        limits.maxComputeWorkGroupSize[0])};
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

    // init the output
    {
        uint32_t constexpr WORKGROUP_SIZE_X{2};
        uint32_t constexpr WORKGROUP_SIZE_Y{1};
        uint32_t constexpr WORKGROUP_SIZE_Z{1};

        TRY_EXPECTED(auto const vkInitPipeline,
                     utils::get_pipeline(INIT_KERNEL_NAME,  //
                                         WORKGROUP_SIZE_X,  //
                                         WORKGROUP_SIZE_Y,  //
                                         WORKGROUP_SIZE_Z));

        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = result.buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = 8;

            TRY_EXPECTED(uint32_t const descriptorIndexIn,
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
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkInitPipeline);

        uint32_t constexpr GROUP_COUNT_X{1};
        uint32_t constexpr GROUP_COUNT_Y{1};
        uint32_t constexpr GROUP_COUNT_Z{1};

        vkCmdDispatch(commandBuffer, GROUP_COUNT_X, GROUP_COUNT_Y, GROUP_COUNT_Z);
    }

    // barrier
    gpu::utils::set_buffer_barrier(commandBuffer,  //
                                   result,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_WRITE_BIT,  //
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,  //
                                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // compute
    {
        // get number of elements
        uint32_t const n{static_cast<uint32_t>(dataSizeBytes / sizeof(float))};

        uint32_t constexpr WORKGROUP_SIZE_Y{1};
        uint32_t constexpr WORKGROUP_SIZE_Z{1};

        TRY_EXPECTED(auto const vkComputePipeline,
                     utils::get_pipeline(COMPUTE_KERNEL_NAME,  //
                                         workgroupSizeX,  //
                                         WORKGROUP_SIZE_Y,  //
                                         WORKGROUP_SIZE_Z));

        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = in.buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = dataSizeBytes;

            TRY_EXPECTED(uint32_t const descriptorIndexIn,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = result.buffer();
            bufferInfo.range = 8;

            TRY_EXPECTED(uint32_t const descriptorIndexOut,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // see minmax_uint_atomic.comp
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
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkComputePipeline);

        uint32_t const groupCountX{(n + workgroupSizeX - 1) / workgroupSizeX};
        uint32_t constexpr GROUP_COUNT_Y{1};
        uint32_t constexpr GROUP_COUNT_Z{1};

        vkCmdDispatch(commandBuffer, groupCountX, GROUP_COUNT_Y, GROUP_COUNT_Z);

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

auto read(gpu::DeviceBuffer const& resultDevice,  //
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
