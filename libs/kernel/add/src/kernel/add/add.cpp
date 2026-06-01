#include <spdlog/spdlog.h>

#include <gpu/GpuManager.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/add/add.hpp>
#include <kernel/utils/pipeline_helper.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace {

std::string constexpr KERNEL_NAME{"add"};

struct DeviceData {
    gpu::DeviceBuffer a{};
    gpu::DeviceBuffer b{};
    gpu::DeviceBuffer result{};

    ~DeviceData() {
        a.destroy();
        b.destroy();
        result.destroy();
    }
};

auto read_impl(std::span<float> resultHost,  //
               gpu::DeviceBuffer const& resultDevice,  //
               size_t bytesToRead,  //
               gpu::HostVisibleBuffer& stagingBuffer  //
               ) -> std::expected<void, std::string> {
    if (resultHost.size_bytes() < bytesToRead) {
        return std::unexpected(fmt::format("result buffer should have space for at least {} bytes", bytesToRead));
    }

    if (stagingBuffer.size() < bytesToRead) {
        return std::unexpected(fmt::format("staging buffer should have space for at least {} bytes", bytesToRead));
    }

    if ((bytesToRead % sizeof(float)) != 0) {
        return std::unexpected("the size of data to read should be multiple of sizeof(float)");
    }

    TRY_EXPECTED_VOID(gpu::utils::read_data_sync(resultDevice, stagingBuffer, bytesToRead));
    TRY_EXPECTED_VOID(stagingBuffer.copyFrom(resultHost.data(), bytesToRead));

    return {};
}

}  // namespace

namespace kernel::add {

auto run(std::span<float const> a,  //
         std::span<float const> b,  //
         std::span<float> result,  //
         uint32_t workgroupSizeX  //
         ) noexcept -> std::expected<void, std::string> {
    // special case
    if (a.size() == 0) {
        return {};
    }

    // input validation
    if ((a.size() != b.size()) || (a.size() != result.size())) {
        return std::unexpected("input data size mismatch");
    }

    uint32_t const dataSizeBytes{static_cast<uint32_t>(a.size_bytes())};

    // create buffers
    DeviceData deviceData{};

    TRY_EXPECTED_VOID(deviceData.a.init(dataSizeBytes));
    TRY_EXPECTED_VOID(deviceData.b.init(dataSizeBytes));
    TRY_EXPECTED_VOID(deviceData.result.init(dataSizeBytes));

    // copy data
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(deviceData.a,  //
                                                       ::utils::to_byte_span(a)));

        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(deviceData.b,  //
                                                       ::utils::to_byte_span(b)));
    }

    // run compute
    TRY_EXPECTED_VOID(run(deviceData.a, deviceData.b, deviceData.result, workgroupSizeX, dataSizeBytes));

    // readback
    TRY_EXPECTED_VOID(read(result, deviceData.result, dataSizeBytes));

    return {};
}

auto run(gpu::DeviceBuffer const& a,  //
         gpu::DeviceBuffer const& b,  //
         gpu::DeviceBuffer const& result,  //
         uint32_t workgroupSizeX,  //
         std::optional<uint64_t> sizeBytes  //
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
        if (sizeBytes) {
            dataSizeBytes = *sizeBytes;

            if ((a.size() < dataSizeBytes) || (b.size() < dataSizeBytes) || (result.size() < dataSizeBytes)) {
                return std::unexpected("input buffers are too small");
            }

        } else {
            dataSizeBytes = a.size();

            if ((dataSizeBytes != b.size()) || (dataSizeBytes != result.size())) {
                return std::unexpected("input buffers size mismatch");
            }
        }

        if ((dataSizeBytes % sizeof(float)) != 0) {
            return std::unexpected("input size should be multiple of sizeof(float)");
        }
    }

    if (dataSizeBytes == 0) {
        return {};
    }

    uint32_t const n{static_cast<uint32_t>(dataSizeBytes / sizeof(float))};

    uint32_t constexpr WORKGROUP_SIZE_Y{1};
    uint32_t constexpr WORKGROUP_SIZE_Z{1};

    TRY_EXPECTED(auto const vkPipeline,
                 utils::get_pipeline(KERNEL_NAME,  //
                                     workgroupSizeX,  //
                                     WORKGROUP_SIZE_Y,  //
                                     WORKGROUP_SIZE_Z));

    // compute
    {
        TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

        // RAII cleanup
        auto const guard{::utils::make_scope_guard([&] {
            gpuManager.storageDescriptorSetManager().reset();
            if (auto const r{gpuManager.commandManager().resetCommandBuffer(commandBuffer)}; !r) {
                spdlog::warn("{}", r.error());
            }
        })};

        // update descriptors
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = a.buffer();
            bufferInfo.offset = 0;
            bufferInfo.range = dataSizeBytes;

            TRY_EXPECTED(uint32_t const descriptorIndexA,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = b.buffer();

            TRY_EXPECTED(uint32_t const descriptorIndexB,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = result.buffer();

            TRY_EXPECTED(uint32_t const descriptorIndexOut,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // see add.comp
            auto const pushConstData{gpu::utils::get_push_constant_data(n,  //
                                                                        descriptorIndexA,  //
                                                                        descriptorIndexB,  //
                                                                        descriptorIndexOut)};

            vkCmdPushConstants(commandBuffer,  //
                               gpuManager.pipelineLayout(),  //
                               VK_SHADER_STAGE_COMPUTE_BIT,  //
                               0,  //
                               static_cast<uint32_t>(pushConstData.size()),  //
                               pushConstData.data());
        }

        // dispatch and wait
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vkPipeline);

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

auto read(std::span<float> resultHost,  //
          gpu::DeviceBuffer const& resultDevice,  //
          std::optional<size_t> bytesToRead,  //
          std::optional<gpu::HostVisibleBuffer> stagingBuffer  //
          ) -> std::expected<void, std::string> {
    size_t const size{bytesToRead ? *bytesToRead : resultDevice.size()};

    if (stagingBuffer) {
        return read_impl(resultHost, resultDevice, size, *stagingBuffer);
    }

    gpu::HostVisibleBuffer stagingBufferTmp{};
    TRY_EXPECTED_VOID(stagingBufferTmp.init(size, true));

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] { stagingBufferTmp.destroy(); })};

    return read_impl(resultHost, resultDevice, size, stagingBufferTmp);
}

}  // namespace kernel::add
