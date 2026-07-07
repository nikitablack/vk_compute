#include <spdlog/spdlog.h>

#include <gpu/DeviceBuffer.hpp>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/Add.hpp>
#include <kernel/utils/pipeline_helper.hpp>
#include <kernel/utils/workgroup_helper.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

namespace {

std::string constexpr KERNEL_NAME{"add"};

struct DeviceData {
    gpu::DeviceBuffer a{};
    gpu::DeviceBuffer b{};
    gpu::DeviceBuffer c{};

    ~DeviceData() {
        a.destroy();
        b.destroy();
        c.destroy();
    }
};

}  // namespace

namespace kernel {

Add::Add(Add&& other) noexcept {
    m_pipeline = other.m_pipeline;
    m_workgroupSizeX = other.m_workgroupSizeX;

    other.m_workgroupSizeX = 0;
    other.m_pipeline = VK_NULL_HANDLE;
}

auto Add::init(uint32_t workgroupSizeX) noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    // check that requested workgroup size works for the selected device
    {
        if (workgroupSizeX == 0) {
            return std::unexpected{"workgroup size can't be 0"};
        }

        auto const& limits{gpuManager.physicalDeviceProperties().properties.limits};

        if (workgroupSizeX > limits.maxComputeWorkGroupSize[0]) {
            return std::unexpected{
                fmt::format("provided workgroup size x ({}) exceeds the maximum compute workgroup size ({})",
                            workgroupSizeX, limits.maxComputeWorkGroupSize[0])};
        }
    }

    m_workgroupSizeX = workgroupSizeX;

    // create pipeline
    {
        TRY_EXPECTED(m_pipeline, utils::create_pipeline(KERNEL_NAME,  //
                                                        m_workgroupSizeX,  //
                                                        1,  //
                                                        1));
    }

    return {};
}

auto Add::create(uint32_t workgroupSizeX) noexcept -> std::expected<Add, std::string> {
    Add add{};

    TRY_EXPECTED_VOID(add.init(workgroupSizeX));

    return add;
}

auto Add::operator()(std::span<float const> a,  //
                     std::span<float const> b,  //
                     std::span<float> c  //
) const noexcept -> std::expected<void, std::string> {
    // input validation
    if ((a.size() != b.size()) || (a.size() != c.size())) {
        return std::unexpected("input data size mismatch");
    }

    // special case
    if (a.size() == 0) {
        return {};
    }

    auto const sizeBytes{a.size_bytes()};

    // create buffers
    DeviceData deviceData{};

    TRY_EXPECTED_VOID(deviceData.a.init(sizeBytes));
    TRY_EXPECTED_VOID(deviceData.b.init(sizeBytes));
    TRY_EXPECTED_VOID(deviceData.c.init(sizeBytes));

    // copy data
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(deviceData.a,  //
                                                       ::utils::to_byte_span(a)));

        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(deviceData.b,  //
                                                       ::utils::to_byte_span(b)));
    }

    // run compute
    TRY_EXPECTED_VOID((*this)(deviceData.a, deviceData.b, deviceData.c, sizeBytes));

    // readback
    TRY_EXPECTED_VOID(read(c, deviceData.c, sizeBytes));

    return {};
}

// TODO: add span for device buffers
auto Add::operator()(gpu::DeviceBuffer const& a,  //
                     gpu::DeviceBuffer const& b,  //
                     gpu::DeviceBuffer const& c,  //
                     size_t sizeBytes  //
) const noexcept -> std::expected<void, std::string> {
    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    // input validation
    {
        if (sizeBytes == 0) {
            return {};
        }

        if ((a.size() < sizeBytes) || (b.size() < sizeBytes) || (c.size() < sizeBytes)) {
            return std::unexpected("input buffers are too small");
        }

        if ((sizeBytes % sizeof(float)) != 0) {
            return std::unexpected(fmt::format("input size should be multiple of {}", sizeof(float)));
        }
    }

    auto const nn{sizeBytes / sizeof(float)};

    if (nn > std::numeric_limits<uint32_t>::max()) {
        // TODO: maybe split the workload on multiple parts?
        return std::unexpected(fmt::format("input size is to big, try to reduce it"));
    }

    auto const n{static_cast<uint32_t>(nn)};

    auto const groupCountX{utils::get_workgroupgroup_count(n, m_workgroupSizeX)};
    uint32_t constexpr GROUP_COUNT_Y{1};
    uint32_t constexpr GROUP_COUNT_Z{1};

    // check that the requested number of workgroups does not exceed the allowed maximum
    {
        auto const& limits{gpuManager.physicalDeviceProperties().properties.limits};

        if (groupCountX > limits.maxComputeWorkGroupCount[0]) {
            return std::unexpected{
                fmt::format("the calculated count of workgroups ({}) exceeds the maximum compute workgroup count ({})",
                            groupCountX, limits.maxComputeWorkGroupCount[0])};
        }
    }

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
            bufferInfo.range = sizeBytes;

            TRY_EXPECTED(uint32_t const descriptorIndexA,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = b.buffer();

            TRY_EXPECTED(uint32_t const descriptorIndexB,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            bufferInfo.buffer = c.buffer();

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
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

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

auto Add::read(std::span<float> cHost,  //
               gpu::DeviceBuffer const& cDevice,  //
               size_t sizeBytes,  //
               std::optional<gpu::HostVisibleBuffer> stagingBuffer  //
) const noexcept -> std::expected<void, std::string> {
    gpu::HostVisibleBuffer stagingBufferTmp{};

    // RAII cleanup, will do nothing if the temporary buffer was not initialized
    auto const guard{::utils::make_scope_guard([&] { stagingBufferTmp.destroy(); })};

    gpu::HostVisibleBuffer readbackBuffer{};

    if (stagingBuffer) {
        readbackBuffer = *stagingBuffer;
    } else {
        TRY_EXPECTED_VOID(stagingBufferTmp.init(sizeBytes, true));

        readbackBuffer = stagingBufferTmp;
    }

    if (cHost.size_bytes() < sizeBytes) {
        return std::unexpected(fmt::format("result host buffer should have space for at least {} bytes", sizeBytes));
    }

    if (readbackBuffer.size() < sizeBytes) {
        return std::unexpected(fmt::format("staging buffer should have space for at least {} bytes", sizeBytes));
    }

    if ((sizeBytes % sizeof(float)) != 0) {
        return std::unexpected(fmt::format("the size of data to read should be multiple of {}", sizeof(float)));
    }

    TRY_EXPECTED_VOID(gpu::utils::read_data_sync(cDevice, readbackBuffer, sizeBytes));
    TRY_EXPECTED_VOID(readbackBuffer.copyFrom(cHost.data(), sizeBytes));

    return {};
}

auto Add::destroy() noexcept -> void {
    auto gpuManagerResult{gpu::GpuManager::get()};

    if (!gpuManagerResult.has_value()) {
        spdlog::warn("failed to get gpu manager");
        return;
    }

    auto& gpuManager{gpuManagerResult.value().get()};

    vkDestroyPipeline(gpuManager.device(), m_pipeline, nullptr);
    m_pipeline = VK_NULL_HANDLE;

    m_workgroupSizeX = 0;
}

}  // namespace kernel
