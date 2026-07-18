#include <spdlog/spdlog.h>

#include <gpu/GpuManager.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/submit.hpp>
#include <kernel/Add.hpp>
#include <kernel/utils/pipeline_helper.hpp>
#include <kernel/utils/workgroup_helper.hpp>
#include <utils/ScopeGuard.hpp>
#include <utils/try_expected.hpp>

namespace {

std::string constexpr KERNEL_NAME{"add"};

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

auto Add::operator()(std::span<float const> srcA,  //
                     std::span<float const> srcB,  //
                     std::span<float> dst  //
) const noexcept -> std::expected<void, std::string> {
    // input validation
    if ((srcA.size() != srcB.size()) || (srcA.size() != dst.size())) {
        return std::unexpected{"input data size mismatch"};
    }

    // special case
    if (srcA.size() == 0) {
        return {};
    }

    auto const sizeBytes{srcA.size_bytes()};

    // create single buffer
    // since all sizes (for `srcA, `srcB` and `dst`) must match, use the 3xsize for the buffer
    // use ofsets to point to different memory inside the buffer
    // TRY_EXPECTED(auto inOutBuffer, gpu::DeviceBuffer::create(sizeBytes * 3));
    TRY_EXPECTED(auto inOutBuffer, gpu::Buffer::create(sizeBytes * 3, gpu::Buffer::Type::Device));

    // RAII cleanup
    auto const guard{::utils::make_scope_guard([&] { inOutBuffer.destroy(); })};

    // copy data
    {
        TRY_EXPECTED_VOID(inOutBuffer.copyToBuffer(std::as_bytes(srcA), 0));
        TRY_EXPECTED_VOID(inOutBuffer.copyToBuffer(std::as_bytes(srcB), sizeBytes));
    }

    TRY_EXPECTED(auto const spanSrcA, gpu::BufferSpan::create(inOutBuffer, 0, sizeBytes));
    TRY_EXPECTED(auto const spanSrcB, gpu::BufferSpan::create(inOutBuffer, sizeBytes, sizeBytes));
    TRY_EXPECTED(auto const spanDst, gpu::BufferSpan::create(inOutBuffer, 2 * sizeBytes, sizeBytes));

    // run compute
    {
        auto const& add{*this};
        TRY_EXPECTED_VOID(add(spanSrcA,  //
                              spanSrcB,  //
                              spanDst));
    }

    // readback
    {
        TRY_EXPECTED_VOID(read(dst, spanDst));
    }

    return {};
}

auto Add::operator()(gpu::BufferSpan const& srcA,  //
                     gpu::BufferSpan const& srcB,  //
                     gpu::BufferSpan const& dst  //
) const noexcept -> std::expected<void, std::string> {
    using T = float;

    // input validation
    {
        // check if all sizes equal
        if ((srcA.size() != srcB.size()) || (srcA.size() != dst.size())) {
            return std::unexpected{"input data size mismatch"};
        }

        // special case
        if (srcA.size() == 0) {
            return {};
        }

        // check if ranges overlap
        auto const rangeOverlaps{[&](gpu::BufferSpan const& src) {
            if (src.buffer().buffer() != dst.buffer().buffer()) {
                return false;
            }

            if (src.offset() <= dst.offset()) {
                return (dst.offset() - src.offset()) < src.size();
            }

            return src.offset() - dst.offset() < dst.size();
        }};

        if (rangeOverlaps(srcA) || rangeOverlaps(srcB)) {
            return std::unexpected{"destination memory overlaps with source memory"};
        }

        if ((srcA.size() % sizeof(T)) != 0) {
            return std::unexpected{fmt::format("input size should be multiple of {}", sizeof(T))};
        }
    }

    TRY_EXPECTED_REF(auto& gpuManager, gpu::GpuManager::get());

    size_t const sizeBytes{srcA.size()};

    // number of elements as `size_t`. Since a workgroup size and workgroup count have `uint32_t` type, it's
    // possible that the amount of work is bigger than a GPU can handle when calculating one addition per frame.
    //
    // the possible solutions are:
    //     - automatically split input and dispatch multiple times
    //     - make one thread to calculate more than a single add
    auto const nn{sizeBytes / sizeof(T)};

    // for now return an error in case the input is too big
    if (nn > std::numeric_limits<uint32_t>::max()) {
        return std::unexpected{fmt::format("input size is to big, try to split it")};
    }

    auto const n{static_cast<uint32_t>(nn)};

    // TODO: it's possible that the required number of workgroups is bigger than GPU is capable of; in this
    // case, use multiple dispatches
    auto const groupCountX{utils::get_workgroupgroup_count(n, m_workgroupSizeX)};
    uint32_t constexpr GROUP_COUNT_Y{1};
    uint32_t constexpr GROUP_COUNT_Z{1};

    // for now return an error in case the required workroup count is too big
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
            bufferInfo.range = sizeBytes;

            // srcA
            bufferInfo.buffer = srcA.buffer().buffer();
            bufferInfo.offset = srcA.offset();

            TRY_EXPECTED(uint32_t const descriptorIndexA,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // srcB
            bufferInfo.buffer = srcB.buffer().buffer();
            bufferInfo.offset = srcB.offset();

            TRY_EXPECTED(uint32_t const descriptorIndexB,
                         gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

            // dst
            bufferInfo.buffer = dst.buffer().buffer();
            bufferInfo.offset = dst.offset();

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

auto Add::read(std::span<float> dst,  //
               gpu::BufferSpan const& src,  //
               std::optional<gpu::Buffer> stagingBuffer  //
) const noexcept -> std::expected<void, std::string> {
    using T = float;

    size_t const sizeBytes{src.size()};

    // TODO: it is possible that the device `src` buffer is too big and there's no more GPU memory left to
    // allocate a staging buffer of the same size; in this case try to allocate a smaller staging buffer and use
    // it multiple times to copy parts of data. For now let's hope the data is not big enough.

    gpu::Buffer stagingBufferTmp{};

    // RAII cleanup, will do nothing if the `stagingBufferTmp` was not initialized (when `stagingBuffer` was
    // provided as parameter)
    auto const guard{::utils::make_scope_guard([&] { stagingBufferTmp.destroy(); })};

    // holds pointer to either stagingBuffer or stagingBufferTmp
    gpu::Buffer* readbackBuffer{nullptr};

    // if `stagingBuffer` was provided - use it. If not - use the temporary `stagingBufferTmp`. This condition
    // is necessary to make the `guard` RAII wrapper work.
    if (stagingBuffer) {
        if (stagingBuffer->type() != gpu::Buffer::Type::Readback) {
            return std::unexpected{"wrong staging buffer type"};
        }

        readbackBuffer = &stagingBuffer.value();
    } else {
        TRY_EXPECTED(auto result, gpu::Buffer::create(sizeBytes, gpu::Buffer::Type::Readback));
        stagingBufferTmp = std::move(result);

        readbackBuffer = &stagingBufferTmp;
    }

    if (dst.size_bytes() < sizeBytes) {
        return std::unexpected{"destination buffer is too small"};
    }

    if (readbackBuffer->size() < sizeBytes) {
        return std::unexpected{"staging buffer is too small"};
    }

    if ((sizeBytes % sizeof(T)) != 0) {
        return std::unexpected{fmt::format("the size of data to read should be multiple of {}", sizeof(T))};
    }

    TRY_EXPECTED_VOID(readbackBuffer->copyToBuffer(src, 0));

    TRY_EXPECTED_VOID(readbackBuffer->copyFromBuffer(dst));

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
