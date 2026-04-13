#include <fmt/core.h>

#include <chrono>
#include <functional>
#include <gpu/GpuManager.hpp>
#include <gpu/utils/get_push_constant_data.hpp>
#include <gpu/utils/init_helper.hpp>
#include <gpu/utils/read_helper.hpp>
#include <gpu/utils/submit.hpp>
#include <matrix_add/MatrixAdd.hpp>
#include <matrix_add/impl/create_matrix_add_pipeline.hpp>
#include <utils/to_span.hpp>
#include <utils/try_expected.hpp>

#ifdef VK_ENABLE_RENDERDOC_DEBUG
#include <renderdoc_app.h>

#ifdef __linux
#include <dlfcn.h>
#endif
#endif

namespace {

template <uint32_t RUNS = 10, uint32_t PERCENTILE_COUNT = 10>
[[nodiscard]] auto benchmark_with_percentiles(std::function<std::expected<void, std::string>()> const& fn)
    -> std::expected<void, std::string> {
    static_assert(RUNS > 0, "incorrect number of runs");
    static_assert(PERCENTILE_COUNT > 0, "incorrect number of percentiles");

    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::micro>;

    std::array<double, RUNS> samples{};

    // Run benchmark
    for (size_t i{0}; i < samples.size(); ++i) {
        auto const start{Clock::now()};
        TRY_EXPECTED_VOID(fn());
        auto const end{Clock::now()};

        samples[i] = std::chrono::duration_cast<Duration>(end - start).count();
    }

    // Sort samples
    std::sort(samples.begin(), samples.end());

    // Print results
    fmt::println("runs: {}", RUNS);

    for (size_t i{1}; i <= PERCENTILE_COUNT; ++i) {
        double const p{(100.0 * static_cast<double>(i)) / static_cast<double>(PERCENTILE_COUNT)};
        auto const index{static_cast<size_t>((p / 100.0) * static_cast<double>(RUNS - size_t{1}))};
        double const value{samples[index]};

        fmt::print("p{:5.1f}: {:.2f} us\n", p, value);
    }

    // Optional: min / max
    fmt::println("min: {:.2f} us", samples.front());
    fmt::println("max: {:.2f} us", samples.back());

    return {};
}

}  // namespace

namespace matrix_add {

auto MatrixAdd::destroy() noexcept -> void {
    if (!m_device) {
        return;
    }

    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_bufferA.destroy();
    m_bufferB.destroy();
    m_bufferOut.destroy();
    m_stagingBuffer.destroy();
}

auto MatrixAdd::run(gpu::GpuManager& gpuManager,  //
                    std::span<float const> a,  //
                    std::span<float const> b,  //
                    std::vector<float>& out  //
                    ) noexcept -> std::expected<void, std::string> {
    if (a.size() != b.size()) {
        return std::unexpected("input buffers size mismatch");
    }

#ifdef VK_ENABLE_RENDERDOC_DEBUG
    RENDERDOC_API_1_7_0* renderdocApi{nullptr};

    if (void* mod{dlopen("/home/nc/Downloads/renderdoc_1.43/lib/librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)}) {
        auto const getAPI{reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"))};
        [[maybe_unused]] int ret{getAPI(eRENDERDOC_API_Version_1_7_0, reinterpret_cast<void**>(&renderdocApi))};
    }

    if (renderdocApi) {
        renderdocApi->StartFrameCapture(nullptr, nullptr);
    }
#endif

    uint32_t constexpr WORKGROUP_SIZE_X{1024};
    uint32_t constexpr WORKGROUP_SIZE_Y{1};
    uint32_t constexpr WORKGROUP_SIZE_Z{1};

    if (!m_device) {
        m_device = gpuManager.device();
        TRY_EXPECTED(m_pipeline, impl::create_matrix_add_pipeline(m_device,  //
                                                                  gpuManager.pipelineLayout(),  //
                                                                  WORKGROUP_SIZE_X,  //
                                                                  WORKGROUP_SIZE_Y,  //
                                                                  WORKGROUP_SIZE_Z));
    }

    uint32_t const dataSize{static_cast<uint32_t>(a.size_bytes())};

    // create buffers
    {
        if (dataSize > m_bufferA.size()) {
            m_bufferA.destroy();
            TRY_EXPECTED_VOID(
                m_bufferA.init(gpuManager.allocator(),  //
                               VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,  //
                               dataSize));
        }

        if (dataSize > m_bufferB.size()) {
            m_bufferB.destroy();
            TRY_EXPECTED_VOID(
                m_bufferB.init(gpuManager.allocator(),  //
                               VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,  //
                               dataSize));
        }

        if (dataSize > m_bufferOut.size()) {
            m_bufferOut.destroy();
            TRY_EXPECTED_VOID(
                m_bufferOut.init(gpuManager.allocator(),  //
                                 VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,  //
                                 dataSize));
        }
    }

    // copy data
    {
        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(gpuManager,  //
                                                       m_bufferA,  //
                                                       utils::to_byte_span(a)));

        TRY_EXPECTED_VOID(gpu::utils::init_buffer_sync(gpuManager,  //
                                                       m_bufferB,  //
                                                       utils::to_byte_span(b)));
    }

    // run compute

    TRY_EXPECTED_VOID(benchmark_with_percentiles([&]() -> std::expected<void, std::string> {
        TRY_EXPECTED_VOID(runImpl(gpuManager, static_cast<uint32_t>(a.size()),  //
                                  dataSize,  //
                                  WORKGROUP_SIZE_X,  //
                                  WORKGROUP_SIZE_Y,  //
                                  WORKGROUP_SIZE_Z));
        return {};
    }));

    // read back
    {
        if (dataSize > m_stagingBuffer.size()) {
            m_stagingBuffer.destroy();
            TRY_EXPECTED_VOID(m_stagingBuffer.init(gpuManager.allocator(),  //
                                                   VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,  //
                                                   dataSize,  //
                                                   true));
        }

        TRY_EXPECTED_VOID(gpu::utils::read_data_sync(gpuManager, m_bufferOut, m_stagingBuffer, dataSize));

        if ((out.size() * sizeof(float)) < dataSize) {
            out.resize(dataSize / sizeof(float));
        }

        TRY_EXPECTED_VOID(m_stagingBuffer.copyFrom(out.data(), dataSize, 0));
    }

#ifdef VK_ENABLE_RENDERDOC_DEBUG
    if (renderdocApi) {
        renderdocApi->EndFrameCapture(nullptr, nullptr);
    }
#endif

    return {};
}

auto MatrixAdd::runImpl(gpu::GpuManager& gpuManager,  //
                        uint32_t dataCount,
                        uint32_t dataSizeBytes,  //
                        uint32_t workgroupSizeX,  //
                        uint32_t workgroupSizeY,  //
                        uint32_t workgroupSizeZ  //
                        ) -> std::expected<void, std::string> {
    uint32_t const workgroupSize{workgroupSizeX * workgroupSizeY * workgroupSizeZ};

    TRY_EXPECTED(auto const commandBuffer, gpuManager.commandManager().commandBufferBegin());

    // update descriptors
    {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = m_bufferA.buffer();
        bufferInfo.offset = 0;
        bufferInfo.range = dataSizeBytes;

        TRY_EXPECTED(uint32_t const descriptorIndexA,
                     gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

        bufferInfo.buffer = m_bufferB.buffer();

        TRY_EXPECTED(uint32_t const descriptorIndexB,
                     gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

        bufferInfo.buffer = m_bufferOut.buffer();

        TRY_EXPECTED(uint32_t const descriptorIndexOut,
                     gpuManager.storageDescriptorSetManager().push(commandBuffer, bufferInfo));

        // see matrix_add.cpmp
        auto const pushConstData{gpu::utils::get_push_constant_data(dataCount,  //
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

    uint32_t const numGroupsX{(dataCount + workgroupSize - 1) / workgroupSize};
    vkCmdDispatch(commandBuffer, numGroupsX, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return std::unexpected{"failed to end copy command buffer"};
    }

    TRY_EXPECTED_VOID(gpu::utils::submit(commandBuffer, gpuManager.computeQueue().queue));

    if (vkQueueWaitIdle(gpuManager.computeQueue().queue) != VK_SUCCESS) {
        return std::unexpected{"failed to wait queue"};
    }

    gpuManager.storageDescriptorSetManager().reset();
    TRY_EXPECTED_VOID(gpuManager.commandManager().resetCommandBuffer(commandBuffer));

    return {};
}

}  // namespace matrix_add
