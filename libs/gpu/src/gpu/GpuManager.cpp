#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
//

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include <gpu/GpuManager.hpp>
#include <gpu/impl/RequiredApiVersion.hpp>
#include <gpu/impl/check_instance_version.hpp>
#include <gpu/impl/check_required_instance_extensions.hpp>
#include <gpu/impl/create_allocator.hpp>
#include <gpu/impl/create_descriptor_set_layout.hpp>
#include <gpu/impl/create_device.hpp>
#include <gpu/impl/create_instance.hpp>
#include <gpu/impl/create_pipeline_layout.hpp>
#include <gpu/impl/get_compute_queue_family.hpp>
#include <gpu/impl/get_physical_device_properties.hpp>
#include <gpu/impl/get_queue.hpp>
#include <gpu/impl/get_supported_physical_devices.hpp>
#include <utils/try_expected.hpp>

namespace gpu {

bool GpuManager::m_initialized{false};

auto GpuManager::get() noexcept -> GpuManager& {
    static GpuManager gpuManager{};

    return gpuManager;
}

auto GpuManager::init() noexcept -> std::expected<void, std::string> {
    GpuManager& gpuManager{GpuManager::get()};

    if (!GpuManager::m_initialized) {
        spdlog::cfg::load_env_levels();

        TRY_EXPECTED_VOID(gpuManager.initialize());
        GpuManager::m_initialized = true;
    }

    return {};
}

auto GpuManager::initialized() noexcept -> bool {
    return GpuManager::m_initialized;
}

auto GpuManager::destroy() noexcept -> void {
    if (!GpuManager::m_initialized) {
        return;
    }

    GpuManager& gpuManager{GpuManager::get()};

    gpuManager.destroyImpl();

    GpuManager::m_initialized = false;
}

auto GpuManager::initialize() noexcept -> std::expected<void, std::string> {
    spdlog::trace("initializing gpu manager");
    spdlog::info("minimum supported Vulkan version: {}.{}.0", impl::RequiredApiVersion::MAJOR,
                 impl::RequiredApiVersion::MINOR);

    TRY_EXPECTED_VOID(impl::check_instance_version());
    TRY_EXPECTED_VOID(impl::check_required_instance_extensions());
    TRY_EXPECTED(m_instance, impl::create_instance());
    TRY_EXPECTED_VOID(VulkanFunctions::initialize(m_instance));
    TRY_EXPECTED(auto const supportedPhysicalDevices, impl::get_supported_physical_devices(m_instance));
    m_physicalDevice = supportedPhysicalDevices[0];
    m_physicalDeviceProperties = impl::get_physical_device_properties(m_physicalDevice);

    uint32_t constexpr REQUIRED_QUEUE_COUNT{1};
    TRY_EXPECTED(uint32_t const computeQueueFamily,
                 impl::get_compute_queue_family(m_physicalDevice, REQUIRED_QUEUE_COUNT));

    spdlog::info("selected compute queue family: {}", computeQueueFamily);

    TRY_EXPECTED(m_device, impl::create_device(m_physicalDevice, computeQueueFamily, REQUIRED_QUEUE_COUNT));
    m_debugUtils.initialize(m_device);

    uint32_t constexpr GRAPHICS_QUEUE_INDEX{0};
    static_assert(GRAPHICS_QUEUE_INDEX < REQUIRED_QUEUE_COUNT);
    m_computeQueue.queueFamily = computeQueueFamily;
    m_computeQueue.queue = impl::get_queue(m_device, computeQueueFamily, GRAPHICS_QUEUE_INDEX);

    m_debugUtils.setName(m_computeQueue.queue,
                         fmt::format("compute queue {}.{}", computeQueueFamily, GRAPHICS_QUEUE_INDEX));

    TRY_EXPECTED(m_allocator, impl::create_allocator(m_instance, m_physicalDevice, m_device));
    TRY_EXPECTED_VOID(m_commandManager.init(m_device, computeQueueFamily));

    uint32_t constexpr REQUIRED_STORAGE_DESCRIPTOR_COUNT{100};
    TRY_EXPECTED(m_storageDescriptorSetLayout,
                 impl::create_descriptor_set_layout(m_device,  //
                                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  //
                                                    REQUIRED_STORAGE_DESCRIPTOR_COUNT));

    m_debugUtils.setName(m_storageDescriptorSetLayout, "storage descriptor set layout");

    TRY_EXPECTED(m_pipelineLayout, impl::create_pipeline_layout(m_device, m_storageDescriptorSetLayout));
    m_debugUtils.setName(m_pipelineLayout, "Pipeline layout.");

    TRY_EXPECTED_VOID(m_storageDescriptorSetManager.init(m_device,  //
                                                         m_pipelineLayout,  //
                                                         m_storageDescriptorSetLayout,  //
                                                         REQUIRED_STORAGE_DESCRIPTOR_COUNT));

    return {};
}

auto GpuManager::destroyImpl() noexcept -> void {
    if (!m_device) {
        return;
    }

    spdlog::trace("destroying");

    flush();

    for (auto const& p : m_dataToPipeline) {
        vkDestroyPipeline(m_device, p.second, nullptr);
    }
    m_dataToPipeline.clear();

    m_storageDescriptorSetManager.destroy();

    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_pipelineLayout = VK_NULL_HANDLE;

    vkDestroyDescriptorSetLayout(m_device, m_storageDescriptorSetLayout, nullptr);
    m_storageDescriptorSetLayout = VK_NULL_HANDLE;

    m_commandManager.destroy();

    vmaDestroyAllocator(m_allocator);
    m_allocator = VK_NULL_HANDLE;

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;

    vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
}

auto GpuManager::addPipeline(VkPipeline pipeline,  //
                             std::string const& name,  //
                             uint32_t workgroupSizeX,  //
                             uint32_t workgroupSizeY,  //
                             uint32_t workgroupSizeZ  //
                             ) noexcept -> void {
    PipelineData pipelineData{};
    pipelineData.name = name;
    pipelineData.workgroupSizeX = workgroupSizeX;
    pipelineData.workgroupSizeY = workgroupSizeY;
    pipelineData.workgroupSizeZ = workgroupSizeZ;

    VkPipeline existingPipeline{m_dataToPipeline[pipelineData]};
    if (existingPipeline) {
        vkDestroyPipeline(m_device, existingPipeline, nullptr);
    }

    m_dataToPipeline[pipelineData] = pipeline;
}

auto GpuManager::getPipeline(std::string const& name,  //
                             uint32_t workgroupSizeX,  //
                             uint32_t workgroupSizeY,  //
                             uint32_t workgroupSizeZ  //
                             ) noexcept -> std::optional<VkPipeline> {
    PipelineData pipelineData{};
    pipelineData.name = name;
    pipelineData.workgroupSizeX = workgroupSizeX;
    pipelineData.workgroupSizeY = workgroupSizeY;
    pipelineData.workgroupSizeZ = workgroupSizeZ;

    VkPipeline pipeline{m_dataToPipeline[pipelineData]};
    if (pipeline) {
        return pipeline;
    }

    return std::nullopt;
}

auto GpuManager::flush() const noexcept -> void {
    spdlog::trace("flushing");

    if (!m_device) {
        return;
    }

    if (auto const r{vkDeviceWaitIdle(m_device)}; r != VK_SUCCESS) {
        spdlog::warn("failed to synchronize in flush(): {}", static_cast<int32_t>(r));
    }
}

auto GpuManager::allocator() const noexcept -> VmaAllocator {
    return m_allocator;
}

auto GpuManager::commandManager() noexcept -> CommandManager& {
    return m_commandManager;
}

auto GpuManager::computeQueue() const noexcept -> VulkanQueue {
    return m_computeQueue;
}

// auto GraphicsManager::debugUtils() const noexcept -> VulkanDebugUtils const& {
//     return m_debugUtils;
// }

auto GpuManager::device() const noexcept -> VkDevice {
    return m_device;
}

auto GpuManager::pipelineLayout() const noexcept -> VkPipelineLayout {
    return m_pipelineLayout;
}

auto GpuManager::physicalDeviceProperties() const noexcept -> VkPhysicalDeviceProperties2 const& {
    return m_physicalDeviceProperties;
}

auto GpuManager::storageDescriptorSetManager() noexcept -> StorageDescriptorSetManager& {
    return m_storageDescriptorSetManager;
}

}  // namespace gpu
