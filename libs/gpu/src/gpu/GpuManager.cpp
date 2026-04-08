#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
//

#include <fmt/core.h>

#include <gpu/GpuManager.hpp>
#include <gpu/impl/RequiredApiVersion.hpp>
#include <gpu/impl/check_instance_version.hpp>
#include <gpu/impl/check_required_instance_extensions.hpp>
#include <gpu/impl/create_instance.hpp>
#include <gpu/impl/get_compute_queue_family.hpp>
#include <gpu/impl/get_physical_device_properties.hpp>
#include <gpu/impl/get_supported_physical_devices.hpp>
#include <utils/try_expected.hpp>

namespace gpu {

auto GpuManager::initialize() noexcept -> std::expected<void, std::string> {
    fmt::println("initializing gpu manager");
    fmt::println("minimum supported Vulkan version: {}.{}.0", impl::RequiredApiVersion::MAJOR,
                 impl::RequiredApiVersion::MINOR);

    TRY_EXPECTED_VOID(impl::check_instance_version());
    TRY_EXPECTED_VOID(impl::check_required_instance_extensions());
    TRY_EXPECTED(m_instance, impl::create_instance());
    TRY_EXPECTED(auto const supportedPhysicalDevices, impl::get_supported_physical_devices(m_instance));
    m_physicalDevice = supportedPhysicalDevices[0];
    m_physicalDeviceProperties = impl::get_physical_device_properties(m_physicalDevice);

    uint32_t constexpr REQUIRED_QUEUE_COUNT{1};
    TRY_EXPECTED(uint32_t const graphicsQueueFamily,
                 impl::get_compute_queue_family(m_physicalDevice, REQUIRED_QUEUE_COUNT));

    fmt::println("selected compute queue family: {}", graphicsQueueFamily);

    // TRY_EXPECTED(m_device, impl::create_device(m_physicalDevice, graphicsQueueFamily, REQUIRED_QUEUE_COUNT));

    // m_debugUtils.initialize(m_device);

    // uint32_t constexpr GRAPHICS_QUEUE_INDEX{0};
    // static_assert(GRAPHICS_QUEUE_INDEX < REQUIRED_QUEUE_COUNT);
    // m_graphicsQueue.queueFamily = graphicsQueueFamily;
    // m_graphicsQueue.queue = impl::get_graphics_queue(m_device, graphicsQueueFamily, GRAPHICS_QUEUE_INDEX);
    // m_debugUtils.setName(m_graphicsQueue.queue, fmt::format("graphics queue {}", GRAPHICS_QUEUE_INDEX));

    // TRY_EXPECTED(m_allocator, impl::create_allocator(m_instance, m_physicalDevice, m_device));

    // TRY_EXPECTED_VOID(m_commandManager.init(m_device, graphicsQueueFamily));

    // TRY_EXPECTED(m_storageDescriptorSetLayout,
    //              impl::create_descriptor_set_layout(m_device,  //
    //                                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  //
    //                                                 Constants::REQUIRED_STORAGE_DESCRIPTOR_COUNT));

    // m_debugUtils.setName(m_storageDescriptorSetLayout, "storage descriptor set layout");

    // TRY_EXPECTED(m_pipelineLayout,
    //              impl::create_pipeline_layout(m_device, m_storageDescriptorSetLayout, m_cisDescriptorSetLayout));

    // m_debugUtils.setName(m_pipelineLayout, "Pipeline layout.");

    // TRY_EXPECTED_VOID(m_storageDescriptorSetManager.init(m_device,  //
    //                                                      m_pipelineLayout,  //
    //                                                      m_storageDescriptorSetLayout,  //
    //                                                      Constants::REQUIRED_STORAGE_DESCRIPTOR_COUNT));

    // TRY_EXPECTED_VOID(m_immediateDataBufferManager.init(m_allocator, m_physicalDeviceProperties));

    // TRY_EXPECTED(m_fullscreenTrianglePipeline, impl::create_fullscreen_triangle_pipeline(
    //                                                m_device, m_pipelineLayout,
    //                                                m_surfaceFormat.surfaceFormat.format));
    // m_debugUtils.setName(m_fullscreenTrianglePipeline, "fullscreen triangle pipeline");

    return {};
}

// auto GraphicsManager::destroy() noexcept -> void {
//     fmt::println("Destroying.");

//     flush();

//     destroyDevice();

//     vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
//     m_surface = VK_NULL_HANDLE;

//     vkDestroyInstance(m_instance, nullptr);
//     m_instance = VK_NULL_HANDLE;
// }

// auto GraphicsManager::destroyDevice() noexcept -> void {
//     if (!m_device) {
//         return;
//     }

//     flush();

//     m_uiManager.destroy();

//     vkDestroySampler(m_device, m_fullscreenTriangleSampler, nullptr);
//     m_fullscreenTriangleSampler = VK_NULL_HANDLE;

//     vkDestroyPipeline(m_device, m_fullscreenTrianglePipeline, nullptr);
//     m_fullscreenTrianglePipeline = VK_NULL_HANDLE;

//     m_immediateDataBufferManager.destroy();

//     m_storageDescriptorSetManager.destroy();
//     m_cisDescriptorSetManager.destroy();

//     vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
//     m_pipelineLayout = VK_NULL_HANDLE;

//     vkDestroyDescriptorSetLayout(m_device, m_storageDescriptorSetLayout, nullptr);
//     m_storageDescriptorSetLayout = VK_NULL_HANDLE;

//     vkDestroyDescriptorSetLayout(m_device, m_cisDescriptorSetLayout, nullptr);
//     m_cisDescriptorSetLayout = VK_NULL_HANDLE;

//     m_commandManager.destroy();

//     for (auto const fence : m_fences) {
//         vkDestroyFence(m_device, fence, nullptr);
//     }
//     m_fences.clear();

//     for (auto const semaphore : m_renderingFinishedSemaphores) {
//         vkDestroySemaphore(m_device, semaphore, nullptr);
//     }
//     m_renderingFinishedSemaphores.clear();

//     for (auto const semaphore : m_imageAvailableSemaphores) {
//         vkDestroySemaphore(m_device, semaphore, nullptr);
//     }
//     m_imageAvailableSemaphores.clear();

//     for (auto& img : m_depthBuffers) {
//         img.destroy();
//     }
//     m_depthBuffers.clear();

//     for (auto& img : m_renderTargets) {
//         img.destroy();
//     }
//     m_renderTargets.clear();

//     for (auto view : m_swapchainImageViews) {
//         vkDestroyImageView(m_device, view, nullptr);
//     }
//     m_swapchainImageViews.clear();

//     vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
//     m_swapchain = VK_NULL_HANDLE;

//     vmaDestroyAllocator(m_allocator);
//     m_allocator = VK_NULL_HANDLE;

//     vkDestroyDevice(m_device, nullptr);
//     m_device = VK_NULL_HANDLE;
// }

// auto GraphicsManager::flush() const noexcept -> void {
//     fmt::println("Flushing.");

//     if (!m_device) {
//         return;
//     }

//     if (auto const r{vkDeviceWaitIdle(m_device)}; r != VK_SUCCESS) {
//         fmt::println("Failed to synchronize in GraphicsManager::flush(): {}", static_cast<int32_t>(r));
//     }
// }

// auto GraphicsManager::resize(window::Window& window) noexcept -> std::expected<void, std::string> {
//     fmt::println("resizing");

//     flush();

//     TRY_EXPECTED(VkSurfaceCapabilities2KHR const surfaceCapabilities,
//                  impl::get_surface_capabilities(m_physicalDevice, m_surface));
//     m_surfaceExtent = impl::get_surface_extent(surfaceCapabilities, window);

//     // destroy old swapchain image views
//     for (auto const view : m_swapchainImageViews) {
//         vkDestroyImageView(m_device, view, nullptr);
//     }

//     // create new swapchain
//     TRY_EXPECTED(m_swapchain, impl::create_swapchain(m_device,  //
//                                                      m_surface,  //
//                                                      m_surfaceFormat,  //
//                                                      m_presentMode,  //
//                                                      surfaceCapabilities,  //
//                                                      m_surfaceExtent,  //
//                                                      m_swapchain));

//     m_debugUtils.setName(m_swapchain, "swapchain");

//     TRY_EXPECTED(m_swapchainImages, impl::get_swapchain_images(m_device, m_swapchain));

//     for (size_t i{0}; i < m_swapchainImages.size(); ++i) {
//         m_debugUtils.setName(m_swapchainImages[i], fmt::format("swapchain image {}", i));
//     }

//     TRY_EXPECTED(m_swapchainImageViews,
//                  impl::create_swapchain_image_views(m_device, m_swapchainImages,
//                  m_surfaceFormat.surfaceFormat.format));

//     for (size_t i{0}; i < m_swapchainImageViews.size(); ++i) {
//         m_debugUtils.setName(m_swapchainImageViews[i], fmt::format("swapchain image view {}", i));
//     }

//     fmt::println("new window size: {}x{}", m_surfaceExtent.width, m_surfaceExtent.height);

//     // rebuild render targets
//     for (auto& img : m_renderTargets) {
//         img.destroy();
//     }
//     m_renderTargets.clear();

//     for (uint32_t i{0}; i < Constants::CONCURRENT_FRAME_COUNT; ++i) {
//         DeviceImage2d img{};
//         TRY_EXPECTED_VOID(img.init(m_allocator,  //
//                                    m_device,  //
//                                    m_renderTargetFormat,  //
//                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,  //
//                                    VK_IMAGE_ASPECT_COLOR_BIT,  //
//                                    m_surfaceExtent.width,  //
//                                    m_surfaceExtent.height));

//         m_debugUtils.setName(img.image(), fmt::format("render target image {}", i));
//         m_debugUtils.setName(img.view(), fmt::format("render target image view {}", i));

//         m_renderTargets.push_back(img);
//     }

//     // rebuild depth buffers
//     for (auto& img : m_depthBuffers) {
//         img.destroy();
//     }
//     m_depthBuffers.clear();

//     for (uint32_t i{0}; i < Constants::CONCURRENT_FRAME_COUNT; ++i) {
//         DeviceImage2d img{};
//         TRY_EXPECTED_VOID(img.init(m_allocator,  //
//                                    m_device,  //
//                                    m_depthFormat,  //
//                                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,  //
//                                    VK_IMAGE_ASPECT_DEPTH_BIT,  //
//                                    m_surfaceExtent.width,  //
//                                    m_surfaceExtent.height));

//         m_debugUtils.setName(img.image(), fmt::format("depth buffer image {}", i));
//         m_debugUtils.setName(img.view(), fmt::format("depth buffer image view {}", i));

//         m_depthBuffers.push_back(img);
//     }

//     return {};
// }

// auto GraphicsManager::startFrame(window::Window& window) noexcept -> std::expected<FrameData, std::string> {
//     m_frameIndex = (m_frameIndex + 1) % Constants::CONCURRENT_FRAME_COUNT;

//     // wait previous frame to finish
//     auto const imageAvailableSemaphore{m_imageAvailableSemaphores[m_frameIndex]};
//     auto const fence{m_fences[m_frameIndex]};

//     if (vkWaitForFences(m_device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()) != VK_SUCCESS) {
//         return std::unexpected{fmt::format("failed to wait fence {}", m_frameIndex)};
//     }

//     if (vkResetFences(m_device, 1, &fence) != VK_SUCCESS) {
//         return std::unexpected{fmt::format("failed to reset fence {}", m_frameIndex)};
//     }

//     auto [windowSize, resized]{window.size()};

//     // rebuild vulkan data if window was resized
//     if (resized) {
//         TRY_EXPECTED_VOID(resize(window));
//     }

//     // rebuild vulkan data if swapchain is outdated or suboptimal
//     uint32_t constexpr MAX_REBUILD_COUNT{10};
//     uint32_t counter{0};
//     impl::GetSwapchainImageIndexResult imageIndexResult{};
//     while (true) {
//         TRY_EXPECTED(imageIndexResult, impl::get_swapchain_image_index(m_device, m_swapchain,
//         imageAvailableSemaphore)); if (imageIndexResult.shouldRebuildSwapchain) {
//             TRY_EXPECTED_VOID(resize(window));

//             ++counter;

//             if (counter > 1) {
//                 fmt::println("rebuilding swapchain attempt {}", counter);
//             }

//             if (counter >= MAX_REBUILD_COUNT) {
//                 return std::unexpected{"failed to rebuild swapchain"};
//             }
//         } else {
//             break;
//         }
//     }

//     TRY_EXPECTED_VOID(m_commandManager.startFrame());

//     TRY_EXPECTED(VkCommandBuffer const commandBuffer, m_commandManager.commandBufferBegin());

//     TRY_EXPECTED_VOID(m_storageDescriptorSetManager.startFrame(commandBuffer));
//     TRY_EXPECTED_VOID(m_cisDescriptorSetManager.startFrame(commandBuffer));
//     m_immediateDataBufferManager.startFrame();

//     m_uiManager.startFrame();

//     // begin render target rendering
//     impl::begin_render_target_rendering(commandBuffer,  //
//                                         {0.8f, 0.2f, 0.2f, 1.0f},  //
//                                         m_renderTargets[m_frameIndex],  //
//                                         m_depthBuffers[m_frameIndex],  //
//                                         m_surfaceExtent);

//     FrameData frameData{commandBuffer, imageIndexResult.index};

//     return frameData;
// }

// auto GraphicsManager::endFrame(FrameData&& frameData) noexcept -> std::expected<void, std::string> {
//     auto const swapChainImage{m_swapchainImages[frameData.swapchainImageIndex]};
//     auto const swapChainImageView{m_swapchainImageViews[frameData.swapchainImageIndex]};
//     auto const imageAvailableSemaphore{m_imageAvailableSemaphores[m_frameIndex]};
//     // note the indexing here
//     auto const renderingFinishedSemaphore{m_renderingFinishedSemaphores[frameData.swapchainImageIndex]};
//     auto const fence{m_fences[m_frameIndex]};

//     // end render target rendering
//     vkCmdEndRendering(frameData.commandBuffer);

//     // final composition:
//     impl::begin_compose_rendering(frameData.commandBuffer,  //
//                                   m_renderTargets[m_frameIndex],  //
//                                   swapChainImage,  //
//                                   swapChainImageView,  //
//                                   m_surfaceExtent);

//     impl::set_fullscreen_triangle_rendering_state(frameData.commandBuffer,  //
//                                                   m_surfaceExtent,  //
//                                                   m_fullscreenTrianglePipeline);

//     TRY_EXPECTED_VOID(impl::draw_fullscreen_triangle(frameData.commandBuffer,  //
//                                                      m_fullscreenTriangleSampler,  //
//                                                      m_renderTargets[m_frameIndex].view(),  //
//                                                      m_pipelineLayout,  //
//                                                      m_cisDescriptorSetManager));

//     // draw ui
//     TRY_EXPECTED_VOID(m_uiManager.draw(frameData.commandBuffer, *this));

//     // end composition
//     vkCmdEndRendering(frameData.commandBuffer);

//     utils::set_image_barrier(frameData.commandBuffer,  //
//                              swapChainImage,  //
//                              VK_IMAGE_ASPECT_COLOR_BIT,  //
//                              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  //
//                              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,  //
//                              VK_PIPELINE_STAGE_2_NONE | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  //
//                              VK_ACCESS_2_NONE,  //
//                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

//     if (vkEndCommandBuffer(frameData.commandBuffer) != VK_SUCCESS) {
//         return std::unexpected{"failed to end command buffer"};
//     }

//     TRY_EXPECTED_VOID(impl::submit(frameData.commandBuffer,  //
//                                    m_graphicsQueue.queue,  //
//                                    imageAvailableSemaphore,  //
//                                    renderingFinishedSemaphore,  //
//                                    fence));

//     TRY_EXPECTED_VOID(impl::present(m_graphicsQueue.queue,  //
//                                     m_swapchain,  //
//                                     frameData.swapchainImageIndex,  //
//                                     renderingFinishedSemaphore));

//     return {};
// }

// auto GraphicsManager::allocator() const noexcept -> VmaAllocator {
//     return m_allocator;
// }

// auto GraphicsManager::combinedImageSamplerDescriptorSetManager() noexcept ->
// CombinedImageSamplerDescriptorSetManager& {
//     return m_cisDescriptorSetManager;
// }

// auto GraphicsManager::commandManager() noexcept -> CommandManager& {
//     return m_commandManager;
// }

// auto GraphicsManager::debugUtils() const noexcept -> VulkanDebugUtils const& {
//     return m_debugUtils;
// }

// auto GraphicsManager::depthFormat() const noexcept -> VkFormat {
//     return m_depthFormat;
// }

// auto GraphicsManager::device() const noexcept -> VkDevice {
//     return m_device;
// }

// auto GraphicsManager::graphicsQueue() const noexcept -> VulkanQueue {
//     return m_graphicsQueue;
// }

// auto GraphicsManager::immediateDataBufferManager() noexcept -> ImmediateDataBufferManager& {
//     return m_immediateDataBufferManager;
// }

// auto GraphicsManager::pipelineLayout() const noexcept -> VkPipelineLayout {
//     return m_pipelineLayout;
// }

// auto GraphicsManager::renderTargetFormat() const noexcept -> VkFormat {
//     return m_renderTargetFormat;
// }

// auto GraphicsManager::surfaceExtent() const noexcept -> VkExtent2D {
//     return m_surfaceExtent;
// }

// auto GraphicsManager::surfaceFormat() const noexcept -> VkFormat {
//     return m_surfaceFormat.surfaceFormat.format;
// }

// auto GraphicsManager::storageDescriptorSetManager() noexcept -> StorageDescriptorSetManager& {
//     return m_storageDescriptorSetManager;
// }

}  // namespace gpu
