#include <fmt/core.h>

#include <cassert>
#include <gpu/CommandManager.hpp>
#include <utils/try_expected.hpp>
#include <vulkan/utility/vk_struct_helper.hpp>

namespace {

uint32_t constexpr NUM_BUFFERS_TO_ALLOCATE_AT_ONCE{10};

}

namespace gpu {

auto CommandManager::init(VkDevice device, uint32_t queueFamily) noexcept -> std::expected<void, std::string> {
    fmt::println("creating command manager");

    m_device = device;

    VkCommandPoolCreateInfo info = vku::InitStructHelper{};
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = queueFamily;

    if (vkCreateCommandPool(m_device, &info, nullptr, &m_commandPool) != VK_SUCCESS) {
        return std::unexpected{"failed to create command pool"};
    }

    m_availableCommandBuffers.reserve(NUM_BUFFERS_TO_ALLOCATE_AT_ONCE);

    return {};
}

auto CommandManager::destroy() noexcept -> void {
    if (!m_device) {
        return;
    }

    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    m_commandPool = VK_NULL_HANDLE;

    m_device = VK_NULL_HANDLE;

    m_availableCommandBuffers.clear();
}

auto CommandManager::commandBuffer() noexcept -> std::expected<VkCommandBuffer, std::string> {
    if (!m_device) {
        return std::unexpected{"CommandManager is not initialized"};
    }

    // if there are no available command buffers, allocate NUM_BUFFERS_TO_ALLOCATE_AT_ONCE and keep in a vector
    if (m_availableCommandBuffers.empty()) {
        fmt::println("allocating {} command buffers", NUM_BUFFERS_TO_ALLOCATE_AT_ONCE);

        VkCommandBufferAllocateInfo info = vku::InitStructHelper{};
        info.commandPool = m_commandPool;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = NUM_BUFFERS_TO_ALLOCATE_AT_ONCE;

        m_availableCommandBuffers.resize(NUM_BUFFERS_TO_ALLOCATE_AT_ONCE);
        if (vkAllocateCommandBuffers(m_device, &info, m_availableCommandBuffers.data()) != VK_SUCCESS) {
            return std::unexpected{"failed to allocate command buffers"};
        }
    }

    auto const commandBuffer{m_availableCommandBuffers.back()};
    m_availableCommandBuffers.pop_back();

    return commandBuffer;
}

auto CommandManager::commandBufferBegin() noexcept -> std::expected<VkCommandBuffer, std::string> {
    TRY_EXPECTED(VkCommandBuffer const commandBuffer, this->commandBuffer());

    VkCommandBufferBeginInfo info = vku::InitStructHelper{};
    info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    info.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(commandBuffer, &info) != VK_SUCCESS) {
        return std::unexpected{"failed to begin command buffer"};
    }

    // it's the caller's responsibility to end the command buffer
    // with vkEndCommandBuffer(commandBuffer);
    return commandBuffer;
}

auto CommandManager::resetCommandBuffer(VkCommandBuffer const commandBuffer) noexcept
    -> std::expected<void, std::string> {
    if (vkResetCommandBuffer(commandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT)) {
        return std::unexpected{"failed to reset command buffer"};
    }

    m_availableCommandBuffers.push_back(commandBuffer);

    return {};
}

}  // namespace gpu
