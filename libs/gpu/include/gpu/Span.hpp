#pragma once

#include <vulkan/vulkan.h>

#include <expected>
#include <string>

namespace gpu {

class DeviceBuffer;
class HostVisibleBuffer;

class Span {
public:
    Span() noexcept = default;
    Span(DeviceBuffer const& buffer) noexcept;
    Span(HostVisibleBuffer const& buffer) noexcept;

public:
    [[nodiscard]] auto static create(DeviceBuffer const& buffer,  //
                                     size_t offset,  //
                                     size_t size  //
                                     ) noexcept -> std::expected<Span, std::string>;

    [[nodiscard]] auto static create(HostVisibleBuffer const& buffer,  //
                                     size_t offset,  //
                                     size_t size  //
                                     ) noexcept -> std::expected<Span, std::string>;

private:
    Span(VkBuffer buffer, size_t offset, size_t size) noexcept;

public:
    auto offset() const noexcept -> size_t;
    auto size() const noexcept -> size_t;
    auto buffer() const noexcept -> VkBuffer;

private:
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_offset{0};
    uint64_t m_size{0};
};

}  // namespace gpu
