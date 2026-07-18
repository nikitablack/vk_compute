#include <fmt/core.h>

#include <gpu/DeviceBuffer.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/Span.hpp>

namespace gpu {

Span::Span(DeviceBuffer const& buffer) noexcept : Span{buffer.buffer(), 0, buffer.size()} {}
Span::Span(HostVisibleBuffer const& buffer) noexcept : Span{buffer.buffer(), 0, buffer.size()} {}

auto Span::create(DeviceBuffer const& buffer,  //
                  size_t offset,  //
                  size_t size  //
                  ) noexcept -> std::expected<Span, std::string> {
    if (offset >= buffer.size()) {
        return std::unexpected{fmt::format("{}: beginning of span ({}) is out of buffer's range ({})",  //
                                           "Span::create",
                                           offset,  //
                                           buffer.size())};
    }

    if (buffer.size() < (offset + size)) {
        return std::unexpected{fmt::format("{}: end of span ({}) is out of buffer's range ({})", "Span::create",  //
                                           offset + size,  //
                                           buffer.size())};
    }

    Span s{buffer.buffer(), offset, size};

    return s;
}

auto Span::create(HostVisibleBuffer const& buffer,  //
                  size_t offset,  //
                  size_t size  //
                  ) noexcept -> std::expected<Span, std::string> {
    if (offset >= buffer.size()) {
        return std::unexpected{fmt::format("{}: beginning of span ({}) is out of buffer's range ({})",  //
                                           "Span::create",
                                           offset,  //
                                           buffer.size())};
    }

    if (buffer.size() < (offset + size)) {
        return std::unexpected{fmt::format("{}: end of span ({}) is out of buffer's range ({})", "Span::create",  //
                                           offset + size,  //
                                           buffer.size())};
    }

    Span s{buffer.buffer(), offset, size};

    return s;
}

Span::Span(VkBuffer buffer, size_t offset, size_t size) noexcept
    : m_buffer{buffer},  //
      m_offset{offset},  //
      m_size{size}  //
{}

auto Span::offset() const noexcept -> size_t {
    return m_offset;
}

auto Span::size() const noexcept -> size_t {
    return m_size;
}

auto Span::buffer() const noexcept -> VkBuffer {
    return m_buffer;
}

}  // namespace gpu
