#pragma once

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <expected>
#include <span>
#include <string>
#include <utils/try_expected.hpp>

namespace gpu {

class BufferSpan;

class Buffer {
public:
    enum class Type : uint32_t { None, Device, Upload, Readback };

public:
    Buffer() = default;

    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;

    Buffer(Buffer&& other) noexcept;
    auto operator=(Buffer&& other) noexcept -> Buffer&;

public:
    [[nodiscard]] auto static create(size_t sizeBytes,  //
                                     Type type  //
                                     ) noexcept -> std::expected<Buffer, std::string>;

    template <typename T>
    [[nodiscard]] auto static create(std::span<T const> initialData,  //
                                     Type type  //
                                     ) noexcept -> std::expected<Buffer, std::string> {
        TRY_EXPECTED(auto buffer, create(initialData.size_bytes(), type));

        TRY_EXPECTED_VOID(buffer.copyToBuffer(std::as_bytes(initialData), 0));

        return buffer;
    }

    [[nodiscard]] auto copyToBuffer(gpu::BufferSpan const& src,  //
                                    size_t dstOffset = 0  //
                                    ) noexcept -> std::expected<void, std::string>;

    [[nodiscard]] auto copyToBuffer(std::span<std::byte const> src,  //
                                    size_t dstOffset = 0  //
                                    ) noexcept -> std::expected<void, std::string>;

    template <typename T>
    [[nodiscard]] auto copyToBuffer(std::span<T const> src,  //
                                    size_t dstOffset = 0  //
                                    ) noexcept -> std::expected<void, std::string> {
        return copyToBuffer(std::as_bytes(src), dstOffset);
    }

    template <typename T>
    [[nodiscard]] auto copyFromBuffer(std::span<T> dst,  //
                                      size_t srcOffset = 0  //
                                      ) noexcept -> std::expected<void, std::string> {
        return copyFromBuffer(std::as_writable_bytes(dst), srcOffset);
    }

    [[nodiscard]] auto copyFromBuffer(std::span<std::byte> dst,  //
                                      size_t srcOffset = 0  //
                                      ) noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;
    auto size() const noexcept -> size_t;
    auto buffer() const noexcept -> VkBuffer;
    auto type() const noexcept -> Type;

private:
    [[nodiscard]] auto init(size_t sizeBytes, Type type) noexcept -> std::expected<void, std::string>;

private:
    VkBuffer m_buffer{VK_NULL_HANDLE};
    uint64_t m_size{0};
    VmaAllocation m_allocation{VK_NULL_HANDLE};
    Type m_type{Type::None};
};

class BufferSpan {
public:
    BufferSpan() noexcept = default;

    BufferSpan(Buffer const& buffer) noexcept;

    [[nodiscard]] static auto create(Buffer const& buffer,  //
                                     size_t offset,  //
                                     size_t size  //
                                     ) noexcept -> std::expected<BufferSpan, std::string>;

public:
    auto buffer() const noexcept -> Buffer const&;
    auto size() const noexcept -> size_t;
    auto offset() const noexcept -> size_t;

private:
    Buffer const& m_buffer;
    size_t m_offset{0};
    size_t m_size{0};
};

}  // namespace gpu
