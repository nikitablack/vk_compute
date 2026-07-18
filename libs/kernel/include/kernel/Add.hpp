#pragma once

#include <expected>
#include <gpu/Buffer.hpp>
#include <gpu/HostVisibleBuffer.hpp>
#include <gpu/Span.hpp>
#include <optional>
#include <span>
#include <string>

namespace kernel {

class Add {
public:
    Add(const Add&) = delete;
    Add(Add&& other) noexcept;
    Add& operator=(const Add&) = delete;
    Add& operator=(Add&&) = delete;

private:
    Add() = default;

    [[nodiscard]] auto init(uint32_t workgroupSizeX) noexcept -> std::expected<void, std::string>;

public:
    [[nodiscard]] static auto create(uint32_t workgroupSizeX = 128) noexcept -> std::expected<Add, std::string>;

    /**
     * Creates temporary device buffer, copies host inputs `srcA` and `srcB` to the device, after computing copies the
     * device buffer to host `dst`.
     *
     * The sizes of all three spans (for `srcA`, `srcB`, `dst`) must be identical.
     *
     * It is valid for all three inputs to overlap, in this case, the overlapped region will be overwritten by the
     * `dst`.
     */
    [[nodiscard]] auto operator()(std::span<float const> srcA,  //
                                  std::span<float const> srcB,  //
                                  std::span<float> dst  //
    ) const noexcept -> std::expected<void, std::string>;

    /**
     * Does not create any temporary buffers. This is useful when a large device buffer is already available and should
     * be reused without additional allocations.
     *
     * The sizes of all three spans (`srcA.size()`, `srcB.size()`, and `dst.size()`) must be identical.
     *
     * The `dst` span must not overlap with `srcA` or `srcB`. Overlap is checked only when the provided buffers refer to
     * the same Vulkan buffer object (`VkBuffer`). If an overlap is detected, the function returns an error. Overlap
     * between different Vulkan buffers cannot be detected; providing overlapping memory regions in this case results in
     * undefined behavior.
     */
    [[nodiscard]] auto operator()(gpu::BufferSpan const& srcA,  //
                                  gpu::BufferSpan const& srcB,  //
                                  gpu::BufferSpan const& dst  //
    ) const noexcept -> std::expected<void, std::string>;

    /**
     * Copies data from device `src` to host `dst`. Uses `stagingBuffer` as intermediate buffer between GPU and CPU, if
     * provided. If `stagingBuffer` is not provided, the function creates a temporary buffer that is destroyed after the
     * function is finished.
     */
    [[nodiscard]] auto read(std::span<float> dst,  //
                            gpu::BufferSpan const& src,  //
                            std::optional<gpu::Buffer> stagingBuffer = std::nullopt  //
    ) const noexcept -> std::expected<void, std::string>;

    auto destroy() noexcept -> void;

private:
    VkPipeline m_pipeline{VK_NULL_HANDLE};
    uint32_t m_workgroupSizeX{0};
};

}  // namespace kernel
