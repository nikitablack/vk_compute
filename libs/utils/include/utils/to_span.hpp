#include <array>
#include <expected>
#include <span>

namespace utils {

template <typename T, size_t N>
auto to_byte_span(std::array<T, N> const& c) noexcept -> std::span<std::byte const> {
    std::span<std::byte const> bytes(reinterpret_cast<std::byte const*>(c.data()), c.size() * sizeof(T));

    return bytes;
}

// TODO: move to .cpp
inline auto to_byte_span(char const* c, size_t n) noexcept -> std::span<std::byte const> {
    std::span<std::byte const> bytes(reinterpret_cast<std::byte const*>(c), n);

    return bytes;
}

template <typename T>
auto to_byte_span(std::span<T> const& c) noexcept -> std::span<std::byte const> {
    std::span<std::byte const> bytes(reinterpret_cast<std::byte const*>(c.data()), c.size() * sizeof(T));

    return bytes;
}

}  // namespace utils
