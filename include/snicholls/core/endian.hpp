//
//  endian.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Explicit byte order: `load_le`, `store_le`, `load_be`, `store_be`, `byteswap`.
/// \ingroup core
///
/// **Overview.**
/// Read and write integers and floating-point values in a fixed byte order,
/// whatever the machine's own. Every wire format and file format in the family
/// needs this - the remote-compute frames, WebSocket lengths, the binary
/// codecs - and each had been writing the shifts out by hand:
///
/// \snippet endian_demo.cpp frame
///
/// The functions are written as shifts, not as a `memcpy` plus a conditional
/// swap, so the result never depends on the host and they work in constant
/// expressions. Compilers recognise the pattern: on a little-endian machine
/// `load_le` compiles to one load, and `load_be` to a load and a `bswap`.
///
/// **Relationships.**
/// `byteswap` is C++23's `std::byteswap`, provided here because the family is
/// C++20. `std::endian` tells you the host's order, which these functions make
/// you never need to ask.
///
/// **Guarantees.**
/// All `constexpr` and `noexcept`. `T` is an integral type other than `bool`,
/// an enumeration (its underlying type's bytes), `float` or `double` (their
/// IEEE bits). The pointer is to `sizeof(T)` readable or writable bytes of
/// `char`, `unsigned char`, `signed char` or `std::byte`; it need not be aligned.
///
/// **Errors.**
/// None. A pointer to fewer than `sizeof(T)` bytes is undefined behaviour, as for
/// `memcpy`.
///
/// **Performance.**
/// Not benchmarked in core. The shift form is the one compilers fold into a
/// single load or store; check a disassembly before replacing it with `memcpy`.
///
/// **Tests and demos.**
/// `tests/endian_test.cpp`, `demos/endian_demo.cpp`.
///
/// **Dependencies.**
/// `<array>`, `<bit>`, `<cstddef>`, `<cstdint>`, `<type_traits>`.
///
/// **Status.**
/// Experimental.

#ifndef snicholls_core_endian_hpp
#define snicholls_core_endian_hpp

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace snicholls {

/// \brief A byte type `load_*` and `store_*` accept: `char`, `unsigned char`, `signed char` or `std::byte`.
/// \tparam B The candidate byte type, cv-qualifiers ignored.
template <typename B>
concept byte_type = std::is_same_v<std::remove_cv_t<B>, char> || std::is_same_v<std::remove_cv_t<B>, unsigned char> ||
                    std::is_same_v<std::remove_cv_t<B>, signed char> || std::is_same_v<std::remove_cv_t<B>, std::byte>;

/// \brief A value with a defined byte representation: a non-`bool` integer, an enumeration, `float` or `double`.
/// \tparam T The candidate type.
template <typename T>
concept byte_order_value = (std::is_integral_v<T> && !std::is_same_v<T, bool>) || std::is_enum_v<T> ||
                           std::is_same_v<T, float> || std::is_same_v<T, double>;

namespace detail {

template <typename T>
struct bits_of {
    using type = std::make_unsigned_t<T>;
};
template <typename T>
    requires std::is_enum_v<T>
struct bits_of<T> {
    using type = std::make_unsigned_t<std::underlying_type_t<T>>;
};
template <>
struct bits_of<float> {
    using type = std::uint32_t;
};
template <>
struct bits_of<double> {
    using type = std::uint64_t;
};

template <typename T>
using bits_t = typename bits_of<T>::type;

template <typename T>
constexpr bits_t<T> to_bits(T v) noexcept
{
    if constexpr (std::is_floating_point_v<T>)
        return std::bit_cast<bits_t<T>>(v);
    else
        return static_cast<bits_t<T>>(v);
}

template <typename T>
constexpr T from_bits(bits_t<T> b) noexcept
{
    if constexpr (std::is_floating_point_v<T>)
        return std::bit_cast<T>(b);
    else
        return static_cast<T>(b);
}

template <byte_type B>
constexpr unsigned char octet(B b) noexcept
{
    return static_cast<unsigned char>(b);
}

} // namespace detail

/// \brief Reverse the bytes of an integer, as C++23's `std::byteswap` does.
/// \tparam T A non-`bool` integral type.
/// \param v The value.
/// \return `v` with its byte order reversed.
template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
constexpr T byteswap(T v) noexcept
{
    // All arithmetic in 64 bits: every shift is narrower than its operand for
    // every T, so no compiler can warn that one might not be (MSVC C4333 did)
    std::uint64_t in = static_cast<std::make_unsigned_t<T>>(v), out = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        out = (out << 8) | (in & 0xffu);
        in >>= 8;
    }
    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(out));
}

/// \brief Write `v` to `p` as `sizeof(T)` bytes, least significant first.
/// \tparam T A byte_order_value.
/// \tparam B A byte_type.
/// \param p Where to write; `sizeof(T)` bytes, any alignment.
/// \param v The value.
template <byte_order_value T, byte_type B>
constexpr void store_le(B* p, T v) noexcept
{
    const std::uint64_t b = detail::to_bits(v);
    for (std::size_t i = 0; i < sizeof(T); ++i)
        p[i] = static_cast<B>(static_cast<unsigned char>((b >> (8 * i)) & 0xffu));
}

/// \brief Write `v` to `p` as `sizeof(T)` bytes, most significant first (network order).
/// \tparam T A byte_order_value.
/// \tparam B A byte_type.
/// \param p Where to write; `sizeof(T)` bytes, any alignment.
/// \param v The value.
template <byte_order_value T, byte_type B>
constexpr void store_be(B* p, T v) noexcept
{
    const std::uint64_t b = detail::to_bits(v);
    for (std::size_t i = 0; i < sizeof(T); ++i)
        p[sizeof(T) - 1 - i] = static_cast<B>(static_cast<unsigned char>((b >> (8 * i)) & 0xffu));
}

/// \brief Read a `T` from `sizeof(T)` bytes at `p`, least significant first.
/// \tparam T A byte_order_value, named explicitly: `load_le<std::uint32_t>(p)`.
/// \tparam B A byte_type.
/// \param p Where to read; `sizeof(T)` bytes, any alignment.
/// \return The value.
template <byte_order_value T, byte_type B>
constexpr T load_le(const B* p) noexcept
{
    std::uint64_t b = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
        b |= std::uint64_t{detail::octet(p[i])} << (8 * i);
    return detail::from_bits<T>(static_cast<detail::bits_t<T>>(b));
}

/// \brief Read a `T` from `sizeof(T)` bytes at `p`, most significant first (network order).
/// \tparam T A byte_order_value, named explicitly: `load_be<std::uint16_t>(p)`.
/// \tparam B A byte_type.
/// \param p Where to read; `sizeof(T)` bytes, any alignment.
/// \return The value.
template <byte_order_value T, byte_type B>
constexpr T load_be(const B* p) noexcept
{
    std::uint64_t b = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i)
        b = (b << 8) | detail::octet(p[i]);
    return detail::from_bits<T>(static_cast<detail::bits_t<T>>(b));
}

/// \brief The little-endian bytes of `v`, as an array.
/// \tparam T A byte_order_value.
/// \param v The value.
/// \return `sizeof(T)` bytes, least significant first.
template <byte_order_value T>
constexpr std::array<unsigned char, sizeof(T)> to_le_bytes(T v) noexcept
{
    std::array<unsigned char, sizeof(T)> out{};
    store_le(out.data(), v);
    return out;
}

/// \brief The big-endian bytes of `v`, as an array.
/// \tparam T A byte_order_value.
/// \param v The value.
/// \return `sizeof(T)` bytes, most significant first.
template <byte_order_value T>
constexpr std::array<unsigned char, sizeof(T)> to_be_bytes(T v) noexcept
{
    std::array<unsigned char, sizeof(T)> out{};
    store_be(out.data(), v);
    return out;
}

} // namespace snicholls

#endif /* snicholls_core_endian_hpp */
