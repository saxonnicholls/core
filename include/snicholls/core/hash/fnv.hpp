//
//  fnv.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief FNV-1a hashing, usable at compile time and the same on every machine.
/// \ingroup core
///
/// **Overview.**
/// The Fowler-Noll-Vo FNV-1a hash in its 32- and 64-bit forms. It is not
/// cryptographic and not the fastest hash for long inputs, but it is tiny,
/// `constexpr`, and specified byte for byte, so a hash computed by the
/// compiler on a Mac equals one computed at run time on a Raspberry Pi. That
/// is what the family uses it for: names in compile-time dispatch tables,
/// switch labels on strings, and structural signatures that two machines must
/// agree on.
///
/// \snippet hash/fnv_demo.cpp switch
///
/// `fnv1a_64_append` folds an integer into a running hash through its
/// little-endian bytes, so composite hashes do not depend on the host's byte
/// order either.
///
/// **Relationships.**
/// `hash_combine` mixes `std::hash` values for unordered
/// containers, where only one process needs to agree; use FNV-1a where a hash
/// crosses a process or machine boundary or must be a constant expression.
///
/// **Guarantees.**
/// Every function is `constexpr` and `noexcept`. The values are the published
/// FNV-1a test vectors (see the tests), and never change between versions.
///
/// **Errors.**
/// None.
///
/// **Performance.**
/// One multiply and one xor per byte. Not benchmarked; for hashing large
/// buffers at run time use a modern non-cryptographic hash instead.
///
/// **Tests and demos.**
/// `tests/hash/fnv_test.cpp`, `demos/hash/fnv_demo.cpp`.
///
/// **Dependencies.**
/// `<cstdint>`, `<string_view>`, `<type_traits>`, and `endian.hpp`.
///
/// **Status.**
/// Experimental.

#ifndef snicholls_core_hash_fnv_hpp
#define snicholls_core_hash_fnv_hpp

#include <snicholls/core/endian.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace snicholls {

/// \brief The 64-bit FNV offset basis: the hash of the empty input.
inline constexpr std::uint64_t fnv1a_64_basis = 0xcbf29ce484222325ull;
/// \brief The 64-bit FNV prime.
inline constexpr std::uint64_t fnv1a_64_prime = 0x100000001b3ull;
/// \brief The 32-bit FNV offset basis: the hash of the empty input.
inline constexpr std::uint32_t fnv1a_32_basis = 0x811c9dc5u;
/// \brief The 32-bit FNV prime.
inline constexpr std::uint32_t fnv1a_32_prime = 0x01000193u;

/// \brief 64-bit FNV-1a of the bytes of `s`, continuing from `h`.
/// \param s The bytes to hash.
/// \param h The running hash; the default starts a new one.
/// \return The hash.
constexpr std::uint64_t fnv1a_64(std::string_view s, std::uint64_t h = fnv1a_64_basis) noexcept
{
    for (const char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= fnv1a_64_prime;
    }
    return h;
}

/// \brief 32-bit FNV-1a of the bytes of `s`, continuing from `h`.
/// \param s The bytes to hash.
/// \param h The running hash; the default starts a new one.
/// \return The hash.
constexpr std::uint32_t fnv1a_32(std::string_view s, std::uint32_t h = fnv1a_32_basis) noexcept
{
    for (const char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= fnv1a_32_prime;
    }
    return h;
}

/// \brief Fold a value into a running 64-bit FNV-1a hash through its little-endian bytes.
///
/// The same on every host: `fnv1a_64_append(h, std::uint32_t{1})` hashes the
/// bytes 01 00 00 00 whether the machine is little- or big-endian.
/// \tparam T A byte_order_value (integer, enumeration, `float` or `double`).
/// \param h The running hash.
/// \param v The value to fold in.
/// \return The new hash.
template <byte_order_value T>
constexpr std::uint64_t fnv1a_64_append(std::uint64_t h, T v) noexcept
{
    const auto bytes = to_le_bytes(v);
    for (const unsigned char b : bytes) {
        h ^= b;
        h *= fnv1a_64_prime;
    }
    return h;
}

/// \brief Literals for compile-time string hashes: `"GET"_fnv64`, `"GET"_fnv32`.
///
/// For `switch` on a string: the labels are constant expressions, and the
/// subject is `fnv1a_64(s)` at run time. Check the matched string too if a
/// collision would matter.
namespace literals {

/// \brief `"text"_fnv64` is `fnv1a_64("text")`, as a constant.
/// \param s The literal's characters.
/// \param n Their count.
/// \return The 64-bit hash.
consteval std::uint64_t operator""_fnv64(const char* s, std::size_t n) noexcept { return fnv1a_64({s, n}); }

/// \brief `"text"_fnv32` is `fnv1a_32("text")`, as a constant.
/// \param s The literal's characters.
/// \param n Their count.
/// \return The 32-bit hash.
consteval std::uint32_t operator""_fnv32(const char* s, std::size_t n) noexcept { return fnv1a_32({s, n}); }

} // namespace literals

} // namespace snicholls

#endif /* snicholls_core_hash_fnv_hpp */
