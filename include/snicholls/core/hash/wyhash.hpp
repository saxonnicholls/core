//
//  wyhash.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The arithmetic is the reference's; the reads are assembled byte by byte so the hash is the same on every
//  byte order and usable in a constant expression, and at run time contiguous input is read a word at a time
//  with memcpy (swapped on a big-endian host). The fingerprint helpers of the source stay with the sequences
//  module; the 128-bit multiply moved to detail/wide_mul.hpp.
//

/// \file
/// \brief wyhash final version 4.3: a fast 64-bit hash, usable at compile time and the same on every machine.
/// \ingroup core
///
/// **Overview.** The hash for a hash table, a fingerprint or a bloom filter: 48 bytes a round
/// through 64 x 64 -> 128-bit multiplies, passes SMHasher, a handful of instructions for the
/// short keys that dominate (a symbol, a token, a quantised state). It is `constexpr`, so a
/// closed alphabet's fingerprint or a dispatch table's keys are compile-time constants, and it
/// reads its input in little-endian order whatever the host, so the value is the same on a
/// big-endian machine as on this one, which the reference's `memcpy` reads are not.
///
/// \snippet hash/wyhash_demo.cpp hash
///
/// `wymix` is the 64-bit mixer the hash is built from, for folding two words into one;
/// `splitmix64` is a cheap integer mixer for seeds and ids. Neither is cryptographic, and
/// neither is `fnv1a_64`: use FNV where a tiny, specified-by-the-byte hash is wanted, wyhash
/// where speed over long keys or avalanche in the low bits matters.
///
/// **Relationships.** `hash/fnv.hpp`, the other machine-independent hash; `hash_combine`, for
/// `std::hash` values that need not cross a process; `endian.hpp`, whose `byteswap` keeps the
/// run-time reads little-endian; `containers/bloom_filter.hpp`, which hashes with it. Group:
/// \ref core.
///
/// **Guarantees.** Every function is `constexpr` and `noexcept`. The values are those of the
/// upstream `test_vector.cpp` for final version 4.3 (pinned in the tests), on every host and
/// in constant and run-time evaluation alike, and never change between versions. The default
/// secret is upstream's; a different secret is a different hash function. In a constant
/// expression, and for input that is not contiguous, the bytes are read one at a time
/// through `b[i]`; at run time, contiguous single-byte input (`std::string_view`, a span of
/// bytes, a `std::array<char, N>`) is read a word at a time with `memcpy`, byte-swapped on a
/// big-endian host, so the two paths agree bit for bit.
///
/// **Errors.** None.
///
/// **Performance.** Benchmarked in `benchmarks/hash/wyhash_benchmark.cpp` against the reference
/// written out with `memcpy` reads and the compiler's 128-bit multiply, on 4,096 keys of 1 to
/// 96 bytes, after a fairness check that every value agrees; `fnv1a_64` runs beside them for
/// scale. Release, Apple clang 17, Xeon W-3245, 2026-11-06: 8.9 ns per key through `wyhash`,
/// 7.9 ns through the reference, 38.6 ns through FNV-1a. The nanosecond is the call into the
/// outlined hash body, which the reference, one function in the benchmark's own translation
/// unit, does not pay; the arithmetic and the reads are the same.
///
/// **Tests and demos.** `tests/hash/wyhash_test.cpp`, `demos/hash/wyhash_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<cstring>`, `<span>`,
/// `<string_view>`, `<type_traits>`, `detail/wide_mul.hpp` and `endian.hpp` (for `byteswap`).
///
/// **Known issues.** Final version 4.3 is not the same function as final 4.0 or 4.1 (a
/// different secret and final mix): Zig's standard library and abseil carry older versions,
/// so their values differ from these. The literal `_wyhash` is in `snicholls::literals`.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_hash_wyhash_hpp
#define snicholls_core_hash_wyhash_hpp

#include <snicholls/core/detail/wide_mul.hpp>
#include <snicholls/core/endian.hpp>

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <type_traits>

namespace snicholls {

    /// \brief The default secret of wyhash final version 4.3: part of the function's definition.
    /// \ingroup core
    inline constexpr std::uint64_t wyhash_secret[4] = {0x2d358dccaa6c78a5ull, 0x8bb84b93962eacc9ull,
                                                       0x4b33a62ed433d4a3ull, 0x4d5a2da51de1aa47ull};

    /// \brief The wyhash mixer: the low and high halves of `a * b`, xored.
    /// \ingroup core
    /// \param a One word.
    /// \param b The other.
    /// \return The mix. Commutative, and 0 when either argument is 0.
    [[nodiscard]] constexpr std::uint64_t wymix(std::uint64_t a, std::uint64_t b) noexcept
    {
        const detail::wide_product p = detail::umul128(a, b);
        return p.low ^ p.high;
    }

    /// \brief The splitmix64 integer mixer: a 64-bit permutation with good avalanche, for seeds and ids.
    /// \ingroup core
    /// \param x The value.
    /// \return The mixed value.
    [[nodiscard]] constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept
    {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    namespace detail {

        // The reads the hash is built from, over anything with b[i] convertible to a byte: little-endian,
        // assembled byte by byte, so they are allowed in a constant expression (where reinterpret_cast is
        // not) and give the same bits on every host.
        template <typename B>
        struct wy_indexed {
            const B& b;

            constexpr std::uint64_t byte(std::size_t i) const noexcept
            {
                return static_cast<std::uint64_t>(static_cast<unsigned char>(b[i]));
            }
            constexpr std::uint64_t read8(std::size_t o) const noexcept
            {
                return byte(o) | (byte(o + 1) << 8) | (byte(o + 2) << 16) | (byte(o + 3) << 24) | (byte(o + 4) << 32) |
                       (byte(o + 5) << 40) | (byte(o + 6) << 48) | (byte(o + 7) << 56);
            }
            constexpr std::uint64_t read4(std::size_t o) const noexcept
            {
                return byte(o) | (byte(o + 1) << 8) | (byte(o + 2) << 16) | (byte(o + 3) << 24);
            }
            constexpr std::uint64_t read3(std::size_t o, std::size_t k) const noexcept
            {
                return (byte(o) << 16) | (byte(o + (k >> 1)) << 8) | byte(o + k - 1);
            }
        };

        // The same reads from a pointer at run time: one memcpy each, which is one load, swapped on a
        // big-endian host so the value is the little-endian one the constant-expression path produces.
        struct wy_pointer {
            const unsigned char* p;

            std::uint64_t read8(std::size_t o) const noexcept
            {
                std::uint64_t v = 0;
                std::memcpy(&v, p + o, sizeof v);
                return std::endian::native == std::endian::little ? v : byteswap(v);
            }
            std::uint64_t read4(std::size_t o) const noexcept
            {
                std::uint32_t v = 0;
                std::memcpy(&v, p + o, sizeof v);
                return std::endian::native == std::endian::little ? v : byteswap(v);
            }
            std::uint64_t read3(std::size_t o, std::size_t k) const noexcept
            {
                return (static_cast<std::uint64_t>(p[o]) << 16) | (static_cast<std::uint64_t>(p[o + (k >> 1)]) << 8) |
                       p[o + k - 1];
            }
        };

        // A contiguous container of single-byte elements: data() is its bytes.
        template <typename B>
        concept wy_contiguous_bytes = requires(const B& b) {
            { b.data() } -> std::convertible_to<const void*>;
            requires sizeof(*b.data()) == 1;
        };

        template <wy_contiguous_bytes B>
        const unsigned char* wy_data(const B& b) noexcept
        {
            return reinterpret_cast<const unsigned char*>(b.data());
        }

        constexpr const unsigned char* wy_data(const auto&) noexcept { return nullptr; }   // never read: see wyhash_bytes

        // wyhash final version 4.3 over a reader, taken by value: a pointer or a reference, in a register.
        template <typename Reader>
        constexpr std::uint64_t wy_hash(const Reader r, std::size_t len, std::uint64_t seed) noexcept
        {
            seed ^= wymix(seed ^ wyhash_secret[0], wyhash_secret[1]);
            std::uint64_t a = 0, hi = 0;
            std::size_t p = 0;   // the read offset, in place of the reference's moving pointer

            if (len <= 16) {
                if (len >= 4) {
                    const std::size_t off = (len >> 3) << 2;
                    a = (r.read4(0) << 32) | r.read4(off);
                    hi = (r.read4(len - 4) << 32) | r.read4(len - 4 - off);
                } else if (len > 0) {
                    a = r.read3(0, len);
                }
            } else {
                std::size_t i = len;
                if (i > 48) {
                    std::uint64_t see1 = seed, see2 = seed;
                    do {
                        seed = wymix(r.read8(p) ^ wyhash_secret[1], r.read8(p + 8) ^ seed);
                        see1 = wymix(r.read8(p + 16) ^ wyhash_secret[2], r.read8(p + 24) ^ see1);
                        see2 = wymix(r.read8(p + 32) ^ wyhash_secret[3], r.read8(p + 40) ^ see2);
                        p += 48;
                        i -= 48;
                    } while (i > 48);
                    seed ^= see1 ^ see2;
                }
                while (i > 16) {
                    seed = wymix(r.read8(p) ^ wyhash_secret[1], r.read8(p + 8) ^ seed);
                    i -= 16;
                    p += 16;
                }
                a = r.read8(p + i - 16);
                hi = r.read8(p + i - 8);
            }

            a ^= wyhash_secret[1];
            hi ^= seed;
            const wide_product m = umul128(a, hi);
            return wymix(m.low ^ wyhash_secret[0] ^ len, m.high ^ wyhash_secret[1]);
        }

    } // namespace detail

    /// \brief wyhash of the first `len` bytes of anything indexable: `b[i]` must convert to `unsigned char`.
    /// \ingroup core
    ///
    /// The form behind the overloads below, for a container the overloads do not cover (a
    /// `std::array<char, N>`, a `basic_fixed_string`). At run time a container with `data()`
    /// and single-byte elements is read a word at a time; everything else, and every constant
    /// expression, byte by byte through `b[i]`.
    /// \tparam B A type with `b[i]` convertible to `unsigned char`.
    /// \param b    The bytes.
    /// \param len  How many of them; `b[0]` to `b[len - 1]` are read.
    /// \param seed The seed; a different seed is a different hash function.
    /// \return The 64-bit hash.
    template <typename B>
        requires requires(const B& b, std::size_t i) { static_cast<unsigned char>(b[i]); }
    [[nodiscard]] constexpr std::uint64_t wyhash_bytes(const B& b, std::size_t len, std::uint64_t seed = 0) noexcept
    {
        // Both conditions are constants in any one instantiation, so only one path is ever compiled in.
        if (!std::is_constant_evaluated() && detail::wy_contiguous_bytes<B>)
            return detail::wy_hash(detail::wy_pointer{detail::wy_data(b)}, len, seed);
        return detail::wy_hash(detail::wy_indexed<B>{b}, len, seed);
    }

    /// \brief wyhash of the characters of `s`.
    /// \ingroup core
    /// \param s    The text; its bytes are hashed as they are.
    /// \param seed The seed.
    /// \return The 64-bit hash.
    [[nodiscard]] constexpr std::uint64_t wyhash(std::string_view s, std::uint64_t seed = 0) noexcept
    {
        return wyhash_bytes(s, s.size(), seed);
    }

    /// \brief wyhash of a span of bytes.
    /// \ingroup core
    /// \param bytes The bytes.
    /// \param seed  The seed.
    /// \return The 64-bit hash.
    [[nodiscard]] constexpr std::uint64_t wyhash(std::span<const std::byte> bytes, std::uint64_t seed = 0) noexcept
    {
        return wyhash_bytes(bytes, bytes.size(), seed);
    }

    /// \brief wyhash of a span of unsigned chars (`std::uint8_t`).
    /// \ingroup core
    /// \param bytes The bytes.
    /// \param seed  The seed.
    /// \return The 64-bit hash.
    [[nodiscard]] constexpr std::uint64_t wyhash(std::span<const unsigned char> bytes, std::uint64_t seed = 0) noexcept
    {
        return wyhash_bytes(bytes, bytes.size(), seed);
    }

    namespace literals {

        /// \brief `"text"_wyhash` is `wyhash("text")`, as a constant: a `case` label for a string.
        /// \param s The literal's characters.
        /// \param n Their count.
        /// \return The 64-bit hash.
        consteval std::uint64_t operator""_wyhash(const char* s, std::size_t n) noexcept { return wyhash({s, n}); }

    } // namespace literals

} // namespace snicholls

#endif /* snicholls_core_hash_wyhash_hpp */
