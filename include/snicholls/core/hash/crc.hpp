//
//  crc.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief CRC-32 (zlib's), CRC-32C (Castagnoli) and CRC-64/XZ, with their tables built by the compiler.
/// \ingroup core
///
/// **Overview.** Cyclic redundancy checks detect accidental corruption: a flipped bit on a wire or a disk, a
/// truncated file. They are not hashes against an adversary, who can forge any CRC. Three are provided, each the
/// parameterisation its users mean:
///
/// | Name | Polynomial (reflected) | Check value of `"123456789"` | Used by |
/// |---|---|---|---|
/// | `crc32` | `0xEDB88320` | `0xCBF43926` | zlib, gzip, PNG, ZIP, Ethernet, SATA |
/// | `crc32c` | `0x82F63B78` | `0xE3069283` | iSCSI, SCTP, ext4, Btrfs, LevelDB, RocksDB, Kafka, gRPC |
/// | `crc64_xz` | `0xC96C5795D7870F42` | `0x995DC9BBDF1939FA` | xz, 7-Zip (CRC-64/ECMA-182 reflected) |
///
/// Each is a class with the incremental interface of the hashes (`update`, `finish`) and a one-shot `hash`,
/// returning the integer as zlib does rather than a digest. `resume(crc)` continues from a finished value, which
/// is zlib's `crc32(crc, buf, len)`: `crc32::resume(crc32::hash(a)).update(b).finish() == crc32::hash(a + b)`.
///
/// \snippet hash/crc_demo.cpp crc
///
/// The tables (eight of 256 entries each, for slicing-by-8) are `constexpr` values the compiler builds from the
/// polynomial; every function is `constexpr`, so a CRC of a literal is a constant.
///
/// **Relationships.** `hash_input` (`hash/digest.hpp`); `fnv1a_64` and `wyhash` (`hash/fnv.hpp`,
/// `hash/wyhash.hpp`) for hash tables; `sha256` or `blake3` where tampering must be detected. Group: \ref core.
///
/// **Implementations / variants.** `reflected_crc<T, Poly, Init, XorOut>` is the one class (a reflected CRC with
/// equal input and output reflection, which all three are); the names above are its aliases.
///
/// **Guarantees.** Every member is `constexpr` and `noexcept`; no allocation; a CRC object is one integer.
/// Complexity: eight table lookups per eight bytes, one per remaining byte.
///
/// **Errors.** None.
///
/// **Examples.** Above; `demos/hash/crc_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/crc_benchmark.cpp`: against the same slicing-by-8 written by hand and the
/// byte-at-a-time loop, after a fairness check. Release, Apple clang 17, Xeon W-3245 (a shared machine), 2026-11-20, 16 KiB / 64-byte messages: CRC-32 core 2.11 /
/// 2.76 GB/s, hand 2.14 / 2.84, byte at a time 0.49 / 0.57; CRC-32C and CRC-64/XZ alike. Hardware CRC-32C (SSE4.2 `crc32`, ARMv8 `crc32c`) and
/// zlib's PCLMUL / PMULL folding are several times faster; core has no platform headers, so a library above core
/// may add them, checked against these definitions.
///
/// **Tests and demos.** `tests/hash/crc_test.cpp` (the catalogue's check values, Python's `zlib.crc32` and
/// `crcmod` on random lengths, every split, `resume`), `demos/hash/crc_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `hash/digest.hpp`.
///
/// **Known issues.** CRC-64 has several incompatible parameterisations (ECMA-182 unreflected as in
/// CRC-64/ECMA-182, CRC-64/GO-ISO, CRC-64/NVME); only CRC-64/XZ is provided.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_crc_hpp
#define snicholls_core_hash_crc_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/hash/digest.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace snicholls {

namespace detail {

    // Slicing-by-8: table[0] is the byte-at-a-time table; table[k][b] is table[0][b] advanced by k zero bytes.
    template <typename T, T Poly>
    inline constexpr std::array<std::array<T, 256>, 8> crc_tables = [] {
        std::array<std::array<T, 256>, 8> t{};
        for (std::size_t b = 0; b < 256; ++b) {
            T c = static_cast<T>(b);
            for (int k = 0; k < 8; ++k)
                c = static_cast<T>((c & 1) ? (c >> 1) ^ Poly : c >> 1);
            t[0][b] = c;
        }
        for (std::size_t k = 1; k < 8; ++k)
            for (std::size_t b = 0; b < 256; ++b)
                t[k][b] = static_cast<T>((t[k - 1][b] >> 8) ^ t[0][t[k - 1][b] & 0xff]);
        return t;
    }();

} // namespace detail

/// \brief A reflected CRC: the class behind `crc32`, `crc32c` and `crc64_xz`.
/// \ingroup core
/// \tparam T The register type (`std::uint32_t`, `std::uint64_t`).
/// \tparam Poly The reflected polynomial.
/// \tparam Init The initial register.
/// \tparam XorOut What the register is xored with to give the CRC.
template <typename T, T Poly, T Init, T XorOut>
class reflected_crc : public detail::hash_updates<reflected_crc<T, Poly, Init, XorOut>>,
                      public detail::one_shot_hash<reflected_crc<T, Poly, Init, XorOut>>
{
public:
    /// \brief The CRC's type.
    using value_type = T;

    /// \brief A CRC of nothing so far.
    constexpr reflected_crc() noexcept = default;

    /// \brief A CRC that continues from a finished value: zlib's `crc32(crc, ...)`.
    /// \param crc What `finish` or `hash` returned for the input so far.
    /// \return The CRC object.
    [[nodiscard]] static constexpr reflected_crc resume(T crc) noexcept
    {
        reflected_crc c;
        c.reg_ = static_cast<T>(crc ^ XorOut);
        return c;
    }

    /// \brief The CRC of everything absorbed. `const`: more can be absorbed afterwards.
    /// \return The CRC.
    [[nodiscard]] constexpr T finish() const noexcept { return static_cast<T>(reg_ ^ XorOut); }

private:
    friend detail::hash_updates<reflected_crc>;

    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        constexpr const auto& t = detail::crc_tables<T, Poly>;
        T c = reg_;
        for (; n >= 8; p += 8, n -= 8) {
            const std::uint64_t w = static_cast<std::uint64_t>(c) ^ detail::load_le_word<std::uint64_t>(p);
            c = static_cast<T>(t[7][w & 0xff] ^ t[6][(w >> 8) & 0xff] ^ t[5][(w >> 16) & 0xff] ^ t[4][(w >> 24) & 0xff] ^
                               t[3][(w >> 32) & 0xff] ^ t[2][(w >> 40) & 0xff] ^ t[1][(w >> 48) & 0xff] ^ t[0][w >> 56]);
        }
        for (std::size_t i = 0; i < n; ++i)
            c = static_cast<T>((c >> 8) ^ t[0][(c ^ detail::octet_of(p[i])) & 0xff]);
        reg_ = c;
    }

    T reg_ = Init;
};

/// \brief CRC-32 as zlib, gzip, PNG and ZIP compute it (CRC-32/ISO-HDLC): check value `0xCBF43926`.
/// \ingroup core
using crc32 = reflected_crc<std::uint32_t, 0xEDB88320u, 0xFFFFFFFFu, 0xFFFFFFFFu>;

/// \brief CRC-32C, Castagnoli's polynomial (CRC-32/ISCSI): check value `0xE3069283`.
/// \ingroup core
using crc32c = reflected_crc<std::uint32_t, 0x82F63B78u, 0xFFFFFFFFu, 0xFFFFFFFFu>;

/// \brief CRC-64/XZ (the ECMA-182 polynomial, reflected): check value `0x995DC9BBDF1939FA`.
/// \ingroup core
using crc64_xz = reflected_crc<std::uint64_t, 0xC96C5795D7870F42ull, ~0ull, ~0ull>;

} // namespace snicholls

#endif /* snicholls_core_hash_crc_hpp */
