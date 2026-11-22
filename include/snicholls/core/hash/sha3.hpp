//
//  sha3.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief SHA3-224/256/384/512 and SHAKE128/256 (FIPS 202), and Ethereum's Keccak-256, on one Keccak-f[1600].
/// \ingroup core
///
/// **Overview.** The Keccak sponge: a 1600-bit state, permuted by `keccak_f1600`, absorbing input `rate` bytes
/// at a time and squeezing output the same way. The hashes differ only in their rate and their padding byte:
///
/// | Name | Rate (bytes) | Padding | Output |
/// |---|---|---|---|
/// | `sha3_224` / `sha3_256` / `sha3_384` / `sha3_512` | 144 / 136 / 104 / 72 | `0x06` (FIPS 202) | 28 / 32 / 48 / 64 bytes |
/// | `shake128` / `shake256` | 168 / 136 | `0x1F` | any length (an extendable-output function) |
/// | `keccak256` | 136 | `0x01` (Keccak's submission, before FIPS 202) | 32 bytes |
///
/// `keccak256` is not SHA3-256: Ethereum adopted Keccak before NIST changed the padding, so addresses, function
/// selectors, event topics, storage slots and EIP-712 hashes are Keccak-256, and a SHA3-256 in its place gives a
/// different, silently wrong, answer. Every function is `constexpr`, so a function selector is a compile-time
/// constant:
///
/// \snippet hash/sha3_demo.cpp keccak
///
/// The fixed-length hashes have the incremental interface of every hash in core (`update`, `finish`) and a one-shot
/// `hash`. SHAKE has `update`, then `squeeze(out)` any number of times for any lengths (the stream is the same
/// however it is cut), `finish<N>()` for one `digest<N>`, and `hash<N>(input)`.
///
/// **Relationships.** `digest<N>` and `hash_input` (`hash/digest.hpp`); `hmac<sha3_256>` (`hash/hmac.hpp`), whose
/// block size is the rate; `uint256` (`uint256.hpp`) for the amounts an ABI word carries. Group: \ref core.
///
/// **Implementations / variants.** One sponge, `keccak_sponge<Rate, Pad, Digest>`; the names above are its
/// aliases, and `keccak_xof<Rate>` is the same sponge squeezing without a fixed length.
///
/// **Guarantees.** Every member is `constexpr` and `noexcept`; no allocation; a hasher is a trivially copyable
/// value (208 bytes of state and buffer, a little more for the rate). Complexity: one permutation (24 rounds) per
/// `rate` bytes absorbed or squeezed. After `finish`, or for SHAKE after the first `squeeze`, `update` must not be
/// called again.
///
/// **Errors.** None.
///
/// **Examples.** Above; `demos/hash/sha3_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/sha3_benchmark.cpp`: Keccak-256 and SHA3-256 against a hand-written sponge
/// and OpenSSL (when installed), after a fairness check. Release, Apple clang 17, Xeon W-3245 (a shared machine), 2026-11-20, 16 KiB / 64-byte messages: Keccak-256
/// core 315 / 139 MB/s, the hand sponge 198 / 91; SHA3-256 core 319 / 137, OpenSSL 423 / 91. (OpenSSL 3 before
/// 3.2 has no KECCAK-256; the row is skipped where it is missing.)
///
/// **Tests and demos.** `tests/hash/sha3_test.cpp` (FIPS 202's examples, NIST's byte-oriented short-message and
/// Monte Carlo vectors for SHA-3 and SHAKE, Ethereum's well-known Keccak-256 values, Python's `hashlib` and
/// PyCryptodome's Keccak on random lengths, every split, every squeeze split), `demos/hash/sha3_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `hash/digest.hpp`.
///
/// **Known issues.** cSHAKE, KMAC, TupleHash and ParallelHash (SP 800-185) are not provided. SHAKE's security is
/// capped by its capacity (128 or 256 bits) whatever the output length.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_sha3_hpp
#define snicholls_core_hash_sha3_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/hash/digest.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls {

/// \brief The Keccak-f[1600] permutation (FIPS 202 3.3): 24 rounds over 25 64-bit lanes, `A[x + 5y]`.
/// \ingroup core
/// \param a The state, permuted in place.
constexpr void keccak_f1600(std::array<std::uint64_t, 25>& a) noexcept
{
    constexpr std::uint64_t rc[24] = {
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000, 0x000000000000808b,
        0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008a, 0x0000000000000088,
        0x0000000080008009, 0x000000008000000a, 0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
        0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008};
    // rho and pi together: lane pi_lane[i] takes the previous lane, rotated by rho[i]
    constexpr unsigned rho[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    constexpr std::size_t pi_lane[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
    for (std::size_t round = 0; round < 24; ++round) {
        std::uint64_t c[5]{};
        for (std::size_t x = 0; x < 5; ++x)
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (std::size_t x = 0; x < 5; ++x) {
            const std::uint64_t d = c[(x + 4) % 5] ^ detail::rotl<std::uint64_t>(c[(x + 1) % 5], 1);
            for (std::size_t y = 0; y < 25; y += 5)
                a[y + x] ^= d;
        }
        std::uint64_t t = a[1];
        for (std::size_t i = 0; i < 24; ++i) {
            const std::uint64_t next = a[pi_lane[i]];
            a[pi_lane[i]] = detail::rotl<std::uint64_t>(t, rho[i]);
            t = next;
        }
        for (std::size_t y = 0; y < 25; y += 5) {
            const std::uint64_t r0 = a[y], r1 = a[y + 1], r2 = a[y + 2], r3 = a[y + 3], r4 = a[y + 4];
            a[y] = r0 ^ (~r1 & r2);
            a[y + 1] = r1 ^ (~r2 & r3);
            a[y + 2] = r2 ^ (~r3 & r4);
            a[y + 3] = r3 ^ (~r4 & r0);
            a[y + 4] = r4 ^ (~r0 & r1);
        }
        a[0] ^= rc[round];
    }
}

namespace detail {

    // The sponge's state, absorption and squeezing, shared by the fixed-length hashes and the XOFs.
    template <std::size_t Rate, std::uint8_t Pad>
    class keccak_core
    {
        static_assert(Rate % 8 == 0 && Rate > 0 && Rate < 200, "snicholls::keccak: the rate is a whole number of lanes");

    public:
        template <typename U>
        constexpr void absorb(const U* p, std::size_t n) noexcept
        {
            in_.feed(p, n, [this](const auto* b) { this->absorb_block(b); });
        }

        // Pads the last block and permutes; the first squeezed byte is lane 0's lowest.
        constexpr void pad() noexcept
        {
            for (std::size_t i = in_.fill; i < Rate; ++i)
                in_.buf[i] = 0;
            in_.buf[in_.fill] ^= Pad;
            in_.buf[Rate - 1] ^= 0x80;
            absorb_block(in_.buf.data());
            position_ = 0;
        }

        template <typename Out>
        constexpr void squeeze(Out* out, std::size_t n) noexcept
        {
            for (std::size_t i = 0; i < n; ++i) {
                if (position_ == Rate) {
                    keccak_f1600(a_);
                    position_ = 0;
                }
                out[i] = static_cast<Out>(static_cast<std::uint8_t>(a_[position_ / 8] >> (8 * (position_ % 8))));
                ++position_;
            }
        }

    private:
        template <typename U>
        constexpr void absorb_block(const U* b) noexcept
        {
            for (std::size_t i = 0; i < Rate / 8; ++i)
                a_[i] ^= load_le_word<std::uint64_t>(b + 8 * i);
            keccak_f1600(a_);
        }

        std::array<std::uint64_t, 25> a_{};
        block_feeder<Rate> in_{};
        std::size_t position_ = 0;
    };

} // namespace detail

/// \brief A fixed-length Keccak sponge hash: the class behind `sha3_224` ... `sha3_512` and `keccak256`.
/// \ingroup core
/// \tparam Rate The rate in bytes (200 minus twice the digest size, for these).
/// \tparam Pad The domain padding byte: `0x06` for SHA-3, `0x01` for Keccak.
/// \tparam DigestSize The digest size in bytes.
template <std::size_t Rate, std::uint8_t Pad, std::size_t DigestSize>
class keccak_sponge : public detail::one_shot_hash<keccak_sponge<Rate, Pad, DigestSize>>
{
    static_assert(DigestSize <= Rate, "snicholls::keccak_sponge: the digest fits in one squeeze");

public:
    /// \brief The rate in bytes: what `hmac` pads the key to.
    static constexpr std::size_t block_size = Rate;
    /// \brief The digest size in bytes.
    static constexpr std::size_t digest_size = DigestSize;
    /// \brief The digest type.
    using digest_type = digest<DigestSize>;

    /// \brief A hasher with nothing absorbed.
    constexpr keccak_sponge() noexcept = default;

    /// \brief Pads, permutes and returns the digest. The hasher is not used afterwards.
    /// \return The digest.
    [[nodiscard]] constexpr digest_type finish() noexcept
    {
        digest_type out{};
        core_.pad();
        core_.squeeze(out.bytes.data(), DigestSize);
        return out;
    }

private:
    friend detail::hash_updates<keccak_sponge>;
    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        core_.absorb(p, n);
    }

    detail::keccak_core<Rate, Pad> core_{};
};

/// \brief A Keccak extendable-output function: the class behind `shake128` and `shake256`.
/// \ingroup core
///
/// `update` absorbs; the first `squeeze` pads, and every `squeeze` continues the same output stream, so
/// `squeeze(10 bytes)` then `squeeze(22 bytes)` writes what one `squeeze(32 bytes)` would.
/// \tparam Rate The rate in bytes: 168 for SHAKE128, 136 for SHAKE256.
template <std::size_t Rate>
class keccak_xof : public detail::hash_updates<keccak_xof<Rate>>
{
public:
    /// \brief The rate in bytes.
    static constexpr std::size_t block_size = Rate;

    /// \brief A function with nothing absorbed.
    constexpr keccak_xof() noexcept = default;

    /// \brief Writes the next `out.size()` bytes of output; the first call ends the input.
    /// \param out Where to write.
    constexpr void squeeze(std::span<std::byte> out) noexcept
    {
        if (!squeezing_) {
            core_.pad();
            squeezing_ = true;
        }
        core_.squeeze(out.data(), out.size());
    }

    /// \brief The next `N` bytes of output, as a digest; the first call ends the input.
    /// \tparam N The length.
    /// \return The bytes.
    template <std::size_t N>
    [[nodiscard]] constexpr digest<N> finish() noexcept
    {
        digest<N> out{};
        squeeze(out.bytes);
        return out;
    }

    /// \brief The first `N` bytes of output for `input`.
    /// \tparam N The length.
    /// \param input The input.
    /// \return The bytes.
    template <std::size_t N, hash_input S>
    [[nodiscard]] static constexpr digest<N> hash(const S& input) noexcept
    {
        keccak_xof x;
        x.update(input);
        return x.template finish<N>();
    }

private:
    friend detail::hash_updates<keccak_xof>;
    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        core_.absorb(p, n);
    }

    detail::keccak_core<Rate, 0x1f> core_{};
    bool squeezing_ = false;
};

/// \brief SHA3-224 (FIPS 202): rate 144, 28 bytes.
/// \ingroup core
using sha3_224 = keccak_sponge<144, 0x06, 28>;
/// \brief SHA3-256 (FIPS 202): rate 136, 32 bytes.
/// \ingroup core
using sha3_256 = keccak_sponge<136, 0x06, 32>;
/// \brief SHA3-384 (FIPS 202): rate 104, 48 bytes.
/// \ingroup core
using sha3_384 = keccak_sponge<104, 0x06, 48>;
/// \brief SHA3-512 (FIPS 202): rate 72, 64 bytes.
/// \ingroup core
using sha3_512 = keccak_sponge<72, 0x06, 64>;
/// \brief Keccak-256 with the original padding `0x01`: Ethereum's hash. Not SHA3-256.
/// \ingroup core
using keccak256 = keccak_sponge<136, 0x01, 32>;
/// \brief SHAKE128 (FIPS 202): rate 168, any output length, 128-bit security.
/// \ingroup core
using shake128 = keccak_xof<168>;
/// \brief SHAKE256 (FIPS 202): rate 136, any output length, 256-bit security.
/// \ingroup core
using shake256 = keccak_xof<136>;

} // namespace snicholls

#endif /* snicholls_core_hash_sha3_hpp */
