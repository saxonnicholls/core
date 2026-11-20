//
//  blake2.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief BLAKE2b and BLAKE2s (RFC 7693), keyed or not, with any output length, usable at compile time.
/// \ingroup core
///
/// **Overview.** BLAKE2b (64-bit words, 128-byte blocks, 1 to 64 bytes of output) and BLAKE2s (32-bit words,
/// 64-byte blocks, 1 to 32 bytes), each a class template over its output length: `blake2b<32>` is BLAKE2b-256,
/// `blake2b<64>` BLAKE2b-512. Faster than SHA-2 in software, and keyed directly: `blake2b<32>{key}` is a MAC with
/// no HMAC around it (RFC 7693 2.9). The incremental interface of every hash in core (`update`, `finish`) and a
/// one-shot `hash`; every function is `constexpr`.
///
/// \snippet hash/blake2_demo.cpp blake2
///
/// Where it is used: Polkadot and Substrate (SS58 addresses, storage keys), Zcash, Argon2's internals, libsodium's
/// `crypto_generichash`, WireGuard (BLAKE2s), and anywhere a fast keyed hash is wanted.
///
/// **Relationships.** `digest<N>` and `hash_input` (`hash/digest.hpp`); `blake3` (`hash/blake3.hpp`), its faster
/// successor; `hmac` (`hash/hmac.hpp`) is unnecessary with BLAKE2's own key but works. Group: \ref core.
///
/// **Implementations / variants.** `blake2_hash<Traits, N>` is the one class; `blake2b<N>` and `blake2s<N>` its
/// alias templates, and `blake2b_256`, `blake2b_512`, `blake2s_256` the common lengths.
///
/// **Guarantees.** Every member is `constexpr`; the unkeyed constructor, `update`, `finish` and `hash` are
/// `noexcept`. No allocation; a hasher is a trivially copyable value. Complexity: one compression (12 rounds for
/// BLAKE2b, 10 for BLAKE2s) per block. The last block is held back until `finish`, as the algorithm requires.
///
/// \pre `1 <= N <= 64` (BLAKE2b) or `<= 32` (BLAKE2s): a compile error otherwise.
///
/// **Errors.** The keyed constructor throws `std::length_error` for a key longer than 64 (BLAKE2b) or 32
/// (BLAKE2s) bytes; in a constant expression that is a compile error.
///
/// **Examples.** Above; `demos/hash/blake2_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/blake2_benchmark.cpp`: BLAKE2b-512 and BLAKE2s-256 against a hand-written
/// loop and OpenSSL (when installed), after a fairness check; figures in BENCHMARKS.md. Portable code only: the
/// AVX2 and NEON versions of libb2 or libsodium are faster.
///
/// **Tests and demos.** `tests/hash/blake2_test.cpp` (RFC 7693 Appendices A and B, the reference's keyed vectors,
/// Python's `hashlib` keyed and unkeyed on random lengths and output lengths, every split),
/// `demos/hash/blake2_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<stdexcept>`, `hash/digest.hpp`.
///
/// **Known issues.** The salt and personalisation fields of the BLAKE2 paper's parameter block (which RFC 7693
/// leaves out) are not provided, nor the tree modes or BLAKE2bp / BLAKE2sp.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_blake2_hpp
#define snicholls_core_hash_blake2_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/hash/digest.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace snicholls {

namespace detail {

    inline constexpr std::uint8_t blake2_sigma[12][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
        {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
        {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
        {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
        {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

    struct blake2b_traits
    {
        using word = std::uint64_t;
        static constexpr std::size_t block_size = 128;
        static constexpr std::size_t max_size = 64;
        static constexpr std::size_t rounds = 12;
        static constexpr unsigned r1 = 32, r2 = 24, r3 = 16, r4 = 63;
        static constexpr word iv[8] = {0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                                       0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};
    };

    struct blake2s_traits
    {
        using word = std::uint32_t;
        static constexpr std::size_t block_size = 64;
        static constexpr std::size_t max_size = 32;
        static constexpr std::size_t rounds = 10;
        static constexpr unsigned r1 = 16, r2 = 12, r3 = 8, r4 = 7;
        static constexpr word iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    };

    // RFC 7693 3.2, F: the compression of one block at byte offset (t_low, t_high), `last` for the final block.
    template <typename T, typename U>
    constexpr void blake2_compress(std::array<typename T::word, 8>& h, const U* block, typename T::word t_low,
                                   typename T::word t_high, bool last) noexcept
    {
        using W = typename T::word;
        W m[16]{};
        for (std::size_t i = 0; i < 16; ++i)
            m[i] = load_le_word<W>(block + i * sizeof(W));
        W v[16]{};
        for (std::size_t i = 0; i < 8; ++i) {
            v[i] = h[i];
            v[i + 8] = T::iv[i];
        }
        v[12] ^= t_low;
        v[13] ^= t_high;
        if (last)
            v[14] = static_cast<W>(~v[14]);
        const auto g = [&v](std::size_t a, std::size_t b, std::size_t c, std::size_t d, W x, W y) {
            v[a] = static_cast<W>(v[a] + v[b] + x);
            v[d] = rotr<W>(v[d] ^ v[a], T::r1);
            v[c] = static_cast<W>(v[c] + v[d]);
            v[b] = rotr<W>(v[b] ^ v[c], T::r2);
            v[a] = static_cast<W>(v[a] + v[b] + y);
            v[d] = rotr<W>(v[d] ^ v[a], T::r3);
            v[c] = static_cast<W>(v[c] + v[d]);
            v[b] = rotr<W>(v[b] ^ v[c], T::r4);
        };
        for (std::size_t r = 0; r < T::rounds; ++r) {
            const std::uint8_t* s = blake2_sigma[r];
            g(0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for (std::size_t i = 0; i < 8; ++i)
            h[i] ^= v[i] ^ v[i + 8];
    }

} // namespace detail

/// \brief BLAKE2 (RFC 7693) with an `N`-byte output: the class behind `blake2b<N>` and `blake2s<N>`.
/// \ingroup core
/// \tparam Traits The variant (internal).
/// \tparam N The output length in bytes.
template <typename Traits, std::size_t N>
class blake2_hash : public detail::hash_updates<blake2_hash<Traits, N>>, public detail::one_shot_hash<blake2_hash<Traits, N>>
{
    static_assert(N >= 1 && N <= Traits::max_size, "snicholls::blake2: the output length is 1 to 64 bytes (BLAKE2b) or 32 (BLAKE2s)");
    using word = typename Traits::word;

public:
    /// \brief The block size in bytes (128 for BLAKE2b, 64 for BLAKE2s).
    static constexpr std::size_t block_size = Traits::block_size;
    /// \brief The digest size in bytes, `N`.
    static constexpr std::size_t digest_size = N;
    /// \brief The longest key in bytes (64 for BLAKE2b, 32 for BLAKE2s).
    static constexpr std::size_t max_key_size = Traits::max_size;
    /// \brief The digest type.
    using digest_type = digest<N>;

    /// \brief An unkeyed hasher.
    constexpr blake2_hash() noexcept { init(0); }

    /// \brief A keyed hasher (RFC 7693 2.9): a MAC, or a hash personalised by a secret.
    /// \param key The key, 0 to `max_key_size` bytes; an empty key is the unkeyed hash.
    /// \throws std::length_error if the key is longer than `max_key_size`.
    template <hash_input K>
    constexpr explicit blake2_hash(const K& key)
    {
        const std::size_t n = detail::input_size(key);
        if (n > max_key_size)
            throw std::length_error("snicholls::blake2: the key is longer than the variant allows");
        init(n);
        if (n != 0) {
            const auto* k = detail::input_data(key);
            for (std::size_t i = 0; i < n; ++i)
                buf_[i] = detail::octet_of(k[i]);
            fill_ = block_size;   // the key block is the first block, held back like any other
        }
    }

    /// \brief One-shot keyed hash: `blake2_hash{key}.update(input).finish()`.
    /// \param key The key, 0 to `max_key_size` bytes.
    /// \param input The input.
    /// \return The digest.
    /// \throws std::length_error if the key is too long.
    template <hash_input K, hash_input S>
    [[nodiscard]] static constexpr digest_type keyed_hash(const K& key, const S& input)
    {
        blake2_hash h{key};
        h.update(input);
        return h.finish();
    }

    /// \brief Compresses the last block and returns the digest. The hasher is not used afterwards.
    /// \return The digest.
    [[nodiscard]] constexpr digest_type finish() noexcept
    {
        count(fill_);
        for (std::size_t i = fill_; i < block_size; ++i)
            buf_[i] = 0;
        detail::blake2_compress<Traits>(h_, buf_.data(), t_low_, t_high_, true);
        std::array<std::uint8_t, 8 * sizeof(word)> bytes{};
        for (std::size_t i = 0; i < 8; ++i)
            detail::store_le_word(bytes.data() + i * sizeof(word), h_[i]);
        digest_type out{};
        for (std::size_t i = 0; i < N; ++i)
            out.bytes[i] = static_cast<std::byte>(bytes[i]);
        return out;
    }

private:
    friend detail::hash_updates<blake2_hash>;

    constexpr void init(std::size_t key_size) noexcept
    {
        for (std::size_t i = 0; i < 8; ++i)
            h_[i] = Traits::iv[i];
        h_[0] ^= static_cast<word>(0x01010000u ^ (key_size << 8) ^ N);
    }

    constexpr void count(std::size_t n) noexcept
    {
        t_low_ = static_cast<word>(t_low_ + n);
        if (t_low_ < n)
            ++t_high_;
    }

    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        while (n != 0) {
            if (fill_ == block_size) {   // more input follows, so the held block is not the last
                count(block_size);
                detail::blake2_compress<Traits>(h_, buf_.data(), t_low_, t_high_, false);
                fill_ = 0;
            }
            if (fill_ == 0 && n > block_size) {   // a whole block that is not the last: straight from the input
                count(block_size);
                detail::blake2_compress<Traits>(h_, p, t_low_, t_high_, false);
                p += block_size;
                n -= block_size;
                continue;
            }
            const std::size_t take = n < block_size - fill_ ? n : block_size - fill_;
            for (std::size_t i = 0; i < take; ++i)
                buf_[fill_ + i] = detail::octet_of(p[i]);
            fill_ += take;
            p += take;
            n -= take;
        }
    }

    std::array<word, 8> h_{};
    std::array<std::uint8_t, Traits::block_size> buf_{};
    std::size_t fill_ = 0;
    word t_low_ = 0, t_high_ = 0;
};

/// \brief BLAKE2b with an `N`-byte output, 1 to 64 (64-bit words, 128-byte blocks).
/// \ingroup core
template <std::size_t N = 64>
using blake2b = blake2_hash<detail::blake2b_traits, N>;

/// \brief BLAKE2s with an `N`-byte output, 1 to 32 (32-bit words, 64-byte blocks).
/// \ingroup core
template <std::size_t N = 32>
using blake2s = blake2_hash<detail::blake2s_traits, N>;

/// \brief BLAKE2b-256.
/// \ingroup core
using blake2b_256 = blake2b<32>;
/// \brief BLAKE2b-512.
/// \ingroup core
using blake2b_512 = blake2b<64>;
/// \brief BLAKE2s-256.
/// \ingroup core
using blake2s_256 = blake2s<32>;

} // namespace snicholls

#endif /* snicholls_core_hash_blake2_hpp */
