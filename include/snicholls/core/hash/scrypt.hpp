//
//  scrypt.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief scrypt (RFC 7914), the memory-hard password KDF, in working memory the caller supplies.
/// \ingroup core
///
/// **Overview.** scrypt makes each password guess cost memory as well as time: ROMix fills `N` blocks of `128 r`
/// bytes and then reads them back in an order that depends on their contents, so an attacker who keeps less memory
/// pays in recomputation. Use it for a new password store or a wallet key (Ethereum's keystore v3, Litecoin's
/// proof of work, many vaults use it); `N = 2^17, r = 8, p = 1` (128 MiB) is the 2026 interactive-login setting
/// of OWASP's guidance, `N = 2^20` for a file encrypted once.
///
/// Core never allocates, so the `128 r (N + p + 1)` bytes of working memory are the caller's: ask
/// `scrypt_work_words(params)` for the size, provide that many `std::uint32_t` (a `std::vector`, a buffer
/// reused between calls, an `mmap`), and `scrypt` fills `out`:
///
/// \snippet hash/scrypt_demo.cpp scrypt
///
/// **Relationships.** `pbkdf2_hmac<sha256>` and `kdf_error` (`hash/kdf.hpp`), which scrypt calls at its start and
/// end; `secure_wipe` (`hash/secure.hpp`), which clears the working memory before returning; `result`
/// (`result.hpp`). Group: \ref core.
///
/// **Guarantees.** `constexpr` and `noexcept`; no allocation. The working memory is wiped (`secure_wipe`) before
/// `scrypt` returns, on success; on an error nothing is written to `out` or `work`. Complexity: `2 N p` BlockMix
/// operations of `2 r` Salsa20/8 cores each, and `N` blocks of memory; the parameters are the caller's to choose
/// for the time and memory they can afford.
///
/// \pre `work` and `out` do not overlap.
///
/// **Errors.** `kdf_error::invalid_cost` unless `N` is a power of two, greater than 1 and below `2^(16 r)`
/// (RFC 7914 2), or when the working memory's size overflows `std::size_t`; `kdf_error::invalid_block_parameters`
/// when `r` or `p` is 0 or `r * p >= 2^30`; `kdf_error::insufficient_work_memory` when `work` is shorter than
/// `scrypt_work_words(params)`; `kdf_error::empty_output` / `output_too_long` as PBKDF2's.
///
/// **Examples.** Above; `demos/hash/scrypt_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/scrypt_benchmark.cpp`: against a hand-written loop over the same working
/// memory and OpenSSL's `EVP_PBE_scrypt` (when installed), after a fairness check; figures in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/hash/scrypt_test.cpp` (RFC 7914's Salsa20/8, BlockMix-free ROMix and scrypt
/// vectors, the `N = 2^20` one only in Release; Python's `hashlib.scrypt` on random inputs; every error; the wipe;
/// a compile-time derivation), `demos/hash/scrypt_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `hash/kdf.hpp`, `hash/sha2.hpp`.
///
/// **Known issues.** `p` lanes run one after another: scrypt's `p` is meant for parallelism, which core leaves to
/// the caller (run `p = 1` derivations on several threads, or a library above core). The working memory holds
/// password-derived data until it is wiped; memory that is swapped out before then can reach a disk.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_scrypt_hpp
#define snicholls_core_hash_scrypt_hpp

#include <snicholls/core/hash/kdf.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls {

/// \brief scrypt's parameters: the cost `n` (a power of two), the block size `r` and the parallelism `p`.
/// \ingroup core
struct scrypt_params
{
    std::uint64_t n = 1u << 14;   ///< CPU and memory cost: a power of two greater than 1, below 2^(16 r)
    std::uint32_t r = 8;          ///< Block size: each block is 128 r bytes
    std::uint32_t p = 1;          ///< Parallelism: independent ROMix lanes
};

/// \brief The working memory `scrypt` needs, in 32-bit words: `32 r (N + p + 1)`.
/// \ingroup core
/// \param params The parameters.
/// \return The word count, or `kdf_error::invalid_cost` / `kdf_error::invalid_block_parameters` for parameters
///         RFC 7914 does not allow, or `kdf_error::invalid_cost` when the size does not fit in `std::size_t`.
[[nodiscard]] constexpr result<std::size_t, kdf_error> scrypt_work_words(const scrypt_params& params) noexcept
{
    if (params.r == 0 || params.p == 0 || std::uint64_t{params.r} * params.p >= (std::uint64_t{1} << 30))
        return err(kdf_error::invalid_block_parameters);
    if (params.n < 2 || (params.n & (params.n - 1)) != 0)
        return err(kdf_error::invalid_cost);
    if (params.r < 4 && params.n >= (std::uint64_t{1} << (16 * params.r)))   // N < 2^(128 r / 8); always true for r >= 4
        return err(kdf_error::invalid_cost);
    const std::size_t max = static_cast<std::size_t>(-1);
    const std::uint64_t blocks = params.n + params.p + 1;   // no overflow: n <= 2^63, p < 2^30
    const std::uint64_t block_words = 32ull * params.r;
    if (blocks > max / block_words)
        return err(kdf_error::invalid_cost);
    return ok(static_cast<std::size_t>(blocks * block_words));
}

namespace detail {

    // RFC 7914 3: Salsa20/8's core, out = in + 8 rounds of in.
    constexpr void salsa20_8(std::uint32_t (&b)[16]) noexcept
    {
        std::uint32_t x[16]{};
        for (std::size_t i = 0; i < 16; ++i)
            x[i] = b[i];
        const auto quarter = [&x](std::size_t a, std::size_t bb, std::size_t c, std::size_t d) {
            x[bb] ^= rotl<std::uint32_t>(x[a] + x[d], 7);
            x[c] ^= rotl<std::uint32_t>(x[bb] + x[a], 9);
            x[d] ^= rotl<std::uint32_t>(x[c] + x[bb], 13);
            x[a] ^= rotl<std::uint32_t>(x[d] + x[c], 18);
        };
        for (int round = 0; round < 8; round += 2) {
            quarter(0, 4, 8, 12);
            quarter(5, 9, 13, 1);
            quarter(10, 14, 2, 6);
            quarter(15, 3, 7, 11);
            quarter(0, 1, 2, 3);
            quarter(5, 6, 7, 4);
            quarter(10, 11, 8, 9);
            quarter(15, 12, 13, 14);
        }
        for (std::size_t i = 0; i < 16; ++i)
            b[i] += x[i];
    }

    // RFC 7914 4: BlockMix over 2r 16-word blocks of b, through y (32r words), the result back in b.
    constexpr void scrypt_block_mix(std::uint32_t* b, std::uint32_t* y, std::size_t r) noexcept
    {
        std::uint32_t t[16]{};
        for (std::size_t k = 0; k < 16; ++k)
            t[k] = b[(2 * r - 1) * 16 + k];
        for (std::size_t i = 0; i < 2 * r; ++i) {
            for (std::size_t k = 0; k < 16; ++k)
                t[k] ^= b[i * 16 + k];
            salsa20_8(t);
            const std::size_t to = (i % 2 == 0 ? i / 2 : r + i / 2) * 16;   // evens first, then odds
            for (std::size_t k = 0; k < 16; ++k)
                y[to + k] = t[k];
        }
        for (std::size_t k = 0; k < 32 * r; ++k)
            b[k] = y[k];
    }

    // RFC 7914 5: ROMix of x (32r words) through v (N blocks) and y.
    constexpr void scrypt_ro_mix(std::uint32_t* x, std::uint32_t* v, std::uint32_t* y, std::size_t r, std::uint64_t n) noexcept
    {
        const std::size_t words = 32 * r;
        for (std::uint64_t i = 0; i < n; ++i) {
            std::uint32_t* vi = v + static_cast<std::size_t>(i) * words;
            for (std::size_t k = 0; k < words; ++k)
                vi[k] = x[k];
            scrypt_block_mix(x, y, r);
        }
        for (std::uint64_t i = 0; i < n; ++i) {
            const std::size_t last = (2 * r - 1) * 16;
            const std::uint64_t j = (std::uint64_t{x[last]} | (std::uint64_t{x[last + 1]} << 32)) & (n - 1);
            const std::uint32_t* vj = v + static_cast<std::size_t>(j) * words;
            for (std::size_t k = 0; k < words; ++k)
                x[k] ^= vj[k];
            scrypt_block_mix(x, y, r);
        }
    }

} // namespace detail

/// \brief scrypt (RFC 7914): derives `out.size()` bytes from `password` and `salt`, in the caller's memory.
/// \ingroup core
/// \param password The password.
/// \param salt The salt: unique per password, 16 bytes or more of randomness.
/// \param params N, r and p.
/// \param work At least `scrypt_work_words(params)` words of working memory; wiped before returning.
/// \param out Where to write the derived key.
/// \return Success, or a `kdf_error` (see Errors) with nothing written.
template <hash_input P, hash_input S>
constexpr result<void, kdf_error> scrypt(const P& password, const S& salt, const scrypt_params& params,
                                         std::span<std::uint32_t> work, std::span<std::byte> out) noexcept
{
    const auto needed = scrypt_work_words(params);
    if (!needed)
        return err(needed.unwrap_err_unchecked());
    if (work.size() < needed.unwrap_unchecked())
        return err(kdf_error::insufficient_work_memory);
    if (out.empty())
        return err(kdf_error::empty_output);
    if (out.size() / sha256::digest_size >= 0xffffffffull)
        return err(kdf_error::output_too_long);

    const std::size_t r = params.r, words = 32 * r;
    std::uint32_t* b = work.data();                       // p lanes of 32r words
    std::uint32_t* v = b + params.p * words;              // N blocks
    std::uint32_t* y = v + static_cast<std::size_t>(params.n) * words;   // BlockMix's scratch

    // B = PBKDF2-HMAC-SHA256(P, S, 1, 128 r p), as little-endian words
    detail::pbkdf2_blocks<sha256>(password, [&salt](hmac<sha256>& m) { m.update(salt); }, 1, 128 * r * params.p,
                                  [b](std::size_t offset, const digest<32>& t, std::size_t count) {
                                      for (std::size_t j = 0; j < count; j += 4)
                                          b[(offset + j) / 4] = detail::load_le_word<std::uint32_t>(t.bytes.data() + j);
                                  });
    for (std::size_t lane = 0; lane < params.p; ++lane)
        detail::scrypt_ro_mix(b + lane * words, v, y, r, params.n);

    // DK = PBKDF2-HMAC-SHA256(P, B, 1, dkLen), B fed back as bytes 64 at a time
    const auto feed_b = [b, total = params.p * words](hmac<sha256>& m) {
        std::array<std::uint8_t, 64> chunk{};
        for (std::size_t w = 0; w < total; w += 16) {
            for (std::size_t k = 0; k < 16; ++k)
                detail::store_le_word(chunk.data() + 4 * k, b[w + k]);
            m.update(chunk);
        }
        secure_wipe(std::span<std::uint8_t, 64>(chunk));
    };
    detail::pbkdf2_blocks<sha256>(password, feed_b, 1, out.size(),
                                  [&out](std::size_t offset, const digest<32>& t, std::size_t count) {
                                      for (std::size_t j = 0; j < count; ++j)
                                          out[offset + j] = t.bytes[j];
                                  });
    secure_wipe(work.first(needed.unwrap_unchecked()));
    return ok();
}

} // namespace snicholls

#endif /* snicholls_core_hash_scrypt_hpp */
