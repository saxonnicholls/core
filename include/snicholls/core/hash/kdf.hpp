//
//  kdf.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief HKDF (RFC 5869) and PBKDF2-HMAC (RFC 8018) over any hash in core, writing into the caller's memory.
/// \ingroup core
///
/// **Overview.** Two key derivation functions, for two different jobs.
///
/// - **HKDF** turns key material that is already strong but not uniform (a Diffie-Hellman shared secret, a master
///   key) into as many independent keys as a protocol needs: `hkdf_extract` concentrates the material into a
///   pseudorandom key, `hkdf_expand` stretches that into output bound to an `info` label, and `hkdf` does both.
///   TLS 1.3, Signal, Noise, HPKE and MLS use it. It is fast: never use it on a password.
/// - **PBKDF2-HMAC** stretches a *password* by iterating HMAC, so that each guess costs an attacker the same
///   `iterations` HMACs it costs you. BIP-39 seeds are PBKDF2-HMAC-SHA512 with 2048 iterations; OWASP's 2023
///   guidance is 600,000 iterations of SHA-256 or 210,000 of SHA-512. Prefer `scrypt` (`hash/scrypt.hpp`) for a new
///   password store, since PBKDF2 costs a GPU little memory.
///
/// \snippet hash/kdf_demo.cpp kdf
///
/// Output goes into a span the caller owns, so nothing allocates; the `<H, L>` forms return a `digest<L>` and are
/// usable in a constant expression. Expected failures (an output too long for the function, zero iterations) are
/// a `result<void, kdf_error>`, never an exception.
///
/// **Relationships.** `hmac<H>` (`hash/hmac.hpp`); `hash_function` and `digest<N>` (`hash/digest.hpp`);
/// `secure_wipe` (`hash/secure.hpp`); `scrypt`, which uses `pbkdf2_hmac<sha256>`; `blake3::derive_key`
/// (`hash/blake3.hpp`), a faster key derivation where HKDF is not required. Group: \ref core.
///
/// **Guarantees.** Every function is `constexpr` and `noexcept`; no allocation. Intermediate blocks are wiped
/// with `secure_wipe`. On an error nothing is written. Complexity: HKDF, `ceil(L / HashLen)` HMACs; PBKDF2,
/// `ceil(L / HashLen) * iterations` HMACs, two compressions each.
///
/// **Errors.** `kdf_error::output_too_long` when HKDF is asked for more than `255 * HashLen` bytes or PBKDF2 for
/// more than `(2^32 - 1) * HashLen`; `kdf_error::empty_output` for an empty output span; and from PBKDF2,
/// `kdf_error::zero_iterations`.
///
/// **Examples.** Above; `demos/hash/kdf_demo.cpp`.
///
/// **Performance.** `benchmarks/hash/kdf_benchmark.cpp`: PBKDF2-HMAC-SHA256 and -SHA512 against a hand-written
/// loop and OpenSSL's `PKCS5_PBKDF2_HMAC` (when installed), after a fairness check; figures in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/hash/kdf_test.cpp` (RFC 5869's test cases 1 to 7, RFC 6070's PBKDF2-HMAC-SHA1
/// vectors, RFC 7914's PBKDF2-HMAC-SHA256 vectors, Python's `hashlib.pbkdf2_hmac` on random inputs, the BIP-39
/// test seed, errors), `demos/hash/kdf_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `hash/hmac.hpp`, `result.hpp` (larger than
/// it looks: about 2,500 lines).
///
/// **Known issues.** RFC 6070's 16,777,216-iteration vector is run by the tests only in Release builds.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_kdf_hpp
#define snicholls_core_hash_kdf_hpp

#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls {

/// \brief Why a key derivation refused to run.
/// \ingroup core
enum class kdf_error {
    output_too_long,           ///< More output than the function can produce
    empty_output,              ///< An empty output span
    zero_iterations,           ///< PBKDF2 with no iterations
    invalid_cost,              ///< scrypt: N not a power of two greater than 1, or not below 2^(16r)
    invalid_block_parameters,  ///< scrypt: r or p zero, or r * p not below 2^30
    insufficient_work_memory,  ///< scrypt: the working memory is smaller than `scrypt_work_words`
};

/// \brief HKDF-Extract (RFC 5869 2.2): the pseudorandom key `HMAC-H(salt, ikm)`.
/// \ingroup core
/// \tparam H The hash.
/// \param salt The salt; empty means `HashLen` zero bytes, which HMAC makes the same thing.
/// \param ikm The input key material.
/// \return The pseudorandom key.
template <hash_function H, hash_input S, hash_input K>
[[nodiscard]] constexpr digest_of_t<H> hkdf_extract(const S& salt, const K& ikm) noexcept
{
    return hmac<H>::mac(salt, ikm);
}

/// \brief HKDF-Expand (RFC 5869 2.3): `out.size()` bytes of output keying material from `prk` and `info`.
/// \ingroup core
/// \tparam H The hash.
/// \param prk The pseudorandom key (from `hkdf_extract`, or a uniformly random key of at least `HashLen` bytes).
/// \param info The context and application label; may be empty.
/// \param out Where to write, 1 to `255 * HashLen` bytes.
/// \return Success, or `kdf_error::output_too_long` / `kdf_error::empty_output` with nothing written.
template <hash_function H, hash_input P, hash_input I>
constexpr result<void, kdf_error> hkdf_expand(const P& prk, const I& info, std::span<std::byte> out) noexcept
{
    if (out.empty())
        return err(kdf_error::empty_output);
    if (out.size() > 255 * H::digest_size)
        return err(kdf_error::output_too_long);
    const hmac<H> keyed{prk};
    digest_of_t<H> t{};
    std::size_t done = 0;
    for (std::uint8_t counter = 1; done < out.size(); ++counter) {
        hmac<H> m = keyed;
        if (counter > 1)
            m.update(t);
        m.update(info);
        m.update(std::array<std::uint8_t, 1>{counter});
        t = m.finish();
        for (std::size_t i = 0; i < H::digest_size && done < out.size(); ++i)
            out[done++] = t.bytes[i];
    }
    secure_wipe(t);
    return ok();
}

/// \brief HKDF-Expand to a fixed length, checked at compile time.
/// \ingroup core
/// \tparam H The hash.
/// \tparam L The output length, 1 to `255 * HashLen`.
/// \param prk The pseudorandom key.
/// \param info The context and application label.
/// \return The output keying material.
template <hash_function H, std::size_t L, hash_input P, hash_input I>
[[nodiscard]] constexpr digest<L> hkdf_expand(const P& prk, const I& info) noexcept
{
    static_assert(L >= 1 && L <= 255 * H::digest_size, "snicholls::hkdf_expand: L is 1 to 255 * HashLen bytes");
    digest<L> out{};
    (void)hkdf_expand<H>(prk, info, std::span<std::byte>(out.bytes));
    return out;
}

/// \brief HKDF (RFC 5869): extract, then expand into `out`.
/// \ingroup core
/// \tparam H The hash.
/// \param salt The salt (may be empty).
/// \param ikm The input key material.
/// \param info The context and application label (may be empty).
/// \param out Where to write, 1 to `255 * HashLen` bytes.
/// \return Success, or the error of `hkdf_expand`.
template <hash_function H, hash_input S, hash_input K, hash_input I>
constexpr result<void, kdf_error> hkdf(const S& salt, const K& ikm, const I& info, std::span<std::byte> out) noexcept
{
    digest_of_t<H> prk = hkdf_extract<H>(salt, ikm);
    auto r = hkdf_expand<H>(prk, info, out);
    secure_wipe(prk);
    return r;
}

/// \brief HKDF (RFC 5869) to a fixed length, checked at compile time.
/// \ingroup core
/// \tparam H The hash.
/// \tparam L The output length, 1 to `255 * HashLen`.
/// \param salt The salt (may be empty).
/// \param ikm The input key material.
/// \param info The context and application label (may be empty).
/// \return The output keying material.
template <hash_function H, std::size_t L, hash_input S, hash_input K, hash_input I>
[[nodiscard]] constexpr digest<L> hkdf(const S& salt, const K& ikm, const I& info) noexcept
{
    digest_of_t<H> prk = hkdf_extract<H>(salt, ikm);
    digest<L> out = hkdf_expand<H, L>(prk, info);
    secure_wipe(prk);
    return out;
}

namespace detail {

    // PBKDF2's blocks (RFC 8018 5.2): T_i for i = 1, 2, ... until `length` bytes; `salt(m)` feeds the salt into an
    // HMAC, `emit(offset, T_i, count)` takes the first `count` bytes of T_i for output offset `offset`.
    template <hash_function H, hash_input P, typename Salt, typename Emit>
    constexpr void pbkdf2_blocks(const P& password, Salt&& salt, std::uint32_t iterations, std::size_t length, Emit&& emit) noexcept
    {
        const hmac<H> keyed{password};
        digest_of_t<H> u{}, t{};
        std::size_t done = 0;
        for (std::uint32_t block = 1; done < length; ++block) {
            hmac<H> m = keyed;
            salt(m);
            const std::array<std::uint8_t, 4> index{static_cast<std::uint8_t>(block >> 24), static_cast<std::uint8_t>(block >> 16),
                                                    static_cast<std::uint8_t>(block >> 8), static_cast<std::uint8_t>(block)};
            m.update(index);
            u = m.finish();
            t = u;
            for (std::uint32_t i = 1; i < iterations; ++i) {
                hmac<H> next = keyed;
                next.update(u);
                u = next.finish();
                for (std::size_t j = 0; j < H::digest_size; ++j)
                    t.bytes[j] ^= u.bytes[j];
            }
            const std::size_t count = length - done < H::digest_size ? length - done : H::digest_size;
            emit(done, t, count);
            done += count;
        }
        secure_wipe(u);
        secure_wipe(t);
    }

} // namespace detail

/// \brief PBKDF2 with HMAC-H as its pseudorandom function (RFC 8018 5.2), into `out`.
/// \ingroup core
/// \tparam H The hash: `sha256` or `sha512` for anything new; `sha1` where a protocol fixes it.
/// \param password The password (any length; one longer than H's block is hashed by HMAC).
/// \param salt The salt: unique per password, 16 bytes or more of randomness.
/// \param iterations The iteration count, at least 1.
/// \param out Where to write the derived key.
/// \return Success, or `kdf_error::zero_iterations` / `kdf_error::empty_output` / `kdf_error::output_too_long`
///         with nothing written.
template <hash_function H, hash_input P, hash_input S>
constexpr result<void, kdf_error> pbkdf2_hmac(const P& password, const S& salt, std::uint32_t iterations,
                                              std::span<std::byte> out) noexcept
{
    if (iterations == 0)
        return err(kdf_error::zero_iterations);
    if (out.empty())
        return err(kdf_error::empty_output);
    if (out.size() / H::digest_size >= 0xffffffffull)
        return err(kdf_error::output_too_long);
    detail::pbkdf2_blocks<H>(password, [&salt](hmac<H>& m) { m.update(salt); }, iterations, out.size(),
                             [&out](std::size_t offset, const digest_of_t<H>& t, std::size_t count) {
                                 for (std::size_t j = 0; j < count; ++j)
                                     out[offset + j] = t.bytes[j];
                             });
    return ok();
}

/// \brief PBKDF2-HMAC-H to a fixed length.
/// \ingroup core
/// \tparam H The hash.
/// \tparam L The derived key's length in bytes.
/// \param password The password.
/// \param salt The salt.
/// \param iterations The iteration count, at least 1.
/// \return The derived key, or `kdf_error::zero_iterations`.
template <hash_function H, std::size_t L, hash_input P, hash_input S>
[[nodiscard]] constexpr result<digest<L>, kdf_error> pbkdf2_hmac(const P& password, const S& salt, std::uint32_t iterations) noexcept
{
    static_assert(L >= 1, "snicholls::pbkdf2_hmac: L is at least 1 byte");
    digest<L> out{};
    if (auto r = pbkdf2_hmac<H>(password, salt, iterations, std::span<std::byte>(out.bytes)); !r)
        return err(r.unwrap_err_unchecked());
    return ok(out);
}

} // namespace snicholls

#endif /* snicholls_core_hash_kdf_hpp */
