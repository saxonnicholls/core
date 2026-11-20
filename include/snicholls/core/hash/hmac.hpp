//
//  hmac.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief HMAC (RFC 2104, FIPS 198-1) over any hash in core, usable at compile time.
/// \ingroup core
///
/// **Overview.** `hmac<H>` is the keyed message authentication code built from a hash `H`:
/// `H((K ^ opad) || H((K ^ ipad) || message))`, with a key longer than `H`'s block first hashed. It is the MAC to
/// reach for when a protocol names one (JWT's HS256, AWS Signature V4, TOTP, webhooks' signatures, BIP-32's
/// `HMAC-SHA512("Bitcoin seed", seed)`) and the building block of `hkdf_*` and `pbkdf2_hmac` (`hash/kdf.hpp`).
/// The incremental interface of the hashes (`update`, `finish`), a one-shot `mac`, and `verify`, which compares in
/// constant time.
///
/// \snippet hash/hmac_demo.cpp hmac
///
/// `H` is any `hash_function`: `sha256`, `sha512`, `sha1` (HMAC-SHA1 is still sound: HMAC does not need collision
/// resistance), `sha3_256` (block size the rate, as FIPS 202 specifies), `blake2b<N>`. BLAKE2 and BLAKE3 have keyed
/// modes of their own that are cheaper than HMAC around them.
///
/// **Relationships.** `hash_function`, `digest<N>` (`hash/digest.hpp`); `constant_time_equal` and `secure_wipe`
/// (`hash/secure.hpp`); `hkdf_extract`, `hkdf_expand`, `pbkdf2_hmac` (`hash/kdf.hpp`), `scrypt`
/// (`hash/scrypt.hpp`). Group: \ref core.
///
/// **Guarantees.** Every member is `constexpr` and `noexcept`; no allocation. The padded key is wiped with
/// `secure_wipe` once the two inner states are primed; the object then holds two hash states and no key. An
/// `hmac<H>` is a copyable value: copying a keyed object and updating the copy reuses the key's two compressions,
/// which is how PBKDF2 runs at two compressions per iteration. `verify` takes time independent of where the tags
/// differ.
///
/// **Errors.** None. Any key length is accepted, including empty (RFC 2104).
///
/// **Examples.** Above; `demos/hash/hmac_demo.cpp`.
///
/// **Performance.** Two hash computations, and two extra compressions for the key, which a copied object does not
/// repeat. Benchmarked through PBKDF2 in `benchmarks/hash/kdf_benchmark.cpp`.
///
/// **Tests and demos.** `tests/hash/hmac_test.cpp` (RFC 4231's test cases 1 to 7 for SHA-224/256/384/512,
/// RFC 2202 for SHA-1, Python's `hmac` on random keys and lengths over SHA-1, SHA-2 and SHA-3, every split,
/// `verify`), `demos/hash/hmac_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `hash/digest.hpp`, `hash/secure.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_hmac_hpp
#define snicholls_core_hash_hmac_hpp

#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/secure.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace snicholls {

/// \brief HMAC over the hash `H`.
/// \ingroup core
/// \tparam H A `hash_function`.
template <hash_function H>
class hmac : public detail::hash_updates<hmac<H>>
{
public:
    /// \brief The hash.
    using hash_type = H;
    /// \brief The hash's block size: the key is padded (or hashed, then padded) to it.
    static constexpr std::size_t block_size = H::block_size;
    /// \brief The tag's size in bytes: the hash's digest size.
    static constexpr std::size_t digest_size = H::digest_size;
    /// \brief The tag type.
    using digest_type = digest<digest_size>;

    /// \brief An HMAC keyed by `key`.
    /// \param key The key, any length; one longer than `block_size` is hashed first (RFC 2104).
    template <hash_input K>
    constexpr explicit hmac(const K& key) noexcept
    {
        std::array<std::uint8_t, block_size> pad{};
        const std::size_t n = detail::input_size(key);
        if (n > block_size) {
            digest_type hashed = H::hash(key);
            for (std::size_t i = 0; i < digest_size; ++i)
                pad[i] = static_cast<std::uint8_t>(hashed.bytes[i]);
            secure_wipe(hashed);
        }
        else {
            const auto* k = detail::input_data(key);
            for (std::size_t i = 0; i < n; ++i)
                pad[i] = detail::octet_of(k[i]);
        }
        for (auto& b : pad)
            b ^= 0x36;
        inner_.update(pad);
        for (auto& b : pad)
            b ^= 0x36 ^ 0x5c;
        outer_.update(pad);
        secure_wipe(std::span<std::uint8_t, block_size>(pad));
    }

    /// \brief The tag of everything absorbed. The object is not used afterwards.
    /// \return The tag.
    [[nodiscard]] constexpr digest_type finish() noexcept
    {
        digest_type inner = inner_.finish();
        outer_.update(inner);
        return outer_.finish();
    }

    /// \brief One-shot: the tag of `message` under `key`.
    /// \param key The key.
    /// \param message The message.
    /// \return The tag.
    template <hash_input K, hash_input S>
    [[nodiscard]] static constexpr digest_type mac(const K& key, const S& message) noexcept
    {
        hmac m{key};
        m.update(message);
        return m.finish();
    }

    /// \brief Whether `tag` is the tag of `message` under `key`, compared in constant time.
    /// \param key The key.
    /// \param message The message.
    /// \param tag The tag to check: any `hash_input`, so a truncated tag (fewer bytes) is simply unequal.
    /// \return `true` when the tag is correct.
    template <hash_input K, hash_input S, hash_input T>
    [[nodiscard]] static constexpr bool verify(const K& key, const S& message, const T& tag) noexcept
    {
        return constant_time_equal(mac(key, message), tag);
    }

private:
    friend detail::hash_updates<hmac>;
    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        inner_.update(std::span<const U>(p, n));
    }

    H inner_{};
    H outer_{};
};

} // namespace snicholls

#endif /* snicholls_core_hash_hmac_hpp */
