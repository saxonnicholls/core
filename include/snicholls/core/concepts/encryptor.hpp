//
//  encryptor.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of base-encode-decode's Encryptor (utils/encryption.hpp: encrypt and decrypt of self-contained blobs,
//  algorithm()), made AEAD-shaped: associated data, a key id in the envelope so keys rotate, and errors returned
//  through result instead of thrown, never carrying key material or plaintext. Ciphers stay in crypto (ChaCha20-Poly1305,
//  AES-GCM); core holds no cipher.
//

/// \file
/// \brief `snicholls::encryptor`: authenticated encryption with associated data, behind key ids.
/// \ingroup core
///
/// **Overview.** An encryptor seals bytes under a named key and opens what it sealed: `encrypt(key_id, plaintext,
/// associated_data)` returns a self-contained *sealed envelope*, and `decrypt(sealed, associated_data)` returns the
/// plaintext only when the envelope is authentic and the associated data is the same. The associated data is
/// authenticated, not encrypted: bind a ciphertext to its context (a row's key, a user id, a field name) so it cannot
/// be moved to another. Code that stores secrets (a key-value store's values, a column, a file) holds an `encryptor`
/// and never names an algorithm; serialise, encrypt, encode and store, and the reverse, without any layer knowing the
/// cipher (base-encode-decode's design).
///
/// **The envelope** is the contract's, so any implementation reads a key id written by another: byte 0 the format
/// (`sealed_format_v1`, 1), byte 1 the key id's length `n` (1 to 64), then the key id, then the implementation's own
/// nonce, ciphertext and tag. `sealed_key_id(sealed)` reads the key id without a key, to route an envelope to the
/// encryptor that holds it. A key id is 1 to 64 characters of `A-Z a-z 0-9 _ - . :` (`is_key_id`): never a secret.
///
/// **The contract** (`interfaces/encryptor_conformance.hpp` checks it):
/// - `decrypt(encrypt(k, p, a), a) == p`, for any `p` (the empty one included) and `a`;
/// - two encryptions of the same plaintext differ (a fresh nonce every time; the caller never manages nonces);
/// - any change to the envelope, or other associated data, is `capability_errc::signature_invalid`, and no plaintext
///   is returned: never unauthenticated data;
/// - an unknown key id is `capability_errc::not_found`; an envelope too short to read, or a key id that is not one,
///   is `capability_errc::invalid_argument`;
/// - an envelope sealed under an older key opens while that key is held (rotation adds keys);
/// - errors carry no key material and no plaintext (`capability_error.hpp`'s redaction rule: the message is a literal,
///   and a provider code is the library's short code at most).
///
/// \snippet concepts/encryptor_demo.cpp concept
///
/// **Relationships.** `capability_error.hpp`; the twin `interfaces/encryptor.hpp`, the test double
/// `interfaces/encryptor_reference.hpp`, the kit `interfaces/encryptor_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `insecure_test_encryptor` (**not encryption**: a test double that exercises the contract and
/// protects nothing). Above core, in crypto: ChaCha20-Poly1305 and AES-256-GCM over vetted libraries.
///
/// **Guarantees.** `is_key_id` is `constexpr`; it and `sealed_key_id` are `noexcept` and allocation-free. Thread safety is the
/// implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked: the cost is the cipher's.
///
/// **Tests and demos.** `tests/concepts/encryptor_test.cpp`, `demos/concepts/encryptor_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<span>`, `<string_view>`, `<vector>`, `capability_error.hpp`.
///
/// **Known issues.** The plaintext comes back in a `std::vector<std::byte>`: wipe it with `secure_wipe`
/// (`hash/secure.hpp`) when it held a secret.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_encryptor_hpp
#define snicholls_core_concepts_encryptor_hpp

#include <snicholls/core/capability_error.hpp>

#include <concepts>
#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief The format byte of the sealed envelope this contract defines.
    /// \ingroup core
    inline constexpr std::byte sealed_format_v1{1};

    /// \brief The longest key id.
    /// \ingroup core
    inline constexpr std::size_t max_key_id_length = 64;

    /// \brief Whether `id` is a key id: 1 to 64 characters of `A-Z a-z 0-9 _ - . :`.
    /// \ingroup core
    /// \param id The text.
    /// \return `true` when it is.
    [[nodiscard]] constexpr bool is_key_id(std::string_view id) noexcept
    {
        if (id.empty() || id.size() > max_key_id_length)
            return false;
        for (const char c : id)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                  c == '.' || c == ':'))
                return false;
        return true;
    }

    /// \brief The key id of a sealed envelope, read without a key.
    /// \ingroup core
    /// \param sealed The envelope.
    /// \return The key id (a view into `sealed`), or `invalid_argument` when the envelope is not one.
    [[nodiscard]] inline capability_result<std::string_view> sealed_key_id(std::span<const std::byte> sealed) noexcept
    {
        if (sealed.size() < 2 || sealed[0] != sealed_format_v1)
            return fail(capability_errc::invalid_argument, {}, "not a sealed envelope");
        const auto n = static_cast<std::size_t>(sealed[1]);
        if (n == 0 || n > max_key_id_length || sealed.size() < 2 + n)
            return fail(capability_errc::invalid_argument, {}, "not a sealed envelope");
        char text[max_key_id_length]{};
        for (std::size_t i = 0; i < n; ++i)
            text[i] = static_cast<char>(sealed[2 + i]);
        if (!is_key_id({text, n}))
            return fail(capability_errc::invalid_argument, {}, "the envelope's key id is not one");
        // the id is ASCII, so the envelope's bytes are its characters
        return ok(std::string_view(reinterpret_cast<const char*>(sealed.data() + 2), n));
    }

    /// \brief The envelope's header for `key_id`: the format, the length and the id, which an implementation writes
    /// first.
    /// \ingroup core
    /// \param key_id A key id (`is_key_id`).
    /// \return The header bytes.
    [[nodiscard]] inline std::vector<std::byte> sealed_header(std::string_view key_id)
    {
        std::vector<std::byte> out(2 + key_id.size());   // sized once: GCC 15 misreads push_back after reserve
        out[0] = sealed_format_v1;
        out[1] = static_cast<std::byte>(key_id.size());
        for (std::size_t i = 0; i < key_id.size(); ++i)
            out[2 + i] = static_cast<std::byte>(key_id[i]);
        return out;
    }

    /// \brief An encryptor: `e.encrypt(key_id, plaintext, aad)`, `ce.decrypt(sealed, aad)` and `ce.algorithm()`.
    /// \ingroup core
    template <typename E>
    concept encryptor = requires(E& e, const E& ce, std::string_view key_id, std::span<const std::byte> bytes) {
        { e.encrypt(key_id, bytes, bytes) } -> std::same_as<capability_result<std::vector<std::byte>>>;
        { ce.decrypt(bytes, bytes) } -> std::same_as<capability_result<std::vector<std::byte>>>;
        { ce.algorithm() } -> std::same_as<std::string_view>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_encryptor_hpp
