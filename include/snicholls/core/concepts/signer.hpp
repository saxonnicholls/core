//
//  signer.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signer`: anything that signs with a key it holds by id and never hands out, and publishes each
/// key version's public key.
/// \ingroup core
///
/// **Overview.** A signer signs on request (JWT access tokens, SIWE challenges, EVM transactions over a prehashed
/// digest) with keys it holds by id, and never returns private key material. It publishes the public key of every
/// version not destroyed, for a JWKS and for anyone who verifies. This is the concept half of a pair; the pure
/// virtual half is `signer_interface` (`interfaces/signer.hpp`), with `signer_adapter<S>` and `signer_ref`. Keys are
/// created, rotated and retired through the separate `key_store` capability (`concepts/key_store.hpp`), because a
/// product may sign through a KMS it does not administer.
///
/// **The contract** (`interfaces/signer_conformance.hpp` checks it):
/// - `sign` uses the key's current version and names that version in the signature.
/// - An unknown key is `capability_errc::not_found`; a retired or destroyed key is `capability_errc::conflict`: it
///   may no longer sign.
/// - `payload_kind::digest` requires exactly 32 bytes (`invalid_argument` otherwise) and is refused with
///   `unsupported` by algorithms that hash for themselves (EdDSA, HS256).
/// - `public_key_of` answers for every version not destroyed, retired ones included (tokens they signed still
///   verify); a symmetric algorithm (HS256) has no public key: `unsupported`.
/// - The signature verifies under the published key with the named algorithm.
/// - Errors follow the redaction rule of `capability_error.hpp`: no key material, payload or provider response in
///   them.
///
/// \snippet concepts/signer_demo.cpp concept
///
/// **Relationships.** `capability_error.hpp`, `capability_ids.hpp`, `concepts/key_store.hpp`; the twin
/// `interfaces/signer.hpp`, the reference `interfaces/signer_reference.hpp`, the kit
/// `interfaces/signer_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `memory_signer` (`interfaces/signer_reference.hpp`, which is `memory_key_store`: HS256 over
/// core's `hmac<sha256>`, for tests only). Above core, through constant-time elliptic-curve code: a device signer, a
/// KMS server, AWS KMS. Core signs with no secret-bearing key of its own (CLAUDE.md "Never").
///
/// **Guarantees.** `jose_name` and `is_asymmetric` are `constexpr` and `noexcept`. Thread safety is the
/// implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** `benchmarks/interfaces/key_store_reference_benchmark.cpp` (the reference's `sign`).
///
/// **Tests and demos.** `tests/concepts/signer_test.cpp`, `demos/concepts/signer_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, `<vector>`,
/// `capability_error.hpp`, `capability_ids.hpp`.
///
/// **Known issues.** HS256 is in the enumeration for tests and the reference only; RS256 for verifying third-party
/// OIDC tokens only (no signer of ours signs it).
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_signer_hpp
#define snicholls_core_concepts_signer_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief A signature algorithm, as JOSE (RFC 7518, RFC 8037) names it.
    /// \ingroup core
    enum class signature_algorithm : std::uint8_t {
        es256 = 1,   ///< ECDSA P-256 with SHA-256
        eddsa = 2,   ///< Ed25519
        es256k = 3,  ///< ECDSA secp256k1 (EVM: Keccak-256 digests, recoverable)
        hs256 = 4,   ///< HMAC-SHA256: tests and the reference only, never production tokens
        rs256 = 5,   ///< RSASSA-PKCS1-v1_5 with SHA-256: verification of third-party tokens only; never signs
    };

    /// \brief The JOSE `alg` name: a wire token, not the enumerator's name.
    /// \ingroup core
    /// \param a The algorithm.
    /// \return `"ES256"`, `"EdDSA"`, `"ES256K"`, `"HS256"`, `"RS256"`, or `"unknown"`.
    [[nodiscard]] constexpr std::string_view jose_name(signature_algorithm a) noexcept
    {
        switch (a) {
        case signature_algorithm::es256: return "ES256";
        case signature_algorithm::eddsa: return "EdDSA";
        case signature_algorithm::es256k: return "ES256K";
        case signature_algorithm::hs256: return "HS256";
        case signature_algorithm::rs256: return "RS256";
        }
        return "unknown";
    }

    /// \brief Whether the algorithm has a public key.
    /// \ingroup core
    /// \param a The algorithm.
    /// \return `false` only for HS256.
    [[nodiscard]] constexpr bool is_asymmetric(signature_algorithm a) noexcept { return a != signature_algorithm::hs256; }

    /// \brief What a payload is.
    /// \ingroup core
    enum class payload_kind : std::uint8_t {
        message = 1,  ///< the bytes to sign; the signer hashes as its algorithm requires (a JWS signing input)
        digest = 2,   ///< a 32-byte digest already computed (an EVM transaction or EIP-712 hash)
    };

    /// \brief A key and a version; version 0 is the current one.
    /// \ingroup core
    struct key_ref {
        std::string_view key_id;   ///< the key's stable id (borrowed for the call)
        std::uint32_t version{0};  ///< 1, 2, ... or 0 for the current version

        /// \brief Equality.
        friend constexpr bool operator==(const key_ref&, const key_ref&) noexcept = default;
    };

    /// \brief One request to sign.
    /// \ingroup core
    struct sign_request {
        std::string_view key_id;                   ///< the key (its current version signs)
        std::span<const std::byte> payload;        ///< the bytes or the digest
        payload_kind kind{payload_kind::message};  ///< which of the two
    };

    /// \brief A signature, with what made it.
    /// \ingroup core
    struct signature {
        signature_algorithm algorithm{signature_algorithm::eddsa};  ///< the algorithm
        snicholls::provider_id key_id;                              ///< the key
        std::uint32_t key_version{0};                               ///< the version that signed (a JWT `kid` is `key_id.version`)
        std::vector<std::byte> bytes;  ///< JOSE form: r||s for ECDSA (64 bytes; 65 with v for ES256K), 64 for Ed25519, 32 for HS256

        /// \brief Equality.
        friend bool operator==(const signature&, const signature&) = default;
    };

    /// \brief A published public key.
    /// \ingroup core
    struct public_key {
        signature_algorithm algorithm{signature_algorithm::eddsa};  ///< the algorithm
        snicholls::provider_id key_id;                              ///< the key
        std::uint32_t key_version{0};                               ///< the version
        std::vector<std::byte> bytes;  ///< SEC1 uncompressed point (65 bytes) for ECDSA; 32 raw bytes for Ed25519

        /// \brief Equality.
        friend bool operator==(const public_key&, const public_key&) = default;
    };

    /// \brief A signer: `s.sign(request)` and `cs.public_key_of(ref)`.
    /// \ingroup core
    template <typename S>
    concept signer = requires(S& s, const S& cs, const sign_request& r, const key_ref& k) {
        { s.sign(r) } -> std::same_as<capability_result<signature>>;
        { cs.public_key_of(k) } -> std::same_as<capability_result<public_key>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_signer_hpp
