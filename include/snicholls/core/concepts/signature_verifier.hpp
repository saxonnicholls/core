//
//  signature_verifier.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signature_verifier`: check a JWS signature under a public key, or (HS256, tests only) under a
/// key the verifier holds by id.
/// \ingroup core
///
/// **Overview.** The seam between token verification (a JWT verifier) and the cryptography that checks a signature.
/// Our own access tokens are EdDSA (Ed25519); third-party OpenID Connect ID tokens are mostly RS256, some ES256. Each is
/// verified with **public keys only**: no private key ever reaches a verifier. This is the concept half of a pair; the
/// pure virtual half is `signature_verifier_interface` (`interfaces/signature_verifier.hpp`), with
/// `signature_verifier_adapter<V>` and `signature_verifier_ref`. The algorithms are `signature_algorithm`
/// (`concepts/signer.hpp`), so what a `signer` signs a `signature_verifier` checks.
///
/// **The contract** (`interfaces/signature_verifier_conformance.hpp` checks it):
/// - `verify(r)` is `ok()` exactly when `r.signature` is a valid signature of `r.message` under the key with
///   `r.algorithm`; a wrong signature, a wrong key or a tampered message is `capability_errc::signature_invalid`, and an
///   empty or truncated signature never verifies;
/// - an algorithm the verifier cannot check is `capability_errc::unsupported`, never `ok`;
/// - a malformed key (a wrong length, an Ed25519 key of small order) or a symmetric key without a version is
///   `capability_errc::invalid_argument`; a symmetric key the verifier does not hold is `capability_errc::not_found`;
/// - an asymmetric key is given as bytes in `r.public_key` (SEC1 uncompressed for ECDSA, 32 raw bytes for Ed25519,
///   for RS256 n and e each as a 4-byte big-endian length and its big-endian bytes); a symmetric key (HS256) is named by
///   `r.key_id` and `r.key_version`, and the verifier holds it;
/// - MACs are compared in constant time;
/// - errors follow the redaction rule of `capability_error.hpp`: no key, message, signature or provider response (a
///   KMS's, a JWKS endpoint's) in them.
///
/// \snippet concepts/signature_verifier_demo.cpp concept
///
/// **Relationships.** `concepts/signer.hpp` (`signature_algorithm`), `capability_error.hpp`; `concepts/token_verifier.hpp`
/// (its JWT implementation checks signatures through this); the twin `interfaces/signature_verifier.hpp`, the
/// reference `interfaces/signature_verifier_reference.hpp`, the kit `interfaces/signature_verifier_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations.**
/// - `memory_signature_verifier` (`interfaces/signature_verifier_reference.hpp`): HS256 under a `memory_key_store`'s
///   keys, standard library only, **for tests only**; every asymmetric algorithm is `unsupported`.
/// - Above core: servers' `crypto_signature_verifier` (`auth/crypto_verifiers.hpp`) over crypto's constant-time code:
///   EdDSA through `crypto::ed25519::verify` (small-order keys refused), RS256 through
///   `crypto::rsa::verify_pkcs1v15_sha256`, ES256 through `crypto::p256::verify`. Elliptic curves and RSA are crypto's,
///   never core's.
///
/// **Guarantees.** Thread safety is the implementation's. `verify_request` borrows everything for the call.
///
/// **Errors.** `capability_error`, as the contract says, and `unavailable` when what the verifier consults is down.
///
/// **Performance.** Not benchmarked here: the reference's cost is one `hmac<sha256>`
/// (`benchmarks/interfaces/key_store_reference_benchmark.cpp`).
///
/// **Tests and demos.** `tests/concepts/signature_verifier_test.cpp`, `demos/concepts/signature_verifier_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, `capability_error.hpp`,
/// `concepts/signer.hpp`.
///
/// **Known issues.** HS256 is for tests and the reference only; a production JWT policy refuses it before it reaches
/// a verifier.
///
/// **Status.** New in 0.27.0, lifted from servers. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_signature_verifier_hpp
#define snicholls_core_concepts_signature_verifier_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/concepts/signer.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls {

    /// \brief One signature to check; everything is borrowed for the call.
    /// \ingroup core
    struct verify_request {
        signature_algorithm algorithm{signature_algorithm::eddsa};  ///< the algorithm the token names
        std::string_view key_id;                                    ///< the key's id (a JWT `kid` without its version)
        std::uint32_t key_version{0};                               ///< the version; 0 when not given
        std::span<const std::byte> public_key;                      ///< the public key's bytes; empty for HS256
        std::span<const std::byte> message;                         ///< the signed bytes (the JWS signing input)
        std::span<const std::byte> signature;                       ///< the signature, in JOSE form
    };

    /// \brief A signature verifier: `v.verify(request)`, on a `const` verifier.
    /// \ingroup core
    template <typename V>
    concept signature_verifier = requires(const V& v, const verify_request& r) {
        { v.verify(r) } -> std::same_as<capability_result<void>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_signature_verifier_hpp
