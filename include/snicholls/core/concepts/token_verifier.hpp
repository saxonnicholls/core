//
//  token_verifier.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::token_verifier`: a presented credential in, a verified caller (`token_claims`) out.
/// \ingroup core
///
/// **Overview.** Every authenticated route asks one question: who presented this token? A token verifier answers it
/// without the route knowing what a token is: a JWT checked against a JWKS (EdDSA, ES256, RS256), an opaque session id
/// looked up in a session store, a hashed API key. The route's `authenticate` step calls `verify` and puts the claims
/// in the request's context. This is the concept half of a pair; the pure virtual half is `token_verifier_interface`
/// (`interfaces/token_verifier.hpp`), with `token_verifier_adapter<V>` and `token_verifier_ref`.
///
/// The caller is a **party**, named by its `uuid`: the same id the role store keys grants by
/// (`concepts/role_store.hpp`), so `roles.grants_of(claims.subject, tenant, now)` needs no translation. The claims'
/// `roles` are informational: authorisation re-reads them from the role store on every request, so a revocation does
/// not wait for the token to expire. What a role permits is the policy's (servers' role catalogue), not the token's.
///
/// **The contract** (`interfaces/token_verifier_conformance.hpp` checks it):
/// - a valid token gives `token_claims` naming its party in `subject`;
/// - an unknown, malformed, forged or altered token is `capability_errc::unauthorized`, the empty token too (an
///   implementation that checks a signature may answer `signature_invalid` for an altered one);
/// - an expired token is `capability_errc::stale`, so the client refreshes instead of signing in again;
/// - `verify` is `const` and returns every failure as a value;
/// - **the token never reaches an error**, and neither does a provider's response (a JWKS endpoint's, a session
///   store's): the redaction rule of `capability_error.hpp`. An implementation never logs the token.
///
/// \snippet concepts/token_verifier_demo.cpp concept
///
/// **Relationships.** `capability_error.hpp`, `capability_ids.hpp` (`role_id`, `permission_id`, `wallet_address`,
/// `short_text`), `uuid.hpp`, `time/utc.hpp`; `concepts/role_store.hpp` (the authority on roles),
/// `concepts/signature_verifier.hpp` (what a JWT verifier checks its signature with); the twin
/// `interfaces/token_verifier.hpp`, the reference `interfaces/token_verifier_reference.hpp`, the kit
/// `interfaces/token_verifier_conformance.hpp`. Group: \ref core.
///
/// **Implementations.**
/// - `memory_token_verifier` (`interfaces/token_verifier_reference.hpp`): tokens it issues itself, HS256-tagged with
///   core's `hmac<sha256>` and checked with `constant_time_equal`; standard library only, **for tests only**.
/// - Above core: servers' JWT verifier (`auth/jwt.hpp`: base64url, JSON, the JWKS, the policy that refuses HS256)
///   over a `signature_verifier` whose EdDSA, ES256 and RS256 checks are crypto's; servers' session-store verifier.
///   Neither JSON, base64 nor elliptic curves are core's.
///
/// **Guarantees.** `has_role` and `has_scope` are linear in the claims and allocation-free. Thread safety is the
/// implementation's; a verifier shared by routes must be safe to call concurrently.
///
/// **Errors.** `capability_error`: `unauthorized`, `stale`, `signature_invalid`, and `unavailable` (or `timeout`)
/// when what it consults is down.
///
/// **Performance.** `benchmarks/interfaces/token_verifier_reference_benchmark.cpp` (the reference's `verify`).
///
/// **Tests and demos.** `tests/concepts/token_verifier_test.cpp`, `demos/concepts/token_verifier_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<concepts>`, `<cstdint>`, `<optional>`, `<string_view>`, `<vector>`,
/// `capability_error.hpp`, `capability_ids.hpp`, `uuid.hpp`.
///
/// **Known issues.** `token_claims` allocates for its role and scope lists; everything else in it is bounded.
///
/// **Status.** New in 0.27.0, lifted from servers. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_token_verifier_hpp
#define snicholls_core_concepts_token_verifier_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/uuid.hpp>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief How the caller signed in: the four doors, then the machine credential.
    /// \ingroup core
    enum class sign_in_method : std::uint8_t {
        email = 1,    ///< a one-time code or a magic link
        sms = 2,      ///< a one-time code to an E.164 number
        wallet = 3,   ///< Sign-In with Ethereum (EIP-4361)
        oauth = 4,    ///< OAuth 2.1 / OpenID Connect
        api_key = 5,  ///< a machine credential
    };

    /// \brief A verified caller.
    /// \ingroup core
    struct token_claims {
        uuid subject{};                                ///< the party (`sub`)
        sign_in_method method{sign_in_method::email};  ///< how it signed in
        wallet_address wallet{};                       ///< the SIWE address, lowercase `0x...`, if any
        std::vector<role_id> roles;                    ///< informational: the role store decides
        std::vector<permission_id> scopes;             ///< OAuth or API-key scopes
        short_text audience{};                         ///< `aud`: the service the token is for
        std::optional<utc_s> expires_at;               ///< `exp`
        bool second_factor{false};                     ///< a second factor (TOTP) passed in this session
        utc_s second_factor_at{};                      ///< when it last passed (step-up freshness); the epoch if never
        std::optional<uuid> token_owner;               ///< for an `api_key`: the party it acts for

        /// \brief Whether `role` is claimed.
        /// \param role The role.
        /// \return `true` if so.
        [[nodiscard]] bool has_role(std::string_view role) const noexcept
        {
            return std::any_of(roles.begin(), roles.end(), [role](const role_id& r) { return r == role; });
        }
        /// \brief Whether `scope` is held.
        /// \param scope The scope.
        /// \return `true` if so.
        [[nodiscard]] bool has_scope(std::string_view scope) const noexcept
        {
            return std::any_of(scopes.begin(), scopes.end(), [scope](const permission_id& s) { return s == scope; });
        }

        /// \brief Equality, member by member.
        friend bool operator==(const token_claims&, const token_claims&) = default;
    };

    /// \brief A token verifier: `v.verify(token)`, on a `const` verifier.
    /// \ingroup core
    template <typename V>
    concept token_verifier = requires(const V& v, std::string_view token) {
        { v.verify(token) } -> std::same_as<capability_result<token_claims>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_token_verifier_hpp
