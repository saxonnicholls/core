//
//  role_store.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::role_store`: which party holds which role in which tenant, append-only, with its history.
/// \ingroup core
///
/// **Overview.** Authorisation re-reads a caller's roles from here on every request, so a revocation takes effect on
/// the next call. A grant names a party (its `uuid`, the `subject` of `token_claims`), a tenant (an organisation, or
/// `platform_tenant` for staff), a role, who granted it, why, when, and optionally until when (an auditor's window).
/// Nothing is updated in place: a grant or a revoke appends an event, and the current view is derived. The store
/// records and does not decide: what a role permits (its permissions), rank rules, the single owner and bootstrap
/// admins are the policy's (servers' role catalogue, `routes/rbac.hpp`). This is the concept half of a pair; the pure
/// virtual half is `role_store_interface` (`interfaces/role_store.hpp`), with `role_store_adapter<S>` and
/// `role_store_ref`.
///
/// **The contract** (`interfaces/role_store_conformance.hpp` checks it):
/// - `grants_of(party, tenant, now)` is the party's active grants in that tenant **only** (a staff role in
///   `platform_tenant` never shows in a customer's tenant); a grant with `expires_at <= now` is not active;
/// - `members(tenant, now)` is every active grant in the tenant;
/// - `grant` of an existing (party, tenant, role) replaces it (a new expiry, a new reason) and appends an event;
/// - `revoke` of a grant that is not there is `capability_errc::not_found` and appends nothing;
/// - `history(tenant)` is every event of the tenant in order, oldest first, with the actor, the reason and any
///   confirmation;
/// - a confirmation or reason longer than `short_text` is `invalid_argument`, never truncated;
/// - a failure of the store behind it is returned (`unavailable`, `internal`), never an empty answer, and follows the
///   redaction rule of `capability_error.hpp`.
///
/// \snippet concepts/role_store_demo.cpp concept
///
/// **Relationships.** `uuid.hpp` (the party), `capability_ids.hpp` (`tenant_id`, `role_id`, `short_text`),
/// `time/utc.hpp`; `concepts/token_verifier.hpp` (whose `subject` is the party); the twin `interfaces/role_store.hpp`,
/// the reference `interfaces/role_store_reference.hpp`, the kit `interfaces/role_store_conformance.hpp`. Group:
/// \ref core.
///
/// **Implementations.**
/// - `kv_role_store<S>` (`interfaces/role_store_reference.hpp`) over any `key_value_store`, and `memory_role_store`, it
///   over core's `reference_key_value_store`: standard library only.
/// - Above core: servers' Postgres store, an events table and a materialised current view.
///
/// **Guarantees.** `role_grant` and `role_event` hold no `std::string`: both are trivially copyable. Thread safety is
/// the implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked: reads are a prefix scan of one party's grants.
///
/// **Tests and demos.** `tests/concepts/role_store_test.cpp`, `demos/concepts/role_store_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<optional>`, `<string_view>`, `<vector>`, `capability_error.hpp`,
/// `capability_ids.hpp`, `uuid.hpp`.
///
/// **Known issues.** A role id names a role across tenants; a product that needs per-tenant role definitions keeps
/// them in its policy.
///
/// **Status.** New in 0.27.0, lifted from servers. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_role_store_hpp
#define snicholls_core_concepts_role_store_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/uuid.hpp>

#include <concepts>
#include <optional>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief The tenant that holds staff roles.
    /// \ingroup core
    inline constexpr std::string_view platform_tenant = "platform";

    /// \brief One role held by one party in one tenant.
    /// \ingroup core
    struct role_grant {
        uuid subject{};                   ///< the party
        tenant_id tenant{};               ///< the organisation, or `platform_tenant`
        role_id role{};                   ///< the role
        uuid granted_by{};                ///< the actor
        short_text reason{};              ///< why, for the audit
        utc_s granted_at{};               ///< when
        std::optional<utc_s> expires_at;  ///< the end of time-bounded access, if any

        /// \brief Whether it is in force at `now`.
        /// \param now The time.
        /// \return `true` when it has no expiry or expires after `now`.
        [[nodiscard]] constexpr bool active(utc_s now) const noexcept { return !expires_at || *expires_at > now; }

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const role_grant&, const role_grant&) noexcept = default;
    };

    /// \brief One entry in a tenant's append-only history.
    /// \ingroup core
    struct role_event {
        bool granted{true};        ///< a grant, or a revoke
        role_grant grant{};        ///< what; for a revoke, `granted_by`, `reason` and `granted_at` are the revoke's
        short_text confirmation{}; ///< the sentence the actor confirmed, if any

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const role_event&, const role_event&) noexcept = default;
    };

    /// \brief A role store: `grants_of`, `members` and `history` on a `const` store; `grant` and `revoke`.
    /// \ingroup core
    template <typename S>
    concept role_store = requires(S& s, const S& cs, const uuid& party, std::string_view text, const role_grant& g, utc_s now) {
        { cs.grants_of(party, text, now) } -> std::same_as<capability_result<std::vector<role_grant>>>;
        { cs.members(text, now) } -> std::same_as<capability_result<std::vector<role_grant>>>;
        { s.grant(g, text) } -> std::same_as<capability_result<void>>;
        { s.revoke(party, text, text, party, text, now) } -> std::same_as<capability_result<void>>;
        { cs.history(text) } -> std::same_as<capability_result<std::vector<role_event>>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_role_store_hpp
