//
//  role_store_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_role_store_conformance`: the conformance kit of the `role_store` contract.
/// \ingroup core
///
/// **Overview.** Drives any `role_store`, in either form, through the contract of `concepts/role_store.hpp` and
/// reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type,
/// empty when made, with
/// - `auto& store()`: the implementation (a store, a `role_store_interface&`, a `role_store_ref`);
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next call fails with `code`, as
///   though the database behind it had answered `response`.
///
/// The checks: `a grant is active in its tenant only`, `an expired grant is inactive`, `a regrant replaces the grant
/// and appends an event`, `revoking what is not there is not found and appends nothing`, `history records the actor,
/// the reason and the confirmation in order`, `a provider failure never leaks its response`.
///
/// \snippet interfaces/role_store_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/role_store.hpp`, `interfaces/role_store.hpp`, `interfaces/capability_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/role_store_conformance_test.cpp`,
/// `demos/interfaces/role_store_conformance_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstddef>`, `<optional>`, `<string_view>`, `<type_traits>`, `<utility>`, `concepts/role_store.hpp`,
/// `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_role_store_conformance_hpp
#define snicholls_core_interfaces_role_store_conformance_hpp

#include <snicholls/core/concepts/role_store.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <chrono>
#include <cstddef>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept role_store_probe = std::is_default_constructible_v<Probe> &&
            role_store<std::remove_reference_t<decltype(std::declval<Probe&>().store())>> &&
            requires(Probe& p, capability_errc c, std::string_view r) { p.provider_fails(c, r); };

        inline constexpr uuid role_kit_alice = uuid::from_u64(0xa11ce, 1);
        inline constexpr uuid role_kit_bob = uuid::from_u64(0xb0b, 2);
        inline constexpr uuid role_kit_actor = uuid::from_u64(0xac7, 3);
        inline constexpr utc_s role_kit_t0 = utc_s::from_unix_seconds(1'800'000'000);

        inline role_grant role_kit_grant(const uuid& party, std::string_view tenant, std::string_view role)
        {
            return role_grant{party, *tenant_id::make(tenant), *role_id::make(role), role_kit_actor, "because", role_kit_t0, std::nullopt};
        }
    } // namespace detail

    /// \brief Runs every check of the `role_store` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::role_store_probe<Probe>
    [[nodiscard]] capability_conformance_report run_role_store_conformance()
    {
        capability_conformance_report report;
        using detail::role_kit_actor;
        using detail::role_kit_alice;
        using detail::role_kit_bob;
        using detail::role_kit_grant;
        using detail::role_kit_t0;
        const auto count = [](const auto& r) -> std::size_t { return r.is_ok() ? r->size() : 999; };

        detail::capability_check<Probe>(report, "a grant is active in its tenant only", [&](Probe& p, const auto& expect) {
            auto& s = p.store();
            expect(s.grant(role_kit_grant(role_kit_alice, "acme", "admin"), "").is_ok(), "a grant is recorded");
            expect(s.grant(role_kit_grant(role_kit_alice, platform_tenant, "operator"), "").is_ok(), "a staff grant is recorded");
            const auto in_acme = s.grants_of(role_kit_alice, "acme", role_kit_t0);
            expect(count(in_acme) == 1 && (*in_acme)[0].role == std::string_view{"admin"}, "the tenant's grant, and not the staff role");
            expect(count(s.grants_of(role_kit_alice, "globex", role_kit_t0)) == 0, "nothing in another tenant");
            expect(count(s.grants_of(role_kit_bob, "acme", role_kit_t0)) == 0, "nothing for another party");
            expect(count(s.members("acme", role_kit_t0)) == 1, "the tenant has one member");
            expect(count(s.members(platform_tenant, role_kit_t0)) == 1, "the platform has one member");
        });

        detail::capability_check<Probe>(report, "an expired grant is inactive", [&](Probe& p, const auto& expect) {
            auto& s = p.store();
            auto g = role_kit_grant(role_kit_bob, "acme", "auditor");
            g.expires_at = role_kit_t0 + std::chrono::hours{24};
            expect(s.grant(g, "").is_ok(), "a time-bounded grant is recorded");
            expect(count(s.grants_of(role_kit_bob, "acme", role_kit_t0)) == 1, "active before its expiry");
            expect(count(s.grants_of(role_kit_bob, "acme", role_kit_t0 + std::chrono::hours{24})) == 0, "inactive at its expiry");
            expect(count(s.members("acme", role_kit_t0 + std::chrono::hours{25})) == 0, "not a member after it");
        });

        detail::capability_check<Probe>(report, "a regrant replaces the grant and appends an event", [&](Probe& p, const auto& expect) {
            auto& s = p.store();
            expect(s.grant(role_kit_grant(role_kit_bob, "acme", "member"), "").is_ok(), "the first grant");
            auto again = role_kit_grant(role_kit_bob, "acme", "member");
            again.reason = "renewed";
            expect(s.grant(again, "").is_ok(), "the regrant");
            const auto now = s.grants_of(role_kit_bob, "acme", role_kit_t0);
            expect(count(now) == 1 && (*now)[0].reason == std::string_view{"renewed"}, "one grant, the new one");
            expect(count(s.history("acme")) == 2, "two events");
        });

        detail::capability_check<Probe>(report, "revoking what is not there is not found and appends nothing", [&](Probe& p, const auto& expect) {
            auto& s = p.store();
            expect(s.grant(role_kit_grant(role_kit_bob, "acme", "member"), "").is_ok(), "a grant");
            const auto none = s.revoke(role_kit_bob, "acme", "owner", role_kit_actor, "never held", role_kit_t0);
            expect(none.is_err() && none.unwrap_err().code == capability_errc::not_found, "a role never granted is not_found");
            expect(s.revoke(role_kit_bob, "acme", "member", role_kit_actor, "left", role_kit_t0).is_ok(), "the revoke");
            const auto twice = s.revoke(role_kit_bob, "acme", "member", role_kit_actor, "again", role_kit_t0);
            expect(twice.is_err() && twice.unwrap_err().code == capability_errc::not_found, "a second revoke is not_found");
            expect(count(s.grants_of(role_kit_bob, "acme", role_kit_t0)) == 0, "the grant is gone");
            expect(count(s.history("acme")) == 2, "only the grant and the one revoke are recorded");
        });

        detail::capability_check<Probe>(report, "history records the actor, the reason and the confirmation in order", [&](Probe& p, const auto& expect) {
            auto& s = p.store();
            expect(s.grant(role_kit_grant(role_kit_bob, "acme", "member"), "").is_ok(), "a grant");
            expect(s.grant(role_kit_grant(role_kit_alice, "globex", "member"), "").is_ok(), "a grant elsewhere");
            expect(s.grant(role_kit_grant(role_kit_bob, "acme", "admin"), "Grant admin to bob in acme").is_ok(), "a confirmed grant");
            expect(s.revoke(role_kit_bob, "acme", "member", role_kit_alice, "left", role_kit_t0 + std::chrono::hours{1}).is_ok(), "a revoke");
            const auto h = s.history("acme");
            expect(count(h) == 3, "three events in the tenant, none from another");
            if (count(h) != 3)
                return;
            expect((*h)[0].granted && (*h)[0].grant.role == std::string_view{"member"}, "first, the grant");
            expect((*h)[1].confirmation == std::string_view{"Grant admin to bob in acme"}, "then the confirmed grant");
            expect(!(*h)[2].granted && (*h)[2].grant.granted_by == role_kit_alice, "last, the revoke, by its actor");
            expect((*h)[2].grant.reason == std::string_view{"left"} && (*h)[2].grant.granted_at == role_kit_t0 + std::chrono::hours{1},
                   "with the revoke's reason and time");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [&](Probe& p, const auto& expect) {
            auto& s = p.store();
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto r = s.grant(role_kit_grant(role_kit_alice, "acme", "admin"), "");
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
            expect(count(s.history("acme")) == 0, "a failed grant records nothing");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_role_store_conformance_hpp
