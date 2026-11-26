//
//  token_verifier_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_token_verifier_conformance`: the conformance kit of the `token_verifier` contract.
/// \ingroup core
///
/// **Overview.** Drives any `token_verifier`, in either form, through the contract of `concepts/token_verifier.hpp`
/// and reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& verifier()`: the implementation (a verifier, a `token_verifier_interface&`, a `token_verifier_ref`);
/// - `std::string valid_token()`: a token that verifies, for the party `subject()`;
/// - `uuid subject()`: that party;
/// - `std::string expired_token()`: a token that verified once and has expired;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `verify` fails with `code`, as
///   though what the verifier consults (a JWKS endpoint, a session store) had answered `response`.
///
/// The checks: `a valid token names its party`, `garbage and the empty token are unauthorized`, `an altered token is
/// refused`, `an expired token is stale`, `a refused token is never echoed` (a token carrying the redaction marker is
/// refused without the marker reaching the error), `a provider failure never leaks its response`.
///
/// \snippet interfaces/token_verifier_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/token_verifier.hpp`, `interfaces/token_verifier.hpp`,
/// `interfaces/capability_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/token_verifier_conformance_test.cpp`,
/// `demos/interfaces/token_verifier_conformance_demo.cpp`.
///
/// **Dependencies.** `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `concepts/token_verifier.hpp`,
/// `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** The kit cannot mint a forged token for every scheme, so "altered" is the valid token with its last
/// character changed; a probe's own tests cover forgery of its scheme.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_token_verifier_conformance_hpp
#define snicholls_core_interfaces_token_verifier_conformance_hpp

#include <snicholls/core/concepts/token_verifier.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept token_verifier_probe = std::is_default_constructible_v<Probe> &&
            token_verifier<std::remove_cvref_t<decltype(std::declval<Probe&>().verifier())>> &&
            requires(Probe& p, capability_errc c, std::string_view r) {
                { p.valid_token() } -> std::convertible_to<std::string>;
                { p.expired_token() } -> std::convertible_to<std::string>;
                { p.subject() } -> std::convertible_to<uuid>;
                p.provider_fails(c, r);
            };
    } // namespace detail

    /// \brief Runs every check of the `token_verifier` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::token_verifier_probe<Probe>
    [[nodiscard]] capability_conformance_report run_token_verifier_conformance()
    {
        capability_conformance_report report;

        detail::capability_check<Probe>(report, "a valid token names its party", [](Probe& p, const auto& expect) {
            const std::string token = p.valid_token();
            const auto r = p.verifier().verify(token);
            expect(r.is_ok(), "the valid token verifies");
            if (r.is_ok())
                expect(r->subject == p.subject() && !r->subject.is_nil(), "the claims name the token's party");
        });

        detail::capability_check<Probe>(report, "garbage and the empty token are unauthorized", [](Probe& p, const auto& expect) {
            for (const std::string_view t : {std::string_view{}, std::string_view{"garbage"},
                                             std::string_view{"eyJhbGciOiJub25lIn0.eyJzdWIiOiJ4In0."}}) {
                const auto r = p.verifier().verify(t);
                expect(r.is_err() && r.unwrap_err().code == capability_errc::unauthorized,
                       "the empty token, garbage and an unsigned JWT are unauthorized");
            }
        });

        detail::capability_check<Probe>(report, "an altered token is refused", [](Probe& p, const auto& expect) {
            std::string t = p.valid_token();
            expect(!t.empty(), "the valid token is not empty");
            if (t.empty())
                return;
            t.back() = t.back() == 'A' ? 'B' : 'A';
            const auto r = p.verifier().verify(t);
            expect(r.is_err() && (r.unwrap_err().code == capability_errc::unauthorized ||
                                  r.unwrap_err().code == capability_errc::signature_invalid),
                   "a token with its last character changed is unauthorized (or signature_invalid)");
        });

        detail::capability_check<Probe>(report, "an expired token is stale", [](Probe& p, const auto& expect) {
            const std::string t = p.expired_token();
            const auto r = p.verifier().verify(t);
            expect(r.is_err() && r.unwrap_err().code == capability_errc::stale, "an expired token is stale");
        });

        detail::capability_check<Probe>(report, "a refused token is never echoed", [](Probe& p, const auto& expect) {
            const std::string planted = p.valid_token() + std::string(capability_redaction_marker);
            for (const std::string& t : {std::string(capability_redaction_marker), planted}) {
                const auto r = p.verifier().verify(t);
                expect(r.is_err(), "a token carrying the marker is refused");
                if (r.is_err())
                    expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                           "neither provider_code nor message carries the token");
            }
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            const std::string t = p.valid_token();
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto r = p.verifier().verify(t);
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_token_verifier_conformance_hpp
