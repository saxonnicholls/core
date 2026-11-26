//
//  capability_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief What the capability conformance kits report, and the redaction check they share.
/// \ingroup core
///
/// **Overview.** Each capability's kit (`interfaces/message_sender_conformance.hpp`, `signer_conformance.hpp`,
/// `key_store_conformance.hpp`, `payment_provider_conformance.hpp`, `onramp_provider_conformance.hpp`,
/// `http_handler_conformance.hpp`, `token_verifier_conformance.hpp`, `signature_verifier_conformance.hpp`,
/// `wallet_verifier_conformance.hpp`, `role_store_conformance.hpp`, `https_client_conformance.hpp`) is a plain
/// function template over a *probe*, a default-constructible type the implementation's test writes, which hands the
/// kit the implementation in either form (a model of the concept, an interface reference, a handle). The kit makes a fresh probe for every check, runs the check, and records each
/// expectation that failed in a `capability_conformance_report`, with the check's name. No test framework is
/// included: a test asserts `report.ok()` and prints the failures.
///
/// **The redaction check.** Every kit with a `capability_error` asks the probe to make the implementation's next
/// call fail as though the provider had answered `capability_provider_response`, a body that carries
/// `capability_redaction_marker` next to user data, and checks with `capability_error_leaks` that the marker reaches
/// neither `provider_code` nor `message` (`capability_error.hpp`, "the redaction rule"). The HTTP kit checks the same
/// of a handler that throws with the marker in its message.
///
/// \snippet interfaces/capability_conformance_demo.cpp report
///
/// **Relationships.** `capability_error.hpp`; the eleven kits. Group: \ref core.
///
/// **Implementations / variants.** One report for every capability kit. `kv_conformance_report`
/// (`interfaces/key_value_store_conformance.hpp`) is the same shape, older.
///
/// **Guarantees.** A check never lets an exception out: one that escapes is recorded as the failure "no exception
/// escapes". The report allocates; the contracts do not.
///
/// **Errors.** Failures are data, never thrown.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/capability_conformance_test.cpp`,
/// `demos/interfaces/capability_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<string_view>`, `<utility>`, `<vector>`, `capability_error.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0. Experimental, with the contracts.

#ifndef snicholls_core_interfaces_capability_conformance_hpp
#define snicholls_core_interfaces_capability_conformance_hpp

#include <snicholls/core/capability_error.hpp>

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief One expectation that failed: the check's name and what was expected (both static strings).
    /// \ingroup core
    struct capability_conformance_failure {
        std::string_view check;        ///< the check, e.g. `"the same idempotency key sends once"`
        std::string_view expectation;  ///< what did not hold
    };

    /// \brief What a run of a capability kit found.
    /// \ingroup core
    struct capability_conformance_report {
        std::size_t checks = 0;                                ///< how many checks ran
        std::vector<capability_conformance_failure> failures;  ///< every expectation that failed, in order

        /// \brief Whether every check passed.
        /// \return `failures.empty()`.
        [[nodiscard]] bool ok() const noexcept { return failures.empty(); }

        /// \brief Whether the check `name` recorded a failure.
        /// \param name The check's name.
        /// \return `true` if it failed.
        [[nodiscard]] bool failed(std::string_view name) const noexcept
        {
            for (const auto& f : failures)
                if (f.check == name)
                    return true;
            return false;
        }
    };

    /// \brief The marker the kits plant in a provider's response; it must never reach a `capability_error`.
    /// \ingroup core
    inline constexpr std::string_view capability_redaction_marker = "REDACTION-MARKER-c0re";

    /// \brief The provider response the kits make an implementation fail with: the provider's code first, then a body
    /// that carries the marker beside user data and a secret.
    /// \ingroup core
    inline constexpr std::string_view capability_provider_response =
        "provider_error {\"message\":\"REDACTION-MARKER-c0re for someone@example.test +61400000000\","
        "\"api_key\":\"sk_live_REDACTION-MARKER-c0re\"}";

    /// \brief Whether `marker` appears in the error's provider code or message.
    /// \ingroup core
    /// \param e      The error.
    /// \param marker The text that must not appear.
    /// \return `true` when it leaked.
    [[nodiscard]] constexpr bool capability_error_leaks(const capability_error& e, std::string_view marker) noexcept
    {
        return e.provider_code.view().find(marker) != std::string_view::npos ||
               e.message.view().find(marker) != std::string_view::npos;
    }

    namespace detail {
        // Records the failures of one check.
        struct capability_expect {
            std::string_view check;
            capability_conformance_report& report;

            void operator()(bool holds, std::string_view expectation) const
            {
                if (!holds)
                    report.failures.push_back({check, expectation});
            }
        };

        // Runs one check on a fresh probe; an escaping exception is a failure, not a crash.
        template <typename Probe, typename F>
        void capability_check(capability_conformance_report& report, std::string_view name, F&& f)
        {
            ++report.checks;
            const capability_expect expect{name, report};
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try {
                Probe probe{};
                std::forward<F>(f)(probe, expect);
            } catch (...) {
                expect(false, "no exception escapes");
            }
#else
            Probe probe{};
            std::forward<F>(f)(probe, expect);
#endif
        }
    } // namespace detail

} // namespace snicholls

#endif // snicholls_core_interfaces_capability_conformance_hpp
