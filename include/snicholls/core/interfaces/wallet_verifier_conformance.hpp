//
//  wallet_verifier_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_wallet_verifier_conformance`: the conformance kit of the `wallet_verifier` contract.
/// \ingroup core
///
/// **Overview.** Drives any `wallet_verifier`, in either form, through the contract of `concepts/wallet_verifier.hpp`
/// and reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& verifier()`: the implementation (a verifier, a `wallet_verifier_interface&`, a `wallet_verifier_ref`);
/// - `std::string signer_address()`: a wallet that can sign, and `std::string other_address()`: another;
/// - `std::vector<std::byte> sign(std::string_view message)`: `signer_address()`'s signature of the message;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `verify` fails with `code`, as
///   though the node the verifier asks had answered `response`.
///
/// The checks: `the signer verifies`, `another address is signature_invalid`, `a tampered message is
/// signature_invalid`, `a malformed signature is signature_invalid`, `an address not in the contract's form is
/// invalid_argument`, `a provider failure never leaks its response`.
///
/// \snippet interfaces/wallet_verifier_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/wallet_verifier.hpp`, `interfaces/wallet_verifier.hpp`,
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
/// **Tests and demos.** `tests/interfaces/wallet_verifier_conformance_test.cpp`,
/// `demos/interfaces/wallet_verifier_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`,
/// `concepts/wallet_verifier.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** Signing is the probe's: core holds no wallet key.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_wallet_verifier_conformance_hpp
#define snicholls_core_interfaces_wallet_verifier_conformance_hpp

#include <snicholls/core/concepts/wallet_verifier.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The sign-in message the kit has a wallet sign.
    /// \ingroup core
    inline constexpr std::string_view wallet_conformance_message =
        "example.test wants you to sign in with your Ethereum account:\n"
        "0x0000000000000000000000000000000000000000\n\nURI: https://example.test\nVersion: 1\nChain ID: 1\nNonce: k1t";

    namespace detail {
        template <typename Probe>
        concept wallet_verifier_probe = std::is_default_constructible_v<Probe> &&
            wallet_verifier<std::remove_cvref_t<decltype(std::declval<Probe&>().verifier())>> &&
            requires(Probe& p, std::string_view m, capability_errc c, std::string_view r) {
                { p.signer_address() } -> std::convertible_to<std::string>;
                { p.other_address() } -> std::convertible_to<std::string>;
                { p.sign(m) } -> std::convertible_to<std::vector<std::byte>>;
                p.provider_fails(c, r);
            };

        template <typename Probe>
        struct wallet_kit_case {
            std::string address;
            std::vector<std::byte> signature;
            explicit wallet_kit_case(Probe& p) : address(p.signer_address()), signature(p.sign(wallet_conformance_message)) {}
        };
    } // namespace detail

    /// \brief Runs every check of the `wallet_verifier` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::wallet_verifier_probe<Probe>
    [[nodiscard]] capability_conformance_report run_wallet_verifier_conformance()
    {
        capability_conformance_report report;
        using detail::wallet_kit_case;
        constexpr auto message = wallet_conformance_message;
        const auto is = [](const capability_result<void>& r, capability_errc code) { return r.is_err() && r.unwrap_err().code == code; };

        detail::capability_check<Probe>(report, "the signer verifies", [&](Probe& p, const auto& expect) {
            const wallet_kit_case<Probe> c{p};
            expect(p.verifier().verify({1, c.address, message, c.signature}).is_ok(), "the signer's signature verifies");
        });

        detail::capability_check<Probe>(report, "another address is signature_invalid", [&](Probe& p, const auto& expect) {
            const wallet_kit_case<Probe> c{p};
            const std::string other = p.other_address();
            expect(is(p.verifier().verify({1, other, message, c.signature}), capability_errc::signature_invalid),
                   "the signature does not verify for another wallet");
        });

        detail::capability_check<Probe>(report, "a tampered message is signature_invalid", [&](Probe& p, const auto& expect) {
            const wallet_kit_case<Probe> c{p};
            std::string tampered(message);
            tampered[0] = 'E';
            expect(is(p.verifier().verify({1, c.address, tampered, c.signature}), capability_errc::signature_invalid),
                   "one changed character is signature_invalid");
        });

        detail::capability_check<Probe>(report, "a malformed signature is signature_invalid", [&](Probe& p, const auto& expect) {
            wallet_kit_case<Probe> c{p};
            if (!c.signature.empty())
                c.signature.pop_back();
            expect(is(p.verifier().verify({1, c.address, message, c.signature}), capability_errc::signature_invalid),
                   "a signature one byte short is signature_invalid");
            expect(p.verifier().verify({1, c.address, message, {}}).is_err(), "an empty signature does not verify");
        });

        detail::capability_check<Probe>(report, "an address not in the contract's form is invalid_argument", [&](Probe& p, const auto& expect) {
            const wallet_kit_case<Probe> c{p};
            std::string upper = c.address;
            for (auto& ch : upper)
                if (ch >= 'a' && ch <= 'f')
                    ch = static_cast<char>(ch - 'a' + 'A');
            if (upper == c.address)
                upper[2] = 'g';
            for (const std::string_view a : {std::string_view{upper}, std::string_view{"0x1234"}, std::string_view{}})
                expect(is(p.verifier().verify({1, a, message, c.signature}), capability_errc::invalid_argument),
                       "an address with capitals, too short or empty is invalid_argument");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [&](Probe& p, const auto& expect) {
            const wallet_kit_case<Probe> c{p};
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto r = p.verifier().verify({1, c.address, message, c.signature});
            expect(is(r, capability_errc::unavailable), "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_wallet_verifier_conformance_hpp
