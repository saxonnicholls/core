//
//  signature_verifier_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_signature_verifier_conformance`: the conformance kit of the `signature_verifier` contract.
/// \ingroup core
///
/// **Overview.** Drives any `signature_verifier`, in either form, through the contract of
/// `concepts/signature_verifier.hpp` and reports which check failed (`interfaces/capability_conformance.hpp`). The
/// probe is a default-constructible type with
/// - `auto& verifier()`: the implementation (a verifier, a `signature_verifier_interface&`, a `signature_verifier_ref`);
/// - `signed_sample sign(std::span<const std::byte> message)`: a valid signature over the bytes, with its key;
/// - `signature_algorithm unsupported()`: an algorithm this verifier cannot check;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `verify` fails with `code`, as
///   though what the verifier consults (a key store, a JWKS endpoint) had answered `response`.
///
/// The checks: `a valid signature verifies`, `a tampered message is signature_invalid`, `a tampered signature is
/// signature_invalid`, `a truncated signature never verifies`, `an algorithm it cannot check is unsupported`, `a provider
/// failure never leaks its response`.
///
/// \snippet interfaces/signature_verifier_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/signature_verifier.hpp`, `interfaces/signature_verifier.hpp`,
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
/// **Tests and demos.** `tests/interfaces/signature_verifier_conformance_test.cpp`,
/// `demos/interfaces/signature_verifier_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<span>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `<vector>`, `concepts/signature_verifier.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** Signing is the probe's, because core signs nothing asymmetric.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_signature_verifier_conformance_hpp
#define snicholls_core_interfaces_signature_verifier_conformance_hpp

#include <snicholls/core/concepts/signature_verifier.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief A signature a probe made, with everything needed to verify it.
    /// \ingroup core
    struct signed_sample {
        signature_algorithm algorithm{signature_algorithm::eddsa};  ///< the algorithm
        std::string key_id;                                         ///< the key
        std::uint32_t key_version{0};                               ///< its version
        std::vector<std::byte> public_key;                          ///< empty for HS256
        std::vector<std::byte> signature;                           ///< the signature

        /// \brief The request that checks it over `message`.
        /// \param message The signed bytes.
        /// \return The request, borrowing this sample and the message.
        [[nodiscard]] verify_request request(std::span<const std::byte> message) const noexcept
        {
            return {algorithm, key_id, key_version, public_key, message, signature};
        }
    };

    namespace detail {
        inline std::span<const std::byte> signature_kit_message() noexcept
        {
            static constexpr std::string_view text = "eyJhbGciOiJFZERTQSJ9.eyJzdWIiOiJraXQifQ";
            return std::as_bytes(std::span<const char>(text.data(), text.size()));
        }

        template <typename Probe>
        concept signature_verifier_probe = std::is_default_constructible_v<Probe> &&
            signature_verifier<std::remove_cvref_t<decltype(std::declval<Probe&>().verifier())>> &&
            requires(Probe& p, std::span<const std::byte> m, capability_errc c, std::string_view r) {
                { p.sign(m) } -> std::convertible_to<signed_sample>;
                { p.unsupported() } -> std::convertible_to<signature_algorithm>;
                p.provider_fails(c, r);
            };
    } // namespace detail

    /// \brief Runs every check of the `signature_verifier` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::signature_verifier_probe<Probe>
    [[nodiscard]] capability_conformance_report run_signature_verifier_conformance()
    {
        capability_conformance_report report;
        using detail::signature_kit_message;

        detail::capability_check<Probe>(report, "a valid signature verifies", [](Probe& p, const auto& expect) {
            const signed_sample s = p.sign(signature_kit_message());
            expect(p.verifier().verify(s.request(signature_kit_message())).is_ok(), "the probe's signature verifies");
        });

        detail::capability_check<Probe>(report, "a tampered message is signature_invalid", [](Probe& p, const auto& expect) {
            const signed_sample s = p.sign(signature_kit_message());
            std::vector<std::byte> tampered(signature_kit_message().begin(), signature_kit_message().end());
            tampered.back() ^= std::byte{1};
            const auto r = p.verifier().verify(s.request(tampered));
            expect(r.is_err() && r.unwrap_err().code == capability_errc::signature_invalid, "one flipped bit is signature_invalid");
        });

        detail::capability_check<Probe>(report, "a tampered signature is signature_invalid", [](Probe& p, const auto& expect) {
            signed_sample s = p.sign(signature_kit_message());
            expect(!s.signature.empty(), "the signature has bytes");
            if (s.signature.empty())
                return;
            s.signature.front() ^= std::byte{0x80};
            const auto r = p.verifier().verify(s.request(signature_kit_message()));
            expect(r.is_err() && r.unwrap_err().code == capability_errc::signature_invalid, "one flipped bit is signature_invalid");
        });

        detail::capability_check<Probe>(report, "a truncated signature never verifies", [](Probe& p, const auto& expect) {
            signed_sample s = p.sign(signature_kit_message());
            if (!s.signature.empty())
                s.signature.pop_back();
            expect(p.verifier().verify(s.request(signature_kit_message())).is_err(), "one byte short does not verify");
            s.signature.clear();
            expect(p.verifier().verify(s.request(signature_kit_message())).is_err(), "an empty signature does not verify");
        });

        detail::capability_check<Probe>(report, "an algorithm it cannot check is unsupported", [](Probe& p, const auto& expect) {
            signed_sample s = p.sign(signature_kit_message());
            s.algorithm = p.unsupported();
            const auto r = p.verifier().verify(s.request(signature_kit_message()));
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unsupported, "the algorithm is unsupported, never ok");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            const signed_sample s = p.sign(signature_kit_message());
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto r = p.verifier().verify(s.request(signature_kit_message()));
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_signature_verifier_conformance_hpp
