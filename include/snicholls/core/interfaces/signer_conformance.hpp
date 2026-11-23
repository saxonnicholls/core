//
//  signer_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_signer_conformance`: the conformance kit of the `signer` contract.
/// \ingroup core
///
/// **Overview.** Drives any `signer`, in either form, through the contract of `concepts/signer.hpp` and reports which
/// check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type with
/// - `auto& signer()`: the implementation (a signer, a `signer_interface&`, a `signer_ref`);
/// - `std::string active_key()` and `std::string retired_key()`: the ids of a key that signs and of one retired;
/// - `bool verify(const signature&, std::span<const std::byte> payload)`: the probe's own check of a signature;
/// - `bool deterministic()`: whether two signatures of one payload are equal;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `sign` fails with `code`, as
///   though the KMS had answered `response`.
///
/// The checks: `a signature verifies and names its key and version`, `an unknown key is not found`, `a retired key
/// no longer signs`, `a digest that is not 32 bytes is refused`, `a deterministic signer repeats itself`, `public
/// keys exist exactly for asymmetric algorithms`, `a provider failure never leaks its response`.
///
/// \snippet interfaces/signer_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/signer.hpp`, `interfaces/signer.hpp`, `interfaces/capability_conformance.hpp`. Group:
/// \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/signer_conformance_test.cpp`, `demos/interfaces/signer_conformance_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<span>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `concepts/signer.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** Verification is the probe's, because core has no asymmetric verifier.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_signer_conformance_hpp
#define snicholls_core_interfaces_signer_conformance_hpp

#include <snicholls/core/concepts/signer.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        inline std::span<const std::byte> kit_payload() noexcept
        {
            static constexpr std::string_view text = "eyJhbGciOi.eyJzdWIiOiJ0ZXN0In0-jws";
            return std::as_bytes(std::span<const char>(text.data(), text.size()));
        }

        template <typename Probe>
        concept signer_probe = std::is_default_constructible_v<Probe> &&
            signer<std::remove_reference_t<decltype(std::declval<Probe&>().signer())>> &&
            requires(Probe& p, const signature& s, std::span<const std::byte> b, capability_errc c, std::string_view r) {
                { p.active_key() } -> std::convertible_to<std::string>;
                { p.retired_key() } -> std::convertible_to<std::string>;
                { p.verify(s, b) } -> std::same_as<bool>;
                { p.deterministic() } -> std::same_as<bool>;
                p.provider_fails(c, r);
            };
    } // namespace detail

    /// \brief Runs every check of the `signer` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::signer_probe<Probe>
    [[nodiscard]] capability_conformance_report run_signer_conformance()
    {
        capability_conformance_report report;
        using detail::kit_payload;

        detail::capability_check<Probe>(report, "a signature verifies and names its key and version", [](Probe& p, const auto& expect) {
            const std::string id = p.active_key();
            const auto r = p.signer().sign({id, kit_payload(), payload_kind::message});
            expect(r.is_ok(), "the active key signs");
            if (r.is_ok()) {
                expect(r->key_id == std::string_view{id}, "the signature names its key");
                expect(r->key_version >= 1, "the signature names its version");
                expect(!r->bytes.empty(), "the signature has bytes");
                expect(p.verify(*r, kit_payload()), "the signature verifies");
            }
        });

        detail::capability_check<Probe>(report, "an unknown key is not found", [](Probe& p, const auto& expect) {
            const auto s = p.signer().sign({"no-such-key", kit_payload(), payload_kind::message});
            expect(s.is_err() && s.unwrap_err().code == capability_errc::not_found, "sign with an unknown key is not_found");
            const auto k = p.signer().public_key_of({"no-such-key", 0});
            expect(k.is_err() && k.unwrap_err().code == capability_errc::not_found, "public_key_of an unknown key is not_found");
        });

        detail::capability_check<Probe>(report, "a retired key no longer signs", [](Probe& p, const auto& expect) {
            const std::string id = p.retired_key();
            const auto r = p.signer().sign({id, kit_payload(), payload_kind::message});
            expect(r.is_err() && r.unwrap_err().code == capability_errc::conflict, "a retired key is a conflict");
        });

        detail::capability_check<Probe>(report, "a digest that is not 32 bytes is refused", [](Probe& p, const auto& expect) {
            const std::string id = p.active_key();
            const std::array<std::byte, 31> short_digest{};
            const auto r = p.signer().sign({id, short_digest, payload_kind::digest});
            expect(r.is_err() && (r.unwrap_err().code == capability_errc::invalid_argument ||
                                  r.unwrap_err().code == capability_errc::unsupported),
                   "a 31-byte digest is invalid_argument (or unsupported, where digests are)");
        });

        detail::capability_check<Probe>(report, "a deterministic signer repeats itself", [](Probe& p, const auto& expect) {
            if (!p.deterministic())
                return;
            const std::string id = p.active_key();
            const auto a = p.signer().sign({id, kit_payload(), payload_kind::message});
            const auto b = p.signer().sign({id, kit_payload(), payload_kind::message});
            expect(a.is_ok() && b.is_ok() && *a == *b, "two signatures of one payload are equal");
        });

        detail::capability_check<Probe>(report, "public keys exist exactly for asymmetric algorithms", [](Probe& p, const auto& expect) {
            const std::string id = p.active_key();
            const auto s = p.signer().sign({id, kit_payload(), payload_kind::message});
            expect(s.is_ok(), "the active key signs");
            if (!s.is_ok())
                return;
            const auto k = p.signer().public_key_of({id, 0});
            if (is_asymmetric(s->algorithm)) {
                expect(k.is_ok(), "an asymmetric key publishes its public key");
                if (k.is_ok())
                    expect(k->algorithm == s->algorithm && !k->bytes.empty(), "the public key matches the algorithm");
            } else {
                expect(k.is_err() && k.unwrap_err().code == capability_errc::unsupported,
                       "a symmetric key has no public key: unsupported");
            }
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            const std::string id = p.active_key();
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto r = p.signer().sign({id, kit_payload(), payload_kind::message});
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_signer_conformance_hpp
