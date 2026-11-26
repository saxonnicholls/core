//
//  encryptor_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_encryptor_conformance`: the conformance kit of the `encryptor` contract.
/// \ingroup core
///
/// **Overview.** Drives any `encryptor`, in either form, through the contract of `concepts/encryptor.hpp` and reports
/// which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type with
/// - `auto& encryptor()`: the implementation (an encryptor, an `encryptor_interface&`, an `encryptor_ref`), holding
///   two keys;
/// - `std::string_view key()` and `std::string_view other_key()`: the ids of those keys;
/// - `void forget(std::string_view key_id)`: the implementation no longer holds that key.
///
/// A fresh probe is made for every check. The checks: `what is sealed opens`, `every seal is fresh`, `a changed
/// envelope does not open`, `other associated data does not open`, `an unknown key is not_found`, `a malformed
/// envelope is invalid_argument`, `the key id is in the envelope`, `errors carry no secret`.
///
/// \snippet interfaces/encryptor_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/encryptor.hpp`, `interfaces/encryptor.hpp`, `interfaces/capability_conformance.hpp`;
/// run on `insecure_test_encryptor` in every form by `tests/interfaces/encryptor_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit (a few hundred seals; each byte of one envelope is flipped).
///
/// **Tests and demos.** `tests/interfaces/encryptor_conformance_test.cpp` (the kit catches broken encryptors),
/// `demos/interfaces/encryptor_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<span>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`,
/// `concepts/encryptor.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** The kit tests the contract, never the cipher's strength: the test double passes it.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_encryptor_conformance_hpp
#define snicholls_core_interfaces_encryptor_conformance_hpp

#include <snicholls/core/concepts/encryptor.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept encryptor_probe = std::is_default_constructible_v<Probe> &&
            encryptor<std::remove_reference_t<decltype(std::declval<Probe&>().encryptor())>> && requires(Probe& p) {
                { p.key() } -> std::convertible_to<std::string_view>;
                { p.other_key() } -> std::convertible_to<std::string_view>;
                p.forget(std::string_view{});
            };

        inline std::vector<std::byte> kit_bytes(std::string_view s)
        {
            std::vector<std::byte> out;
            for (const char c : s)
                out.push_back(static_cast<std::byte>(c));
            return out;
        }

        inline bool kit_contains(std::span<const std::byte> hay, std::span<const std::byte> needle)
        {
            if (needle.empty() || hay.size() < needle.size())
                return false;
            for (std::size_t i = 0; i + needle.size() <= hay.size(); ++i) {
                bool same = true;
                for (std::size_t j = 0; j < needle.size() && same; ++j)
                    same = hay[i + j] == needle[j];
                if (same)
                    return true;
            }
            return false;
        }
    } // namespace detail

    /// \brief Runs every check of the `encryptor` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::encryptor_probe<Probe>
    [[nodiscard]] capability_conformance_report run_encryptor_conformance()
    {
        capability_conformance_report report;
        using detail::kit_bytes;
        static const std::vector<std::byte> secret = kit_bytes("card 4111 1111 1111 1111, " + std::string(capability_redaction_marker));
        static const std::vector<std::byte> aad = kit_bytes("users/42/card");

        detail::capability_check<Probe>(report, "what is sealed opens", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const std::vector<std::byte> large(10000, std::byte{0x5a});
            for (const auto* plain : {&secret, &large}) {
                const auto sealed = e.encrypt(p.key(), *plain, aad);
                expect(sealed.is_ok(), "encrypt succeeds");
                if (!sealed)
                    return;
                const auto back = e.decrypt(*sealed, aad);
                expect(back.is_ok() && *back == *plain, "decrypt returns the plaintext");
            }
            const auto empty = e.encrypt(p.key(), {}, {});
            expect(empty.is_ok() && e.decrypt(*empty, {}).is_ok() && e.decrypt(*empty, {})->empty(), "the empty plaintext round-trips");
            expect(!e.algorithm().empty(), "the algorithm has a name");
        });

        detail::capability_check<Probe>(report, "every seal is fresh", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const auto a = e.encrypt(p.key(), secret, aad);
            const auto b = e.encrypt(p.key(), secret, aad);
            expect(a.is_ok() && b.is_ok() && *a != *b, "two seals of the same plaintext differ");
            if (a.is_ok())
                expect(!detail::kit_contains(*a, secret), "the envelope does not contain the plaintext");
        });

        detail::capability_check<Probe>(report, "a changed envelope does not open", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const auto sealed = e.encrypt(p.key(), secret, aad);
            if (!sealed) {
                expect(false, "encrypt succeeds");
                return;
            }
            const std::size_t header = 2 + p.key().size();
            bool all_refused = true, none_leaked = true;
            for (std::size_t i = header; i < sealed->size(); ++i) {   // every byte after the key id
                std::vector<std::byte> changed = *sealed;
                changed[i] ^= std::byte{0x01};
                const auto r = e.decrypt(changed, aad);
                all_refused = all_refused && r.is_err() && r.unwrap_err().code == capability_errc::signature_invalid;
                none_leaked = none_leaked && r.is_err();
            }
            expect(all_refused, "any flipped bit is signature_invalid");
            expect(none_leaked, "no plaintext comes back from a changed envelope");
            std::vector<std::byte> shorter = *sealed;
            shorter.pop_back();
            const auto r = e.decrypt(shorter, aad);
            expect(r.is_err(), "a truncated envelope does not open");
        });

        detail::capability_check<Probe>(report, "other associated data does not open", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const auto sealed = e.encrypt(p.key(), secret, aad);
            if (!sealed) {
                expect(false, "encrypt succeeds");
                return;
            }
            const auto other = e.decrypt(*sealed, kit_bytes("users/43/card"));
            expect(other.is_err() && other.unwrap_err().code == capability_errc::signature_invalid, "another row's aad is refused");
            const auto none = e.decrypt(*sealed, {});
            expect(none.is_err() && none.unwrap_err().code == capability_errc::signature_invalid, "no aad is refused");
        });

        detail::capability_check<Probe>(report, "an unknown key is not_found", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const auto unknown = e.encrypt("no-such-key", secret, aad);
            expect(unknown.is_err() && unknown.unwrap_err().code == capability_errc::not_found, "encrypt under an unknown key");
            const auto sealed = e.encrypt(p.other_key(), secret, aad);
            expect(sealed.is_ok(), "encrypt under the other key");
            const std::string other{p.other_key()};
            p.forget(other);
            if (sealed) {
                const auto r = p.encryptor().decrypt(*sealed, aad);
                expect(r.is_err() && r.unwrap_err().code == capability_errc::not_found, "decrypt after the key is gone");
            }
            const auto still = p.encryptor().encrypt(p.key(), secret, aad);
            expect(still.is_ok(), "the remaining key still seals");
        });

        detail::capability_check<Probe>(report, "a malformed envelope is invalid_argument", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const std::vector<std::byte> empty;
            const std::vector<std::byte> wrong_format{std::byte{9}, std::byte{1}, std::byte{'k'}};
            std::vector<std::byte> bad_id{sealed_format_v1, std::byte{3}, std::byte{'a'}, std::byte{' '}, std::byte{'b'}};
            bad_id.resize(64, std::byte{0});
            const std::vector<std::byte>* cases[] = {&empty, &wrong_format, &bad_id};
            for (const auto* m : cases) {
                const auto r = e.decrypt(*m, aad);
                expect(r.is_err() && r.unwrap_err().code == capability_errc::invalid_argument, "a malformed envelope");
            }
            const auto bad_key = e.encrypt("has space", secret, aad);
            expect(bad_key.is_err() && bad_key.unwrap_err().code == capability_errc::invalid_argument, "a key id that is not one");
        });

        detail::capability_check<Probe>(report, "the key id is in the envelope", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const auto a = e.encrypt(p.key(), secret, aad);
            const auto b = e.encrypt(p.other_key(), secret, aad);
            expect(a.is_ok() && b.is_ok(), "both keys seal");
            if (a.is_ok() && b.is_ok()) {
                const auto ida = sealed_key_id(*a);
                const auto idb = sealed_key_id(*b);
                expect(ida.is_ok() && *ida == p.key(), "sealed_key_id reads the first key");
                expect(idb.is_ok() && *idb == p.other_key(), "sealed_key_id reads the second key");
                expect(e.decrypt(*a, aad).is_ok() && e.decrypt(*b, aad).is_ok(), "both open: rotation keeps old keys");
            }
        });

        detail::capability_check<Probe>(report, "errors carry no secret", [](Probe& p, const auto& expect) {
            auto& e = p.encryptor();
            const auto sealed = e.encrypt(p.key(), secret, aad);
            if (!sealed) {
                expect(false, "encrypt succeeds");
                return;
            }
            std::vector<std::byte> changed = *sealed;
            changed.back() ^= std::byte{0x80};
            const auto r = e.decrypt(changed, aad);
            const auto u = e.encrypt("no-such-key", secret, aad);
            for (const auto* res : {&r, &u})
                if (res->is_err())
                    expect(!capability_error_leaks(res->unwrap_err(), capability_redaction_marker), "no plaintext in the error");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_encryptor_conformance_hpp
