//
//  message_sender_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_message_sender_conformance`: the conformance kit of the `message_sender` contract.
/// \ingroup core
///
/// **Overview.** Drives any `message_sender`, in either form, through the contract of `concepts/message_sender.hpp`
/// and reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& sender()`: the implementation, a model of the concept (a sender, a `message_sender_interface&`, a
///   `message_sender_ref`);
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next send that would be accepted
///   fails with `code`, as though the provider had answered `response`.
///
/// A fresh probe is made for every check. The checks: `a valid message on a supported channel is accepted`, `an
/// unsupported channel is unsupported`, `an invalid message is refused without echoing it`, `the same idempotency
/// key sends once`, `a provider failure never leaks its response`.
///
/// \snippet interfaces/message_sender_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/message_sender.hpp`, `interfaces/message_sender.hpp`,
/// `interfaces/capability_conformance.hpp`; run on `memory_message_sender` in every form by
/// `tests/interfaces/message_sender_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/message_sender_conformance_test.cpp` (the kit catches broken senders),
/// `demos/interfaces/message_sender_conformance_demo.cpp`.
///
/// **Dependencies.** `<initializer_list>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `concepts/message_sender.hpp`,
/// `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** A sender that accepts nothing at all passes the acceptance check vacuously only if it supports
/// no channel; it then fails nothing else either, which `supports` makes visible.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_message_sender_conformance_hpp
#define snicholls_core_interfaces_message_sender_conformance_hpp

#include <snicholls/core/concepts/message_sender.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <initializer_list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        inline constexpr std::string_view kit_code = "482910";

        constexpr message kit_email(std::string_view key = {}) noexcept
        {
            return {channel::email, "someone@example.test", "Your code", "Your code is 482910.", "", key};
        }

        constexpr message kit_sms(std::string_view key = {}) noexcept
        {
            return {channel::sms, "+61400000000", "", "Your code is 482910.", "", key};
        }

        template <typename Probe>
        concept message_sender_probe = std::is_default_constructible_v<Probe> &&
            message_sender<std::remove_reference_t<decltype(std::declval<Probe&>().sender())>> &&
            requires(Probe& p, capability_errc c, std::string_view r) { p.provider_fails(c, r); };
    } // namespace detail

    /// \brief Runs every check of the `message_sender` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::message_sender_probe<Probe>
    [[nodiscard]] capability_conformance_report run_message_sender_conformance()
    {
        capability_conformance_report report;
        using detail::kit_email;
        using detail::kit_sms;

        detail::capability_check<Probe>(report, "a valid message on a supported channel is accepted", [](Probe& p, const auto& expect) {
            auto& s = p.sender();
            expect(s.supports(channel::email) || s.supports(channel::sms), "supports at least one channel");
            for (const message& m : {kit_email(), kit_sms()}) {
                if (!s.supports(m.via))
                    continue;
                const auto r = s.send(m);
                expect(r.is_ok(), "a valid message is accepted");
                if (r.is_ok()) {
                    expect(r->via == m.via, "the receipt names the channel");
                    expect(!r->message_id.empty(), "the receipt has the provider's message id");
                }
            }
        });

        detail::capability_check<Probe>(report, "an unsupported channel is unsupported", [](Probe& p, const auto& expect) {
            auto& s = p.sender();
            for (const message& m : {kit_email(), kit_sms()})
                if (!s.supports(m.via)) {
                    const auto r = s.send(m);
                    expect(r.is_err() && r.unwrap_err().code == capability_errc::unsupported,
                           "a send on an unsupported channel is unsupported");
                }
        });

        detail::capability_check<Probe>(report, "an invalid message is refused without echoing it", [](Probe& p, const auto& expect) {
            auto& s = p.sender();
            message bad_address = kit_email();
            bad_address.to = "not an address";
            message injected = kit_email();
            injected.subject = "Hi\r\nBcc: everyone@example.test";
            message sms_subject = kit_sms();
            sms_subject.subject = "no subjects in SMS";
            message sms_not_e164 = kit_sms();
            sms_not_e164.to = "0400000000";
            message sms_spaced = kit_sms();
            sms_spaced.to = "+61 400 000 000";
            const std::string too_long(max_sms_length + 1, '7');
            message sms_too_long = kit_sms();
            sms_too_long.text = too_long;
            const std::string long_key(client_id::capacity() + 1, 'k');
            message key_too_long = s.supports(channel::email) ? kit_email(long_key) : kit_sms(long_key);
            for (const message& m : {bad_address, injected, sms_subject, sms_not_e164, sms_spaced, sms_too_long, key_too_long}) {
                if (!s.supports(m.via))
                    continue;
                const auto r = s.send(m);
                expect(r.is_err() && r.unwrap_err().code == capability_errc::invalid_argument,
                       "an invalid message is invalid_argument");
                if (r.is_err()) {
                    expect(!capability_error_leaks(r.unwrap_err(), detail::kit_code), "the error does not echo the body");
                    expect(!capability_error_leaks(r.unwrap_err(), "example.test"), "the error does not echo the recipient");
                }
            }
        });

        detail::capability_check<Probe>(report, "the same idempotency key sends once", [](Probe& p, const auto& expect) {
            auto& s = p.sender();
            const message m = s.supports(channel::email) ? kit_email("k-1") : kit_sms("k-1");
            const auto first = s.send(m);
            const auto second = s.send(m);
            expect(first.is_ok() && second.is_ok(), "both sends answer");
            if (first.is_ok() && second.is_ok())
                expect(*first == *second, "a repeat returns the first receipt");
            message other = m;
            other.text = "A different body.";
            const auto third = s.send(other);
            expect(third.is_err() && third.unwrap_err().code == capability_errc::conflict,
                   "the key with another message is a conflict");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            auto& s = p.sender();
            const message m = s.supports(channel::email) ? kit_email() : kit_sms();
            const auto r = s.send(m);
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_message_sender_conformance_hpp
