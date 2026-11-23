//
//  message_sender.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::message_sender`: anything that delivers one email or SMS and returns the provider's receipt.
/// \ingroup core
///
/// **Overview.** A message sender delivers one `message` over one `channel` and returns a `send_receipt` with the
/// provider's message id. Sign-in codes, notifications and alerts send through it; templates, codes and rate limits
/// sit above it, and a sender only delivers. This is the concept half of a pair; the pure virtual half is
/// `message_sender_interface` (`interfaces/message_sender.hpp`), with the adapter `message_sender_adapter<S>` and the
/// handle `message_sender_ref`.
///
/// **The contract** (every implementation; `interfaces/message_sender_conformance.hpp` checks it):
/// - `validate(m)` passes before anything is sent; an invalid message is `capability_errc::invalid_argument` and sends
///   nothing. An SMS goes to an E.164 number in its canonical form (`phone_number.hpp`), an email to a conservative
///   RFC 5321 address; a subject has no CR or LF; an idempotency key is at most 64 characters (`client_id`).
/// - A channel the sender does not support is `capability_errc::unsupported`, and `supports(channel)` says so in advance.
/// - The same non-empty `idempotency_key` with the same message returns the same receipt and sends once; with a
///   different message it is `capability_errc::conflict`.
/// - Nothing logged or returned contains the body of a message (it may hold a one-time code), the recipient or the
///   provider's response: errors follow the redaction rule of `capability_error.hpp`.
///
/// \snippet concepts/message_sender_demo.cpp concept
///
/// **Relationships.** `capability_error.hpp`, `capability_ids.hpp`, `phone_number.hpp`; the twin
/// `interfaces/message_sender.hpp`, the reference `interfaces/message_sender_reference.hpp`, the kit
/// `interfaces/message_sender_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `memory_message_sender` (`interfaces/message_sender_reference.hpp`, standard library only,
/// records an outbox; for tests). Above core: SMTP and SES email, Twilio SMS.
///
/// **Guarantees.** The validators are `constexpr`, `noexcept` and allocation-free. Thread safety is the
/// implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked: a sender is I/O bound.
///
/// **Tests and demos.** `tests/concepts/message_sender_test.cpp`, `demos/concepts/message_sender_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<cstdint>`, `<string_view>`, `capability_error.hpp`,
/// `capability_ids.hpp`, `phone_number.hpp`.
///
/// **Known issues.** The email check is deliberately conservative: quoted local parts and address literals are
/// refused.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_message_sender_hpp
#define snicholls_core_concepts_message_sender_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/phone_number.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace snicholls {

    /// \brief How a message travels.
    /// \ingroup core
    enum class channel : std::uint8_t {
        email = 1,  ///< to an email address
        sms = 2,    ///< to an E.164 number
    };

    /// \brief The longest SMS body accepted: ten concatenated GSM-7 segments of 160 characters.
    /// \ingroup core
    inline constexpr std::size_t max_sms_length = 1600;
    /// \brief The longest email address accepted (RFC 5321's path limit).
    /// \ingroup core
    inline constexpr std::size_t max_email_length = 254;
    /// \brief The longest subject accepted (RFC 5322's line limit).
    /// \ingroup core
    inline constexpr std::size_t max_subject_length = 998;

    /// \brief One message to send, borrowed for the call: a sender that keeps any of it (a queue, an outbox) copies it.
    /// \ingroup core
    struct message {
        channel via{channel::email};       ///< the channel
        std::string_view to;               ///< an email address, or an E.164 number in canonical form (`+61412345678`)
        std::string_view subject;          ///< email only; empty for SMS
        std::string_view text;             ///< the plain-text body; the SMS body
        std::string_view html;             ///< email only, optional: the HTML alternative
        std::string_view idempotency_key;  ///< optional, at most 64 characters: the same key sends once

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const message&, const message&) noexcept = default;
    };

    /// \brief What a sender returns once the provider accepted a message.
    /// \ingroup core
    struct send_receipt {
        channel via{channel::email};         ///< the channel used
        snicholls::provider_id message_id;   ///< the provider's id for the message

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const send_receipt&, const send_receipt&) noexcept = default;
    };

    /// \brief A conservative check of an email address: `local@domain`, one `@`, a dot inside the domain, no space,
    /// control character, bracket, quote or other special, no `..`, at most 254 characters.
    /// \ingroup core
    /// \param s The text.
    /// \return `true` when it may be used as an envelope recipient. CR and LF are always refused.
    [[nodiscard]] constexpr bool is_email_address(std::string_view s) noexcept
    {
        if (s.empty() || s.size() > max_email_length)
            return false;
        const auto at = s.find('@');
        if (at == 0 || at == std::string_view::npos || s.find('@', at + 1) != std::string_view::npos)
            return false;
        const auto domain = s.substr(at + 1);
        const auto dot = domain.rfind('.');
        if (domain.empty() || dot == std::string_view::npos || dot == 0 || dot + 1 == domain.size() || domain.front() == '-')
            return false;
        for (const char c : s) {
            const auto u = static_cast<unsigned char>(c);
            if (u <= 0x20 || u == 0x7f || c == '<' || c == '>' || c == '"' || c == '(' || c == ')' || c == ',' ||
                c == ';' || c == ':' || c == '\\' || c == '[' || c == ']')
                return false;
        }
        return domain.find("..") == std::string_view::npos && s.substr(0, at).find("..") == std::string_view::npos;
    }

    /// \brief Whether `s` contains a CR or LF, which would let text escape into a header.
    /// \ingroup core
    /// \param s The text.
    /// \return `true` when it does.
    [[nodiscard]] constexpr bool has_line_break(std::string_view s) noexcept
    {
        return s.find('\r') != std::string_view::npos || s.find('\n') != std::string_view::npos;
    }

    /// \brief Whether `s` is an E.164 number written canonically: `+` and digits, a country calling code assigned in
    /// E.164 and a valid length (`phone_number.hpp`).
    /// \ingroup core
    /// \param s The text.
    /// \return `true` when `parse_phone_number(s)` succeeds and writes `s` back unchanged.
    [[nodiscard]] constexpr bool is_canonical_e164(std::string_view s) noexcept
    {
        const auto p = parse_phone_number(s);
        return p.is_ok() && to_text(*p).view() == s;
    }

    /// \brief The checks every sender applies before sending.
    /// \ingroup core
    /// \param m The message.
    /// \return Success, or `invalid_argument` with a fixed message naming the field (never its value).
    [[nodiscard]] constexpr capability_result<void> validate(const message& m) noexcept
    {
        if (m.idempotency_key.size() > client_id::capacity())
            return fail(capability_errc::invalid_argument, {}, "idempotency key longer than 64 characters");
        switch (m.via) {
        case channel::email:
            if (!is_email_address(m.to))
                return fail(capability_errc::invalid_argument, {}, "recipient is not an email address");
            if (m.subject.empty() || m.subject.size() > max_subject_length || has_line_break(m.subject))
                return fail(capability_errc::invalid_argument, {}, "subject is empty, too long or has a line break");
            if (m.text.empty() && m.html.empty())
                return fail(capability_errc::invalid_argument, {}, "email has no body");
            return ok();
        case channel::sms:
            if (!is_canonical_e164(m.to))
                return fail(capability_errc::invalid_argument, {}, "recipient is not a canonical E.164 number");
            if (!m.subject.empty() || !m.html.empty())
                return fail(capability_errc::invalid_argument, {}, "an SMS has no subject and no HTML");
            if (m.text.empty() || m.text.size() > max_sms_length)
                return fail(capability_errc::invalid_argument, {}, "SMS body is empty or too long");
            return ok();
        }
        return fail(capability_errc::invalid_argument, {}, "unknown channel");
    }

    /// \brief A message sender: `s.send(m)` and `cs.supports(c)`.
    /// \ingroup core
    template <typename S>
    concept message_sender = requires(S& s, const S& cs, const message& m, channel c) {
        { s.send(m) } -> std::same_as<capability_result<send_receipt>>;
        { cs.supports(c) } -> std::same_as<bool>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_message_sender_hpp
