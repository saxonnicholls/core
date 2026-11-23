//
//  message_sender_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_message_sender`: the standard-library reference `message_sender`; it validates,
/// deduplicates and records an outbox, and sends nothing.
/// \ingroup core
///
/// **Overview.** What a product's tests send email and SMS through: it applies the contract exactly (`validate`,
/// idempotency, channel support) and keeps every accepted message in an outbox the test reads, so no network, mail
/// server or SMS provider is needed. `fail_next(code, response)` makes the next send that would be accepted fail as
/// though the provider had answered `response`, to test a caller's retry path; the error keeps the response's first
/// word as the provider code when it is one (`provider_error`) and nothing else of it.
///
/// \snippet interfaces/message_sender_reference_demo.cpp reference
///
/// **Relationships.** Models `message_sender` (`concepts/message_sender.hpp`); passes
/// `interfaces/message_sender_conformance.hpp` directly and through `interfaces/message_sender.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One; email and SMS can each be switched off.
///
/// **Guarantees.** Not thread-safe (one per test). Strong exception safety: a throwing allocation leaves the outbox
/// unchanged. Message ids are deterministic (`memory-email-1`, `memory-sms-2`). Errors follow the redaction rule.
///
/// **Errors.** As the contract, and whatever `fail_next` armed.
///
/// **Performance.** Not benchmarked: a test double.
///
/// **Tests and demos.** `tests/interfaces/message_sender_reference_test.cpp`,
/// `demos/interfaces/message_sender_reference_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<map>`, `<optional>`, `<string>`, `<string_view>`, `<vector>`,
/// `interfaces/message_sender.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_message_sender_reference_hpp
#define snicholls_core_interfaces_message_sender_reference_hpp

#include <snicholls/core/interfaces/message_sender.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief The reference message sender.
    /// \ingroup core
    class memory_message_sender {
    public:
        /// \brief A message as the outbox keeps it: an owned copy of the borrowed `message`.
        struct kept_message {
            channel via{channel::email};  ///< the channel
            std::string to;               ///< the recipient
            std::string subject;          ///< the subject
            std::string text;             ///< the text body
            std::string html;             ///< the HTML body
            std::string idempotency_key;  ///< the key

            /// \brief A copy of `m`.
            /// \param m The borrowed message.
            /// \return The owned copy.
            [[nodiscard]] static kept_message of(const message& m)
            {
                return {m.via, std::string(m.to), std::string(m.subject), std::string(m.text), std::string(m.html),
                        std::string(m.idempotency_key)};
            }
            /// \brief A borrowed view of this copy.
            /// \return The message, valid while this copy lives.
            [[nodiscard]] message view() const noexcept { return {via, to, subject, text, html, idempotency_key}; }

            /// \brief Equality.
            friend bool operator==(const kept_message&, const kept_message&) = default;
        };

        /// \brief One accepted message and its receipt.
        struct sent {
            kept_message msg;      ///< as given, copied
            send_receipt receipt;  ///< as returned

            /// \brief Equality.
            friend bool operator==(const sent&, const sent&) = default;
        };

        /// \brief A sender for both channels.
        memory_message_sender() = default;

        /// \brief A sender for some channels only.
        /// \param email Whether email is supported.
        /// \param sms   Whether SMS is supported.
        memory_message_sender(bool email, bool sms) noexcept : email_(email), sms_(sms) {}

        /// \brief Sends (records) one message, applying the contract.
        /// \param m The message.
        /// \return The receipt, or the error.
        capability_result<send_receipt> send(const message& m)
        {
            if (!supports(m.via))
                return fail(capability_errc::unsupported);
            if (auto valid = validate(m); !valid)
                return err(valid.unwrap_err());
            if (!m.idempotency_key.empty()) {
                if (const auto it = by_key_.find(m.idempotency_key); it != by_key_.end()) {
                    const sent& earlier = outbox_[it->second];
                    if (earlier.msg.view() != m)
                        return fail(capability_errc::conflict, {}, "idempotency key reused for another message");
                    return ok(earlier.receipt);
                }
            }
            if (injected_) {
                const capability_error e = *injected_;
                injected_.reset();
                return err(e);
            }
            const std::string id = std::string(m.via == channel::email ? "memory-email-" : "memory-sms-") +
                                   std::to_string(outbox_.size() + 1);
            const send_receipt receipt{m.via, *provider_id::make(id)};
            outbox_.push_back({kept_message::of(m), receipt});
            if (!m.idempotency_key.empty()) {
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
                try {
                    by_key_.emplace(std::string(m.idempotency_key), outbox_.size() - 1);
                } catch (...) {
                    outbox_.pop_back();
                    throw;
                }
#else
                by_key_.emplace(std::string(m.idempotency_key), outbox_.size() - 1);
#endif
            }
            return ok(receipt);
        }

        /// \brief Whether the channel is supported.
        /// \param c The channel.
        /// \return As constructed.
        [[nodiscard]] bool supports(channel c) const noexcept { return c == channel::email ? email_ : sms_; }

        /// \brief Every accepted message, in order.
        /// \return The outbox.
        [[nodiscard]] const std::vector<sent>& outbox() const noexcept { return outbox_; }

        /// \brief Makes the next send that would be accepted fail (once) as though the provider answered `response`.
        /// \param code     The error's code.
        /// \param response The provider's response: its first word becomes the provider code when it is one, and
        ///                 nothing else of it is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        bool email_{true};
        bool sms_{true};
        std::vector<sent> outbox_;
        std::map<std::string, std::size_t, std::less<>> by_key_;
        std::optional<capability_error> injected_;
    };

    static_assert(message_sender<memory_message_sender>, "snicholls::memory_message_sender must model snicholls::message_sender");

} // namespace snicholls

#endif // snicholls_core_interfaces_message_sender_reference_hpp
