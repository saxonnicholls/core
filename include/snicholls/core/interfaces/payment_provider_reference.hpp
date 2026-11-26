//
//  payment_provider_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_payment_provider`: the standard-library reference `payment_provider`; pull (card) and push
/// (bank transfer by reference) collections, payouts, refunds, quotes, credits, signed webhooks and facts, in memory.
/// \ingroup core
///
/// **Overview.** It plays both shapes the contract was made for, so a product's tests cover the pull path and the push
/// path without a network:
/// - a **card** collection returns a redirect and waits; the test calls `complete` (the payer paid) or `decline`;
/// - a **bank transfer** collection returns bank details and a reference and waits; the test calls `receive_credit`
///   (money arrived quoting a reference), which completes the matching collection, or leaves an unmatched credit for
///   the caller's suspense handling.
///
/// Webhooks are signed the way Stripe signs them: a `Memory-Signature: t=<unix seconds>,v1=<hex>` header, the tag
/// HMAC-SHA256 (core's `hmac<sha256>`) under the webhook secret over `<t>.<raw body>`, several `v1` accepted during a
/// secret rotation, and a 300-second window. `signed_webhook` makes a delivery for a test; `verify_webhook` checks one
/// in constant time and fails closed. The body is `key=value` lines (`source`, `provider_account`, `id`, `fact_id`,
/// `kind`, `type`, `payment_id`, `our_reference`, `reference`, `occurred_at` in unix seconds, `needs_lookup`), the kind
/// by its enumerator's name: no amount, because amounts come only from `resolve` and `statement`, which read the
/// balance transactions it recorded. `fail_next(code, response)` makes the next `collect`, `payout` or `refund` fail as
/// though the provider had answered `response`.
///
/// \snippet interfaces/payment_provider_reference_demo.cpp reference
///
/// **Relationships.** Models `payment_provider` (`concepts/payment_provider.hpp`); passes
/// `interfaces/payment_provider_conformance.hpp` in every form. Group: \ref core.
///
/// **Implementations / variants.** Card and bank transfer; bank debit is unsupported.
/// `basic_memory_payment_provider<Clock>` reads any `utc_clock` (`time/utc_clock.hpp`), owned by value and reached through
/// `clock()`; `memory_payment_provider` is `basic_memory_payment_provider<manual_utc_clock>`.
///
/// **Guarantees.** Not thread-safe. Ids are deterministic (`mem_pay_1`, `mem_cr_2`, ...). Amounts are exact. Errors
/// follow the redaction rule: a decline keeps the provider's code only.
///
/// **Errors.** As the contract, and whatever `fail_next` armed.
///
/// **Performance.** `verify_webhook`: `benchmarks/interfaces/payment_provider_reference_benchmark.cpp`, against the
/// same HMAC check written by hand. Not yet quoted from a quiet machine: `make
/// benchmarks` records them in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/interfaces/payment_provider_reference_test.cpp`,
/// `demos/interfaces/payment_provider_reference_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<chrono>`, `<cstdint>`, `<functional>`, `<map>`, `<optional>`, `<set>`,
/// `<string>`, `<string_view>`, `<utility>`, `<vector>`, `detail/memory_webhook.hpp` (HMAC-SHA256, shared with
/// `memory_security_provider`), `secret.hpp`, `interfaces/payment_provider.hpp`, `time/utc_clock.hpp`.
///
/// **Known issues.** Bank debit (mandates) is unsupported.
///
/// **Status.** New in 0.23.0; the clock a `utc_clock` template parameter, owned, since 0.27.0 (it was a
/// `std::function<utc_s()>`).

#ifndef snicholls_core_interfaces_payment_provider_reference_hpp
#define snicholls_core_interfaces_payment_provider_reference_hpp

#include <snicholls/core/detail/memory_webhook.hpp>
#include <snicholls/core/interfaces/payment_provider.hpp>
#include <snicholls/core/secret.hpp>
#include <snicholls/core/time/utc_clock.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference payment provider.
    /// \ingroup core
    /// \tparam Clock A `utc_clock` (`time/utc_clock.hpp`), owned by value; `manual_utc_clock` by default, moved by
    ///               `clock().advance(d)`.
    template <utc_clock Clock = manual_utc_clock>
    class basic_memory_payment_provider {
    public:
        /// \brief The clock.
        using clock_type = Clock;
        /// \brief The webhook time window, either side of `received_at`.
        static constexpr std::chrono::seconds webhook_tolerance = detail::memory_webhook::tolerance;
        /// \brief How long a quote lives.
        static constexpr std::chrono::seconds quote_lifetime{60};
        /// \brief The signature header.
        static constexpr std::string_view signature_header = detail::memory_webhook::header;

        /// \brief A provider for `currencies`, signing webhooks with `webhook_secret`.
        /// \param currencies     The currencies it collects and holds.
        /// \param webhook_secret The HMAC key for webhooks (never logged).
        /// \param clock          The clock.
        basic_memory_payment_provider(std::vector<asset> currencies, secret<std::string> webhook_secret, Clock clock)
            : currencies_(std::move(currencies)), secret_(std::move(webhook_secret)), clock_(std::move(clock))
        {
        }

        /// \brief The clock, to move a manual one: `clock().advance(d)`.
        /// \return The clock.
        [[nodiscard]] Clock& clock() noexcept { return clock_; }
        /// \brief The clock.
        /// \return The clock.
        [[nodiscard]] const Clock& clock() const noexcept { return clock_; }

        /// \brief What it does: card and bank transfer; payouts, quotes, refunds and credits.
        /// \return The capabilities.
        [[nodiscard]] provider_capabilities capabilities() const
        {
            return {{payment_method::card, payment_method::bank_transfer}, currencies_, true, true, true, true, provider_account_};
        }

        /// \brief Starts a collection.
        /// \param c The request.
        /// \return The payment and the payer's next step, or the error.
        capability_result<collection> collect(const collection_request& c)
        {
            if (auto bad = check_request(c.idempotency_key, c.our_reference, c.amount))
                return err(*bad);
            if (const auto it = collect_keys_.find(c.idempotency_key); it != collect_keys_.end()) {
                if (it->second.first != fingerprint(c))
                    return fail(capability_errc::conflict, {}, "idempotency key reused for another request");
                return ok(it->second.second);
            }
            if (c.method != payment_method::card && c.method != payment_method::bank_transfer)
                return fail(capability_errc::unsupported, {}, "card and bank transfer only");
            if (auto e = take_injected())
                return err(*e);
            collection out;
            out.pay = new_payment("mem_pay_", movement::collection, c.our_reference, c.method, c.amount);
            if (c.method == payment_method::card) {
                out.pay.status = payment_status::requires_action;
                out.next.kind = next_step_kind::redirect;
                out.next.url = "memory://checkout/" + std::string(out.pay.payment_id.view());
            } else {
                auto reference = c.payer_reference.empty() ? make_id<short_text>("MEM-" + std::to_string(++next_reference_))
                                                           : make_id<short_text>(c.payer_reference);
                if (!reference)
                    return err(reference.unwrap_err());
                out.next.kind = next_step_kind::bank_transfer;
                out.next.bank = bank_details{"Memory Payments", "Memory Bank", "000000001", "000-000", "MEMOXXXX", {}, *reference};
                awaiting_credit_.emplace(std::string(reference->view()), std::string(out.pay.payment_id.view()));
            }
            payments_.insert_or_assign(std::string(out.pay.payment_id.view()), out.pay);
            collect_keys_.emplace(std::string(c.idempotency_key), std::make_pair(fingerprint(c), out));
            return ok(std::move(out));
        }

        /// \brief Pays a registered beneficiary from the balance.
        /// \param o The request.
        /// \return The payout (succeeded at once), or the error.
        capability_result<payment> payout(const payout_request& o)
        {
            if (auto bad = check_request(o.idempotency_key, o.our_reference, o.amount))
                return err(*bad);
            if (const auto it = payout_keys_.find(o.idempotency_key); it != payout_keys_.end()) {
                if (it->second.first != fingerprint(o))
                    return fail(capability_errc::conflict, {}, "idempotency key reused for another request");
                return ok(it->second.second);
            }
            if (auto e = take_injected())
                return err(*e);
            if (!beneficiaries_.contains(o.beneficiary_id))
                return fail(capability_errc::not_found, {}, "no such beneficiary");
            if (o.quote_id) {
                const auto q = quotes_.find(*o.quote_id);
                if (q == quotes_.end())
                    return fail(capability_errc::not_found, {}, "no such quote");
                if (stamp() > q->second.expires_at)
                    return fail(capability_errc::stale, {}, "the quote expired");
            }
            if (auto taken = take(o.amount); !taken)
                return err(taken.unwrap_err());
            payment p = new_payment("mem_out_", movement::payout, o.our_reference, payment_method::bank_transfer, o.amount);
            p.status = payment_status::succeeded;
            p.received = o.amount;
            payments_.insert_or_assign(std::string(p.payment_id.view()), p);
            record(std::string(p.payment_id.view()), fact_kind::payout, -o.amount, money::zero(o.amount.asset_of()),
                   fact_refs{{}, {}, {}, {}, p.payment_id});
            payout_keys_.emplace(std::string(o.idempotency_key), std::make_pair(fingerprint(o), p));
            return ok(std::move(p));
        }

        /// \brief Refunds a succeeded collection, in whole or in part.
        /// \param r The request.
        /// \return The refund, or the error.
        capability_result<payment> refund(const refund_request& r)
        {
            if (r.idempotency_key.empty() || r.idempotency_key.size() > client_id::capacity())
                return fail(capability_errc::invalid_argument, {}, "an idempotency key of 1 to 64 characters");
            if (const auto it = refund_keys_.find(r.idempotency_key); it != refund_keys_.end()) {
                if (it->second.first != fingerprint(r))
                    return fail(capability_errc::conflict, {}, "idempotency key reused for another request");
                return ok(it->second.second);
            }
            if (auto e = take_injected())
                return err(*e);
            const auto found = payments_.find(r.payment_id);
            if (found == payments_.end() || found->second.kind != movement::collection)
                return fail(capability_errc::not_found);
            payment& original = found->second;
            if (original.status != payment_status::succeeded && original.status != payment_status::partially_refunded)
                return fail(capability_errc::conflict, {}, "only a succeeded collection is refunded");
            const auto remaining = subtract(original.received, original.refunded);
            if (!remaining)
                return fail(capability_errc::internal);
            const money amount = r.amount ? *r.amount : *remaining;
            if (amount.asset_of() != original.received.asset_of())
                return fail(capability_errc::invalid_argument, {}, "a refund is in the collection's currency");
            if (amount.is_negative() || amount.is_zero())
                return fail(capability_errc::invalid_argument, {}, "an amount is positive");
            if (amount.magnitude() > remaining->magnitude())
                return fail(capability_errc::conflict, {}, "more than remains to refund");
            if (auto taken = take(amount); !taken)
                return err(taken.unwrap_err());
            original.refunded = *add(original.refunded, amount);
            original.status = original.refunded == original.received ? payment_status::refunded : payment_status::partially_refunded;
            original.updated_at = stamp();
            payment p = new_payment("mem_ref_", movement::refund, original.our_reference.view(), original.method, amount);
            p.status = payment_status::succeeded;
            p.received = amount;
            payments_.insert_or_assign(std::string(p.payment_id.view()), p);
            record(std::string(p.payment_id.view()), fact_kind::refund, -amount, money::zero(amount.asset_of()),
                   fact_refs{original.payment_id, {}, p.payment_id, {}, {}});
            refund_keys_.emplace(std::string(r.idempotency_key), std::make_pair(fingerprint(r), p));
            return ok(std::move(p));
        }

        /// \brief Quotes a conversion at a rate set with `set_rate`, less a fee of `set_fee_bps` on the source.
        /// \param q The request.
        /// \return The quote, or the error.
        capability_result<fx_quote> quote(const quote_request& q)
        {
            if (q.amount.is_negative() || q.amount.is_zero())
                return fail(capability_errc::invalid_argument, {}, "an amount is positive");
            const asset source_unit = q.amount_is_source ? q.amount.asset_of() : q.other;
            const asset target_unit = q.amount_is_source ? q.other : q.amount.asset_of();
            const auto r = rates_.find(key(source_unit, target_unit));
            if (r == rates_.end())
                return fail(capability_errc::unsupported, {}, "no rate for that pair");
            money source, target, fee;
            if (q.amount_is_source) {
                source = q.amount;
                fee = fee_on(q.amount);
                const auto net = subtract(q.amount, fee);
                const auto converted = net ? convert(*net, r->second, rounding::toward_zero)
                                           : result<conversion, money_errc>(err(money_errc::overflow));
                if (!converted)
                    return fail(capability_errc::invalid_argument, {}, "the amount cannot be converted");
                target = converted->converted;
            } else {
                target = q.amount;
                const auto back = r->second.inverse();
                const auto net = back ? convert(q.amount, *back, rounding::ceiling)
                                      : result<conversion, money_errc>(err(money_errc::overflow));
                if (!net)
                    return fail(capability_errc::invalid_argument, {}, "the amount cannot be converted");
                fee = fee_on(net->converted);
                source = *add(net->converted, fee);
            }
            const fx_quote out{id<provider_id>("mem_q_" + std::to_string(++next_id_)), source, target, fee, r->second,
                               stamp() + quote_lifetime};
            quotes_.insert_or_assign(std::string(out.quote_id.view()), out);
            return ok(out);
        }

        /// \brief The current state of a movement.
        /// \param payment_id The id.
        /// \return The movement, or `not_found`.
        capability_result<payment> status(std::string_view payment_id)
        {
            const auto it = payments_.find(payment_id);
            if (it == payments_.end())
                return fail(capability_errc::not_found);
            return ok(it->second);
        }

        /// \brief The balance in each currency that has one, by code.
        /// \return The balances.
        capability_result<std::vector<money>> balances()
        {
            std::vector<money> out;
            for (const auto& [code, m] : balances_)
                out.push_back(m);
            return ok(std::move(out));
        }

        /// \brief Credits in `[from, to)` in one currency, oldest first.
        /// \param q The window.
        /// \return The credits.
        capability_result<std::vector<credit>> credits(const credit_query& q)
        {
            std::vector<credit> out;
            for (const credit& c : credits_)
                if (c.net.asset_of() == q.currency && c.occurred_at >= q.from && c.occurred_at < q.to)
                    out.push_back(c);
            return ok(std::move(out));
        }

        /// \brief Authenticates (HMAC-SHA256, constant time) and parses a delivery; fails closed; no I/O.
        /// \param w The delivery.
        /// \return The event, or `signature_invalid`, `stale` or `invalid_argument`.
        [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const
        {
            return detail::memory_webhook::verify(secret_, w);
        }

        /// \brief The facts behind a verified event, from the balance transactions recorded (never from the event's
        /// body): the payment's, payout's or refund's for `payment_id`, the credit's for `fact_id`.
        /// \param e The event.
        /// \return The facts, oldest first; none for an event that books nothing.
        capability_result<std::vector<payment_fact>> resolve(const webhook_event& e)
        {
            std::vector<payment_fact> out;
            if (e.kind == event_kind::other || e.is_test)   // a test delivery is recorded, never booked
                return ok(std::move(out));
            for (const auto& [related, f] : ledger_)
                if ((!e.payment_id.empty() && related == e.payment_id.view()) ||
                    (e.kind == event_kind::credit_received && !e.fact_id.empty() && f.fact_id == e.fact_id)) {
                    payment_fact copy = f;
                    copy.event_id = e.event_id;
                    out.push_back(copy);
                }
            return ok(std::move(out));
        }

        /// \brief A page of balance transactions in one currency (optionally one payout's), as facts.
        /// \param q The page; the cursor is the index of the next line.
        /// \return The page, or `invalid_argument` for a bad cursor.
        capability_result<statement_page> statement(const statement_query& q)
        {
            std::size_t from = 0;
            if (!q.cursor.empty()) {
                std::int64_t at = 0;
                if (!detail::memory_webhook::parse_int(q.cursor, at) || at < 0)
                    return fail(capability_errc::invalid_argument, {}, "not a cursor this provider gave");
                from = static_cast<std::size_t>(at);
            }
            statement_page page;
            std::size_t i = from;
            for (; i < ledger_.size() && page.facts.size() < q.limit; ++i) {
                const payment_fact& f = ledger_[i].second;
                if (f.settled.gross.asset_of() == q.currency && (q.payout_id.empty() || f.refs.payout == q.payout_id))
                    page.facts.push_back(f);
            }
            if (i < ledger_.size())
                page.next_cursor = id<provider_id>(std::to_string(i));
            return ok(std::move(page));
        }

        /// \brief The payer completed a card collection: it succeeds, net of the fee, into the balance.
        /// \param payment_id The collection.
        /// \return The payment, or `not_found` or `conflict`.
        capability_result<payment> complete(std::string_view payment_id)
        {
            const auto it = payments_.find(payment_id);
            if (it == payments_.end() || it->second.kind != movement::collection)
                return fail(capability_errc::not_found);
            if (is_terminal(it->second.status))
                return fail(capability_errc::conflict, {}, "the collection is final");
            settle(it->second, it->second.amount);
            return ok(it->second);
        }

        /// \brief The card was declined.
        /// \param payment_id The collection.
        /// \param code       The provider's decline code.
        /// \return The payment, or `not_found`.
        capability_result<payment> decline(std::string_view payment_id, provider_code code)
        {
            const auto it = payments_.find(payment_id);
            if (it == payments_.end() || it->second.kind != movement::collection)
                return fail(capability_errc::not_found);
            it->second.status = payment_status::failed;
            it->second.failure_code = code;
            it->second.updated_at = stamp();
            return ok(it->second);
        }

        /// \brief Money arrived by bank transfer quoting `reference`. It completes the waiting collection with that
        /// reference and currency (with whatever amount arrived); otherwise it stays an unmatched credit.
        /// \param net       The amount credited.
        /// \param reference What the payer quoted.
        /// \param sender    The payer's name.
        /// \return The credit, or `invalid_argument`.
        capability_result<credit> receive_credit(const money& net, std::string_view reference, std::string_view sender)
        {
            if (net.is_negative() || net.is_zero())
                return fail(capability_errc::invalid_argument, {}, "a credit is positive");
            auto ref = make_id<short_text>(reference);
            auto from = make_id<short_text>(sender);
            if (!ref || !from)
                return fail(capability_errc::invalid_argument, {}, "a reference or sender longer than 255 characters");
            const credit c{id<provider_id>("mem_cr_" + std::to_string(++next_id_)), net, money::zero(net.asset_of()), *ref,
                           *from, stamp()};
            credit_into(net);
            credits_.push_back(c);
            std::string related;
            if (const auto waiting = awaiting_credit_.find(reference); waiting != awaiting_credit_.end()) {
                payment& p = payments_.at(waiting->second);
                if (p.amount.asset_of() == net.asset_of()) {
                    p.status = payment_status::succeeded;
                    p.received = net;
                    p.updated_at = c.occurred_at;
                    related = std::string(p.payment_id.view());
                    awaiting_credit_.erase(waiting);
                }
            }
            record(related, fact_kind::bank_credit, net, money::zero(net.asset_of()),
                   fact_refs{id<provider_id>(related), {}, {}, {}, {}}, std::string(c.credit_id.view()));
            return ok(c);
        }

        /// \brief Registers a payout beneficiary.
        /// \param beneficiary The beneficiary's id.
        void add_beneficiary(std::string beneficiary) { beneficiaries_.insert(std::move(beneficiary)); }

        /// \brief Sets the rate from `r.from()` to `r.to()`.
        /// \param r The rate, target per source.
        void set_rate(const rate& r) { rates_.insert_or_assign(key(r.from(), r.to()), r); }

        /// \brief Sets the fee on card collections and quotes, in basis points.
        /// \param bps 0 to 10000.
        void set_fee_bps(std::uint32_t bps) noexcept { fee_bps_ = std::min<std::uint32_t>(bps, 10000); }

        /// \brief A delivery of `e`, signed now (or at `signed_at`), as the provider would send it.
        /// \param e         The event.
        /// \param signed_at The signing time; the clock's when empty.
        /// \return The delivery, with `received_at` the clock's time.
        [[nodiscard]] owned_delivery signed_webhook(const webhook_event& e, std::optional<utc_s> signed_at = {}) const
        {
            return detail::memory_webhook::sign(secret_, e, signed_at ? *signed_at : stamp(), stamp());
        }

        /// \brief Makes the next `collect`, `payout` or `refund` fail (once) as though the provider answered `response`.
        /// \param code     The error's code.
        /// \param response The provider's response: its first word becomes the provider code when it is one, and
        ///                 nothing else of it is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        // The clock's time, in the seconds every record of this reference carries.
        [[nodiscard]] utc_s stamp() const noexcept(noexcept(std::declval<const Clock&>().now()))
        {
            return detail::clock_now<std::chrono::seconds>(clock_);
        }
        // An id the provider makes itself, which always fits.
        template <typename Id>
        static Id id(std::string_view text) noexcept
        {
            return *Id::make(text);
        }

        std::optional<capability_error> take_injected() noexcept
        {
            auto e = injected_;
            injected_.reset();
            return e;
        }

        // Every field of a borrowed request, owned, separated: equal fingerprints, equal requests.
        static void field(std::string& out, std::string_view v)
        {
            out += std::to_string(v.size());
            out += ':';
            out += v;
            out += '\x1f';
        }
        static void field(std::string& out, const money& m)
        {
            field(out, to_string(m));
            field(out, m.asset_of().code());
            field(out, std::to_string(m.asset_of().scale()));
        }
        static std::string fingerprint(const collection_request& c)
        {
            std::string out;
            for (const std::string_view v : {c.idempotency_key, c.our_reference, c.payer_reference, c.return_url, c.cancel_url})
                field(out, v);
            field(out, c.amount);
            field(out, std::to_string(static_cast<int>(c.method)));
            for (const auto& [k, v] : c.metadata) {
                field(out, k);
                field(out, v);
            }
            return out;
        }
        static std::string fingerprint(const payout_request& o)
        {
            std::string out;
            for (const std::string_view v : {o.idempotency_key, o.our_reference, o.beneficiary_id, o.reference})
                field(out, v);
            field(out, o.amount);
            field(out, o.quote_id ? "1" : "0");
            field(out, o.quote_id.value_or(std::string_view{}));
            return out;
        }
        static std::string fingerprint(const refund_request& r)
        {
            std::string out;
            for (const std::string_view v : {r.idempotency_key, r.payment_id, r.reason})
                field(out, v);
            if (r.amount)
                field(out, *r.amount);
            else
                field(out, "-");
            return out;
        }

        static std::string key(const asset& from, const asset& to)
        {
            return std::string(from.code()) + ":" + std::to_string(from.scale()) + ">" + std::string(to.code()) + ":" +
                   std::to_string(to.scale());
        }

        std::optional<capability_error> check_request(std::string_view idempotency_key, std::string_view our_reference,
                                                      const money& amount) const
        {
            if (idempotency_key.empty() || idempotency_key.size() > client_id::capacity())
                return capability_error{capability_errc::invalid_argument, {}, "an idempotency key of 1 to 64 characters"};
            if (our_reference.size() > client_id::capacity())
                return capability_error{capability_errc::invalid_argument, {}, "our reference is at most 64 characters"};
            if (amount.is_negative() || amount.is_zero())
                return capability_error{capability_errc::invalid_argument, {}, "an amount is positive"};
            if (std::find(currencies_.begin(), currencies_.end(), amount.asset_of()) == currencies_.end())
                return capability_error{capability_errc::unsupported, {}, "the currency is not held"};
            return std::nullopt;
        }

        payment new_payment(std::string_view prefix, movement kind, std::string_view our_reference, payment_method method,
                            const money& amount)
        {
            payment p;
            p.payment_id = id<provider_id>(std::string(prefix) + std::to_string(++next_id_));
            p.our_reference = id<client_id>(our_reference);
            p.kind = kind;
            p.method = method;
            p.status = payment_status::pending;
            p.amount = amount;
            p.received = p.fee = p.refunded = money::zero(amount.asset_of());
            p.updated_at = stamp();
            return p;
        }

        money fee_on(const money& m) const
        {
            const auto r = rate::make(m.asset_of(), m.asset_of(), fee_bps_, 10000);
            const auto f = r ? convert(m, *r, rounding::ceiling) : result<conversion, money_errc>(err(money_errc::overflow));
            return f ? f->converted : money::zero(m.asset_of());
        }

        void settle(payment& p, const money& gross)
        {
            p.status = payment_status::succeeded;
            p.received = gross;
            p.fee = fee_on(gross);
            p.updated_at = stamp();
            credit_into(*subtract(gross, p.fee));
            record(std::string(p.payment_id.view()), fact_kind::payment, gross, p.fee, fact_refs{p.payment_id, {}, {}, {}, {}});
        }

        // One balance transaction: gross, the fee, and net = gross - fee, so balances(settled) holds by construction.
        void record(std::string related, fact_kind kind, const money& gross, const money& fee, const fact_refs& refs,
                    const std::string& line_id = {})
        {
            payment_fact f;
            f.source = "memory";
            f.provider_account = provider_account_;
            f.fact_id = id<fact_id>(line_id.empty() ? "mem_txn_" + std::to_string(++next_id_) : line_id);
            f.kind = kind;
            f.dates.trade_at = stamp();
            f.dates.known_at = f.dates.trade_at;
            f.dates.settled_at = f.dates.trade_at;   // the reference settles at once
            f.dates.settles_on = f.dates.trade_at.days();
            f.settled = settlement{gross, fee, *subtract(gross, fee)};
            f.refs = refs;
            ledger_.emplace_back(std::move(related), f);
        }

        void credit_into(const money& m)
        {
            const auto it = balances_.find(m.asset_of().code());
            if (it == balances_.end())
                balances_.emplace(std::string(m.asset_of().code()), m);
            else
                it->second = *add(it->second, m);
        }

        capability_result<void> take(const money& m)
        {
            const auto it = balances_.find(m.asset_of().code());
            if (it == balances_.end() || it->second.asset_of() != m.asset_of() || it->second.magnitude() < m.magnitude())
                return fail(capability_errc::insufficient_funds);
            it->second = *subtract(it->second, m);
            return ok();
        }

        std::vector<asset> currencies_;
        secret<std::string> secret_;
        Clock clock_;
        std::uint32_t fee_bps_{0};
        std::uint64_t next_id_{0};
        std::uint64_t next_reference_{0};
        std::map<std::string, payment, std::less<>> payments_;
        // Idempotency: the key, the request's fingerprint (an owned copy of the borrowed request) and the answer.
        std::map<std::string, std::pair<std::string, collection>, std::less<>> collect_keys_;
        std::map<std::string, std::pair<std::string, payment>, std::less<>> payout_keys_;
        std::map<std::string, std::pair<std::string, payment>, std::less<>> refund_keys_;
        std::map<std::string, std::string, std::less<>> awaiting_credit_;
        std::map<std::string, money, std::less<>> balances_;
        std::map<std::string, rate, std::less<>> rates_;
        std::map<std::string, fx_quote, std::less<>> quotes_;
        std::set<std::string, std::less<>> beneficiaries_;
        std::vector<credit> credits_;
        std::vector<std::pair<std::string, payment_fact>> ledger_;   // (related payment id, fact), in order
        snicholls::provider_account provider_account_{"mem_acct_1"};
        std::optional<capability_error> injected_;
    };

    /// \brief The reference with a `manual_utc_clock`: what tests use.
    /// \ingroup core
    using memory_payment_provider = basic_memory_payment_provider<>;

    static_assert(payment_provider<memory_payment_provider>, "snicholls::memory_payment_provider must model snicholls::payment_provider");

} // namespace snicholls

#endif // snicholls_core_interfaces_payment_provider_reference_hpp
