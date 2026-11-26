//
//  security_provider_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_security_provider`: the standard-library reference `security_provider`; a broker and
/// registry in memory, with trades that settle later, any fact booked by hand, signed webhooks, statements and
/// holdings.
/// \ingroup core
///
/// **Overview.** It plays the provider's side for a product's tests, without a network:
/// - `execute(order, quantity, consideration, fees)` fills an order now: a `buy` (positive quantity) or `sell` fact,
///   due to settle two calendar days later (`dates.settles_on`, estimated: business days are the ledger's);
/// - `settle(order)` settles it now (a `settle` fact with the trade's quantity), and `fail_settlement(order)` reports
///   a fail (`settle_failed`);
/// - `book(fact)` records any other fact (a dividend, a split, a repo) after `validate`, filling in the source, the
///   account, an id and `known_at` where they are empty.
///
/// Webhooks are signed as `memory_payment_provider` signs them (a `Memory-Signature` header, HMAC-SHA256 over
/// `<t>.<body>`, a 300-second window); `signed_webhook` makes a delivery for a test. `resolve` returns the facts an
/// event names, by `fact_id` or by the order in `payment_id`, from the records, never from the body. `statement` pages
/// through every fact in order. `holdings(as_of)` folds the facts by the contract's rule: the traded position counts
/// every fact but `settle` and `settle_failed` traded on or before the day, the settled position every `settle` and
/// every other fact with `settled_at` on or before it, both instruments of a two-instrument fact. `fail_next(code,
/// response)` makes the next `resolve`, `statement` or `holdings` fail as though the provider had answered `response`.
///
/// \snippet interfaces/security_provider_reference_demo.cpp reference
///
/// **Relationships.** Models `security_provider` (`concepts/security_provider.hpp`); passes
/// `interfaces/security_provider_conformance.hpp` in every form; shares its webhook scheme with
/// `interfaces/payment_provider_reference.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** Not thread-safe. Ids are deterministic (`mem_sf_1`, ...). Quantities and cash are exact. Errors
/// follow the redaction rule.
///
/// **Errors.** `book` returns `validate`'s error, or `conflict` for a `fact_id` already booked with other content;
/// `execute` returns `invalid_argument` for a zero quantity or an order id that is empty, too long or already filled;
/// `settle` and `fail_settlement` return `not_found` for an unknown order and `conflict` for one already settled or
/// failed; and whatever `fail_next` armed.
///
/// **Performance.** Not benchmarked: a test double.
///
/// **Tests and demos.** `tests/interfaces/security_provider_reference_test.cpp`,
/// `demos/interfaces/security_provider_reference_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstdint>`, `<functional>`, `<map>`, `<optional>`, `<string>`, `<string_view>`,
/// `<utility>`, `<vector>`, `detail/memory_webhook.hpp` (HMAC-SHA256), `interfaces/security_provider.hpp`, `secret.hpp`.
///
/// **Known issues.** Settlement is T+2 calendar days; real settlement is in business days of the market's calendar.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_security_provider_reference_hpp
#define snicholls_core_interfaces_security_provider_reference_hpp

#include <snicholls/core/detail/memory_webhook.hpp>
#include <snicholls/core/interfaces/security_provider.hpp>
#include <snicholls/core/secret.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference security provider.
    /// \ingroup core
    class memory_security_provider {
    public:
        /// \brief Reads the time.
        using clock_source = std::function<utc_s()>;
        /// \brief The webhook time window, either side of `received_at`.
        static constexpr std::chrono::seconds webhook_tolerance = detail::memory_webhook::tolerance;
        /// \brief The signature header.
        static constexpr std::string_view signature_header = detail::memory_webhook::header;
        /// \brief The days from a trade to its settlement.
        static constexpr std::chrono::days settlement_lag{2};

        /// \brief A provider signing webhooks with `webhook_secret`.
        /// \param webhook_secret The HMAC key for webhooks (never logged).
        /// \param clock          The clock.
        memory_security_provider(secret<std::string> webhook_secret, clock_source clock)
            : secret_(std::move(webhook_secret)), clock_(std::move(clock))
        {
        }

        /// \brief What it reports: every kind, and holdings.
        /// \return The capabilities.
        [[nodiscard]] security_capabilities capabilities() const
        {
            security_capabilities c;
            for (auto k = std::uint8_t{1}; k <= static_cast<std::uint8_t>(security_fact_kind::distribution_reinvested); ++k)
                c.kinds.push_back(static_cast<security_fact_kind>(k));
            c.holdings = true;
            c.provider_account = provider_account_;
            return c;
        }

        /// \brief Authenticates (HMAC-SHA256, constant time) and parses a delivery; fails closed; no I/O.
        /// \param w The delivery.
        /// \return The event, or `signature_invalid`, `stale` or `invalid_argument`.
        [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const
        {
            return detail::memory_webhook::verify(secret_, w);
        }

        /// \brief The facts a verified event names, from the records: by `fact_id`, or every fact of the order in
        /// `payment_id`. A test delivery or an `other` event resolves to none.
        /// \param e The event.
        /// \return The facts, oldest first, with the event's id.
        capability_result<std::vector<security_fact>> resolve(const webhook_event& e)
        {
            if (auto bad = take_injected())
                return err(*bad);
            std::vector<security_fact> out;
            if (e.kind == event_kind::other || e.is_test)
                return ok(std::move(out));
            for (const security_fact& f : facts_)
                if ((!e.fact_id.empty() && f.fact_id == e.fact_id) || (!e.payment_id.empty() && f.refs.order == e.payment_id)) {
                    security_fact copy = f;
                    copy.event_id = e.event_id;
                    out.push_back(copy);
                }
            return ok(std::move(out));
        }

        /// \brief A page of facts in booking order, optionally one instrument's.
        /// \param q The page; the cursor is the index of the next fact.
        /// \return The page, or `invalid_argument` for a cursor it did not give.
        capability_result<security_statement_page> statement(const security_statement_query& q)
        {
            if (auto bad = take_injected())
                return err(*bad);
            std::size_t from = 0;
            if (!q.cursor.empty()) {
                std::int64_t at = 0;
                if (!detail::memory_webhook::parse_int(q.cursor, at) || at < 0 || static_cast<std::size_t>(at) > facts_.size())
                    return fail(capability_errc::invalid_argument, {}, "not a cursor this provider gave");
                from = static_cast<std::size_t>(at);
            }
            security_statement_page page;
            std::size_t i = from;
            for (; i < facts_.size() && page.facts.size() < q.limit; ++i) {
                const security_fact& f = facts_[i];
                if (!q.instrument || f.instrument == *q.instrument || f.other_instrument == q.instrument)
                    page.facts.push_back(f);
            }
            if (i < facts_.size())
                page.next_cursor = *provider_id::make(std::to_string(i));
            return ok(std::move(page));
        }

        /// \brief The positions at the end of `as_of`, by trade date and by settlement, in order of instrument code;
        /// instruments with both positions zero are left out.
        /// \param as_of The day.
        /// \return The holdings.
        capability_result<std::vector<security_holding>> holdings(std::chrono::sys_days as_of)
        {
            if (auto bad = take_injected())
                return err(*bad);
            std::map<std::string, security_holding, std::less<>> by_code;
            const auto move = [&](const money& q, bool traded, bool settled) {
                const std::string key = std::string(q.asset_of().code()) + ":" + std::to_string(q.asset_of().scale());
                auto [it, fresh] = by_code.try_emplace(key, security_holding{q.asset_of(), money::zero(q.asset_of()),
                                                                             money::zero(q.asset_of())});
                if (traded)
                    it->second.traded = *add(it->second.traded, q);
                if (settled)
                    it->second.settled = *add(it->second.settled, q);
            };
            for (const security_fact& f : facts_) {
                const bool settlement = f.kind == security_fact_kind::settle;
                const bool traded = !settlement && f.kind != security_fact_kind::settle_failed && f.dates.trade_at.days() <= as_of;
                const bool settled = f.kind != security_fact_kind::settle_failed && f.dates.settled_at &&
                                     f.dates.settled_at->days() <= as_of;
                move(f.quantity, traded, settled);
                if (f.other_quantity)
                    move(*f.other_quantity, traded, settled);
            }
            std::vector<security_holding> out;
            for (const auto& [code, h] : by_code)
                if (!h.traded.is_zero() || !h.settled.is_zero())
                    out.push_back(h);
            return ok(std::move(out));
        }

        /// \brief Records a fact after `validate`, filling the source, the account, an id and `known_at` when empty.
        /// Booking the same fact again changes nothing.
        /// \param f The fact.
        /// \return The fact as booked, or `validate`'s error, or `conflict` for its id booked with other content.
        capability_result<security_fact> book(security_fact f)
        {
            if (f.source.empty())
                f.source = "memory";
            if (f.provider_account.empty())
                f.provider_account = provider_account_;
            if (f.fact_id.empty())
                f.fact_id = *fact_id::make("mem_sf_" + std::to_string(++next_id_));
            if (f.dates.known_at == utc_s{})
                f.dates.known_at = clock_();
            if (auto valid = validate(f); !valid)
                return err(valid.unwrap_err());
            for (const security_fact& g : facts_)
                if (g.fact_id == f.fact_id) {
                    if (g.kind == f.kind && g.quantity == f.quantity && g.consideration == f.consideration &&
                        g.dates.same_content(f.dates))
                        return ok(g);
                    return fail(capability_errc::conflict, {}, "the fact id is booked with other content");
                }
            facts_.push_back(f);
            return ok(f);
        }

        /// \brief Fills an order now: a `buy` for a positive quantity, a `sell` for a negative one, due to settle in
        /// `settlement_lag` (estimated).
        /// \param order         The order's id.
        /// \param quantity      The units, signed, in the instrument.
        /// \param consideration The gross cash: negative paid for a buy, positive received for a sell.
        /// \param fees          The brokerage, not negative, in the consideration's currency.
        /// \return The fact, or `invalid_argument`.
        capability_result<security_fact> execute(std::string_view order, const money& quantity, const money& consideration,
                                                 std::optional<money> fees = std::nullopt)
        {
            auto id = make_id<provider_id>(order);
            if (!id || order.empty() || quantity.is_zero())
                return fail(capability_errc::invalid_argument, {}, "an order id and a quantity");
            for (const security_fact& f : facts_)
                if (f.refs.order == *id)
                    return fail(capability_errc::invalid_argument, {}, "the order is already filled");
            security_fact f;
            f.kind = quantity.is_negative() ? security_fact_kind::sell : security_fact_kind::buy;
            f.instrument = quantity.asset_of();
            f.quantity = quantity;
            f.consideration = consideration;
            f.fees = fees ? *fees : money::zero(consideration.asset_of());
            f.dates.trade_at = clock_();
            f.dates.known_at = f.dates.trade_at;
            f.dates.settles_on = f.dates.trade_at.days() + settlement_lag;
            f.dates.settles_on_estimated = true;
            f.refs.order = *id;
            f.refs.execution = *provider_id::make("mem_exe_" + std::to_string(next_id_ + 1));
            return book(f);
        }

        /// \brief Settles a filled order now: a `settle` fact with the trade's quantity and its cash.
        /// \param order The order's id.
        /// \return The fact, or `not_found` or `conflict`.
        capability_result<security_fact> settle(std::string_view order) { return conclude(order, security_fact_kind::settle); }

        /// \brief Reports that a filled order failed to settle on its date: a `settle_failed` fact.
        /// \param order The order's id.
        /// \return The fact, or `not_found` or `conflict`.
        capability_result<security_fact> fail_settlement(std::string_view order)
        {
            return conclude(order, security_fact_kind::settle_failed);
        }

        /// \brief A delivery of `e`, signed now (or at `signed_at`), as the provider would send it.
        /// \param e         The event.
        /// \param signed_at The signing time; the clock's when empty.
        /// \return The delivery, with `received_at` the clock's time.
        [[nodiscard]] owned_delivery signed_webhook(const webhook_event& e, std::optional<utc_s> signed_at = {}) const
        {
            return detail::memory_webhook::sign(secret_, e, signed_at ? *signed_at : clock_(), clock_());
        }

        /// \brief Makes the next `resolve`, `statement` or `holdings` fail (once) as though the provider answered
        /// `response`.
        /// \param code     The error's code.
        /// \param response The provider's response: its first word becomes the provider code when it is one, and
        ///                 nothing else of it is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        std::optional<capability_error> take_injected() noexcept
        {
            auto e = injected_;
            injected_.reset();
            return e;
        }

        capability_result<security_fact> conclude(std::string_view order, security_fact_kind kind)
        {
            const security_fact* trade = nullptr;
            for (const security_fact& f : facts_)
                if (f.refs.order.view() == order && !order.empty()) {
                    if (f.kind == security_fact_kind::settle || f.kind == security_fact_kind::settle_failed)
                        return fail(capability_errc::conflict, {}, "the order is already settled or failed");
                    trade = &f;
                }
            if (!trade)
                return fail(capability_errc::not_found);
            security_fact f = *trade;
            f.fact_id = {};
            f.kind = kind;
            f.dates.known_at = clock_();
            if (kind == security_fact_kind::settle)
                f.dates.settled_at = clock_();
            f.refs.settlement = *provider_id::make("mem_set_" + std::to_string(next_id_ + 1));
            return book(f);
        }

        secret<std::string> secret_;
        clock_source clock_;
        std::uint64_t next_id_{0};
        std::vector<security_fact> facts_;   // in booking order
        snicholls::provider_account provider_account_{"mem_sec_1"};
        std::optional<capability_error> injected_;
    };

    static_assert(security_provider<memory_security_provider>,
                  "snicholls::memory_security_provider must model snicholls::security_provider");

} // namespace snicholls

#endif // snicholls_core_interfaces_security_provider_reference_hpp
