//
//  security_provider_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_security_provider_conformance`: the conformance kit of the `security_provider` contract.
/// \ingroup core
///
/// **Overview.** Drives any `security_provider`, in either form, through the contract of
/// `concepts/security_provider.hpp` and reports which check failed (`interfaces/capability_conformance.hpp`). The probe
/// is a default-constructible type with
/// - `auto& provider()`: the implementation (a provider, a `security_provider_interface&`, a `security_provider_ref`);
/// - `money quantity()`: a positive quantity of an instrument the provider trades; `money consideration()`: the cash
///   paid for it (negative);
/// - `void trade(std::string_view order, const money& quantity, const money& consideration)`: the provider fills the
///   order now; afterwards an `event_kind::security_updated` event naming the order in `payment_id` resolves to its
///   facts;
/// - `void settle(std::string_view order)`: the provider settles it now;
/// - `owned_delivery deliver(const webhook_event& e)`: `e` as the provider would sign and send it, received now;
///   `owned_delivery deliver_late(const webhook_event& e)`: the same, received outside the provider's time window;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `statement` fails with `code`,
///   as though the provider had answered `response`.
///
/// The checks: `capabilities name the kinds it reports`, `a trade resolves to one fact that validates`, `an event
/// that books nothing resolves to no facts`, `settlement moves the settled holding`, `a webhook verifies and carries
/// its event`, `a webhook fails closed`, `a statement pages through every fact once`, `a provider failure never leaks
/// its response`.
///
/// \snippet interfaces/security_provider_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/security_provider.hpp`, `interfaces/security_provider.hpp`,
/// `interfaces/capability_conformance.hpp`; the sibling `interfaces/payment_provider_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form; the holdings check passes without running for a provider
/// whose capabilities say it has none.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/security_provider_conformance_test.cpp`,
/// `demos/interfaces/security_provider_conformance_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cstddef>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `<vector>`, `concepts/security_provider.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.26.0.

#ifndef snicholls_core_interfaces_security_provider_conformance_hpp
#define snicholls_core_interfaces_security_provider_conformance_hpp

#include <snicholls/core/concepts/security_provider.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept security_provider_probe = std::is_default_constructible_v<Probe> &&
            security_provider<std::remove_reference_t<decltype(std::declval<Probe&>().provider())>> &&
            requires(Probe& p, std::string_view order, const money& m, const webhook_event& e, capability_errc code) {
                { p.quantity() } -> std::same_as<money>;
                { p.consideration() } -> std::same_as<money>;
                p.trade(order, m, m);
                p.settle(order);
                { p.deliver(e) } -> std::same_as<owned_delivery>;
                { p.deliver_late(e) } -> std::same_as<owned_delivery>;
                p.provider_fails(code, order);
            };

        inline webhook_event security_kit_event(std::string_view order = "ord-1")
        {
            webhook_event e;
            e.source = "test";
            e.event_id = "evt_sec_1";
            e.kind = event_kind::security_updated;
            e.type = "trade.filled";
            e.payment_id = *provider_id::make(order);
            e.occurred_at = utc_s::from_unix_seconds(1'800'000'000);
            return e;
        }

        template <typename Provider>
        std::vector<security_fact> kit_resolve(Provider& provider, std::string_view order)
        {
            const auto r = provider.resolve(security_kit_event(order));
            return r.is_ok() ? *r : std::vector<security_fact>{};
        }

        inline const security_holding* kit_find(const std::vector<security_holding>& hs, const asset& a)
        {
            for (const auto& h : hs)
                if (h.instrument == a)
                    return &h;
            return nullptr;
        }
    } // namespace detail

    /// \brief Runs every check of the `security_provider` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::security_provider_probe<Probe>
    [[nodiscard]] capability_conformance_report run_security_provider_conformance()
    {
        capability_conformance_report report;
        const auto code_is = [](const auto& r, capability_errc c) { return r.is_err() && r.unwrap_err().code == c; };

        detail::capability_check<Probe>(report, "capabilities name the kinds it reports", [](Probe& p, const auto& expect) {
            const auto caps = p.provider().capabilities();
            expect(!caps.kinds.empty(), "at least one kind");
            expect(std::find(caps.kinds.begin(), caps.kinds.end(), security_fact_kind::buy) != caps.kinds.end(),
                   "a provider the kit can drive reports buys");
        });

        detail::capability_check<Probe>(report, "a trade resolves to one fact that validates", [](Probe& p, const auto& expect) {
            p.trade("ord-1", p.quantity(), p.consideration());
            const auto facts = detail::kit_resolve(p.provider(), "ord-1");
            expect(facts.size() == 1, "the trade resolves to one fact");
            if (facts.size() != 1)
                return;
            const security_fact& f = facts.front();
            expect(validate(f).is_ok(), "the fact validates");
            expect(f.kind == security_fact_kind::buy, "a positive quantity is a buy");
            expect(f.quantity == p.quantity() && f.consideration == p.consideration(), "quantity and cash are the trade's");
            expect(f.event_id == detail::security_kit_event().event_id, "the fact carries the event that led to it");
            expect(!f.dates.settles_on || *f.dates.settles_on >= f.dates.trade_at.days(), "it settles on or after its trade date");
            const auto again = detail::kit_resolve(p.provider(), "ord-1");
            expect(again.size() == 1 && again.front().fact_id == f.fact_id, "a redelivery points at the same fact");
        });

        detail::capability_check<Probe>(report, "an event that books nothing resolves to no facts", [](Probe& p, const auto& expect) {
            p.trade("ord-1", p.quantity(), p.consideration());
            webhook_event other = detail::security_kit_event();
            other.kind = event_kind::other;
            const auto none = p.provider().resolve(other);
            expect(none.is_ok() && none->empty(), "an other event resolves to no facts");
            webhook_event test = detail::security_kit_event();
            test.is_test = true;
            const auto booked = p.provider().resolve(test);
            expect(booked.is_ok() && booked->empty(), "a test delivery resolves to no facts");
            const auto unknown = p.provider().resolve(detail::security_kit_event("ord-none"));
            expect(unknown.is_ok() && unknown->empty(), "an unknown order resolves to no facts");
        });

        detail::capability_check<Probe>(report, "settlement moves the settled holding", [](Probe& p, const auto& expect) {
            if (!p.provider().capabilities().holdings)
                return;
            const money q = p.quantity();
            p.trade("ord-1", q, p.consideration());
            const auto traded = detail::kit_resolve(p.provider(), "ord-1");
            expect(traded.size() == 1, "the trade resolves");
            if (traded.size() != 1)
                return;
            const auto day = traded.front().dates.trade_at.days();
            const auto before = p.provider().holdings(day - std::chrono::days{1});
            expect(before.is_ok() && detail::kit_find(*before, q.asset_of()) == nullptr, "nothing is held the day before");
            const auto pending = p.provider().holdings(day);
            const security_holding* h = pending.is_ok() ? detail::kit_find(*pending, q.asset_of()) : nullptr;
            expect(h && h->traded == q && h->settled.is_zero(), "traded on the trade date, not yet settled");
            p.settle("ord-1");
            const auto facts = detail::kit_resolve(p.provider(), "ord-1");
            const auto settled_fact = std::find_if(facts.begin(), facts.end(),
                                                   [](const security_fact& f) { return f.kind == security_fact_kind::settle; });
            expect(settled_fact != facts.end() && validate(*settled_fact).is_ok() && settled_fact->dates.settled_at,
                   "the settlement is a settle fact with its time");
            if (settled_fact == facts.end() || !settled_fact->dates.settled_at)
                return;
            const auto after = p.provider().holdings(settled_fact->dates.settled_at->days());
            const security_holding* s = after.is_ok() ? detail::kit_find(*after, q.asset_of()) : nullptr;
            expect(s && s->traded == q && s->settled == q, "settled once the settle fact is in");
        });

        detail::capability_check<Probe>(report, "a webhook verifies and carries its event", [](Probe& p, const auto& expect) {
            const webhook_event e = detail::security_kit_event();
            const owned_delivery d = p.deliver(e);
            const auto v = p.provider().verify_webhook(d.view());
            expect(v.is_ok(), "a signed delivery verifies");
            if (!v.is_ok())
                return;
            expect(v->event_id == e.event_id && v->kind == e.kind && v->payment_id == e.payment_id, "the event round-trips");
            const auto again = p.provider().verify_webhook(d.view());
            expect(again.is_ok() && *again == *v, "the event id is stable across deliveries");
        });

        detail::capability_check<Probe>(report, "a webhook fails closed", [&](Probe& p, const auto& expect) {
            auto& pp = p.provider();
            owned_delivery tampered = p.deliver(detail::security_kit_event());
            tampered.body[tampered.body.size() / 2] = static_cast<char>(tampered.body[tampered.body.size() / 2] ^ 1);
            expect(code_is(pp.verify_webhook(tampered.view()), capability_errc::signature_invalid), "a tampered body");
            owned_delivery unsigned_one = p.deliver(detail::security_kit_event());
            unsigned_one.headers.clear();
            expect(code_is(pp.verify_webhook(unsigned_one.view()), capability_errc::signature_invalid), "no signature");
            const owned_delivery late = p.deliver_late(detail::security_kit_event());
            expect(code_is(pp.verify_webhook(late.view()), capability_errc::stale), "outside the window is stale");
            const webhook_delivery garbage{{}, "\x01\x02 not a webhook", {}};
            expect(code_is(pp.verify_webhook(garbage), capability_errc::signature_invalid), "garbage");
        });

        detail::capability_check<Probe>(report, "a statement pages through every fact once", [](Probe& p, const auto& expect) {
            for (const std::string_view order : {"ord-s1", "ord-s2", "ord-s3"})
                p.trade(order, p.quantity(), p.consideration());
            p.settle("ord-s1");
            std::vector<std::string> seen;
            security_statement_query q{"", std::nullopt, 2};
            std::string cursor;
            for (int guard = 0; guard < 50; ++guard) {
                const auto page = p.provider().statement(q);
                expect(page.is_ok(), "every page answers");
                if (!page.is_ok())
                    return;
                expect(page->facts.size() <= 2, "a page holds at most the limit");
                for (const security_fact& f : page->facts) {
                    expect(validate(f).is_ok(), "every fact validates");
                    seen.emplace_back(f.fact_id.view());
                }
                if (page->next_cursor.empty())
                    break;
                cursor = std::string(page->next_cursor.view());
                q.cursor = cursor;
            }
            expect(seen.size() >= 4, "every trade and the settlement are facts");
            std::sort(seen.begin(), seen.end());
            expect(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "no fact twice");
            security_statement_query other{"", p.quantity().asset_of(), 100};
            const auto one = p.provider().statement(other);
            expect(one.is_ok() && one->facts.size() >= 4, "the instrument's facts are its own");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto r = p.provider().statement({});
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_security_provider_conformance_hpp
