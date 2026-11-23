//
//  onramp_provider_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_onramp_provider_conformance`: the conformance kit of the `onramp_provider` contract.
/// \ingroup core
///
/// **Overview.** Drives any `onramp_provider`, in either form, through the contract of `concepts/onramp_provider.hpp`
/// and reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& venue()`: the implementation, already funded in the quote currency;
/// - `asset unit()`: a stablecoin it delivers; `asset quote_currency()`: what pays for it;
/// - `money ticket()`: a spend in the quote currency the funding covers; `money too_much()`: one it does not;
/// - `std::uint64_t chain()`: a chain the asset is withdrawn on; `std::string address()`: a whitelisted address;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `quote` fails with `code`, as
///   though the venue had answered `response`.
///
/// The checks: `a quote is exact in its assets`, `a quote in a third asset is invalid`, `a buy is idempotent on its
/// client order id`, `a buy never fills above max_price or beyond the balance`, `a withdrawal checks chain, address
/// and balance`, `unknown ids are not found`, `balances show what was bought`, `the statement pages facts that
/// validate`, `a provider failure never leaks its response`.
///
/// \snippet interfaces/onramp_provider_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/onramp_provider.hpp`, `interfaces/onramp_provider.hpp`,
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
/// **Tests and demos.** `tests/interfaces/onramp_provider_conformance_test.cpp`,
/// `demos/interfaces/onramp_provider_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`, `<optional>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `concepts/onramp_provider.hpp`, `interfaces/capability_conformance.hpp`, `money/asset.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_onramp_provider_conformance_hpp
#define snicholls_core_interfaces_onramp_provider_conformance_hpp

#include <snicholls/core/concepts/onramp_provider.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>
#include <snicholls/core/money/asset.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept onramp_provider_probe = std::is_default_constructible_v<Probe> &&
            onramp_provider<std::remove_reference_t<decltype(std::declval<Probe&>().venue())>> &&
            requires(Probe& p, capability_errc c, std::string_view r) {
                { p.unit() } -> std::same_as<asset>;
                { p.quote_currency() } -> std::same_as<asset>;
                { p.ticket() } -> std::same_as<money>;
                { p.too_much() } -> std::same_as<money>;
                { p.chain() } -> std::same_as<std::uint64_t>;
                { p.address() } -> std::convertible_to<std::string>;
                p.provider_fails(c, r);
            };

        template <typename Probe>
        buy_request kit_buy(Probe& p, std::string_view id, const money& amount)
        {
            return {id, p.unit(), p.quote_currency(), amount, std::nullopt};
        }
    } // namespace detail

    /// \brief Runs every check of the `onramp_provider` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::onramp_provider_probe<Probe>
    [[nodiscard]] capability_conformance_report run_onramp_provider_conformance()
    {
        capability_conformance_report report;
        using detail::kit_buy;
        const auto code_is = [](const auto& r, capability_errc c) { return r.is_err() && r.unwrap_err().code == c; };

        detail::capability_check<Probe>(report, "a quote is exact in its assets", [](Probe& p, const auto& expect) {
            const auto q = p.venue().quote({p.unit(), p.quote_currency(), p.ticket()});
            expect(q.is_ok(), "a funded ticket is quoted");
            if (!q.is_ok())
                return;
            expect(q->spend.asset_of() == p.quote_currency(), "spend is in the quote currency");
            expect(q->receive.asset_of() == p.unit(), "receive is in the asset");
            expect(!q->receive.is_zero(), "the ticket buys something");
            expect(q->price.from() == p.unit() && q->price.to() == p.quote_currency(), "the price is quote per asset");
        });

        detail::capability_check<Probe>(report, "a quote in a third asset is invalid", [&](Probe& p, const auto& expect) {
            constexpr asset other = checked_asset("ZZZ", 2);
            const auto q = p.venue().quote({p.unit(), p.quote_currency(), money::from_units(other, 100)});
            expect(code_is(q, capability_errc::invalid_argument), "an amount in neither asset is invalid_argument");
        });

        detail::capability_check<Probe>(report, "a buy is idempotent on its client order id", [&](Probe& p, const auto& expect) {
            auto& v = p.venue();
            const auto first = v.buy(kit_buy(p, "c-1", p.ticket()));
            const auto again = v.buy(kit_buy(p, "c-1", p.ticket()));
            expect(first.is_ok() && again.is_ok(), "both buys answer");
            if (!first.is_ok() || !again.is_ok())
                return;
            expect(first->order_id == again->order_id, "a repeat returns the first order");
            expect(code_is(v.buy(kit_buy(p, "c-1", p.too_much())), capability_errc::conflict), "the id with another buy is a conflict");
            const auto s = v.order_status(first->order_id.view());
            expect(s.is_ok() && s->client_order_id == std::string_view{"c-1"}, "the order keeps its client order id");
            const std::string long_id(client_id::capacity() + 1, 'c');
            expect(code_is(v.buy(kit_buy(p, long_id, p.ticket())), capability_errc::invalid_argument),
                   "a client order id longer than 64 characters is invalid_argument");
        });

        detail::capability_check<Probe>(report, "a buy never fills above max_price or beyond the balance", [&](Probe& p, const auto& expect) {
            auto& v = p.venue();
            auto capped = kit_buy(p, "c-cap", p.ticket());
            capped.max_price = *rate::make(p.unit(), p.quote_currency(), 1, 1'000'000);
            expect(code_is(v.buy(capped), capability_errc::rejected), "above max_price is rejected");
            expect(code_is(v.buy(kit_buy(p, "c-big", p.too_much())), capability_errc::insufficient_funds),
                   "beyond the balance is insufficient_funds");
        });

        detail::capability_check<Probe>(report, "a withdrawal checks chain, address and balance", [&](Probe& p, const auto& expect) {
            auto& v = p.venue();
            const auto bought = v.buy(kit_buy(p, "c-w", p.ticket()));
            expect(bought.is_ok(), "the buy fills");
            if (!bought.is_ok())
                return;
            const std::string address = p.address();
            const money one = money::from_units(p.unit(), 1);
            expect(code_is(v.withdraw({"w-addr", one, p.chain(), "0x1234", ""}), capability_errc::invalid_argument),
                   "a malformed address is invalid_argument");
            expect(code_is(v.withdraw({"w-chain", one, 999'999'999u, address, ""}), capability_errc::invalid_argument),
                   "an unsupported chain is invalid_argument");
            const money huge = money::from_units(p.unit(), bought->filled.magnitude() + uint256{1'000'000'000});
            expect(code_is(v.withdraw({"w-huge", huge, p.chain(), address, ""}), capability_errc::insufficient_funds),
                   "more than the balance is insufficient_funds");
            const auto w = v.withdraw({"w-ok", one, p.chain(), address, ""});
            expect(w.is_ok(), "a valid withdrawal is accepted");
            if (!w.is_ok())
                return;
            expect(!w->withdrawal_id.empty(), "the withdrawal has the venue's id");
            const auto s = v.withdrawal_status(w->withdrawal_id.view());
            expect(s.is_ok() && s->client_reference == std::string_view{"w-ok"}, "the withdrawal keeps its client reference");
            const auto again = v.withdraw({"w-ok", one, p.chain(), address, ""});
            expect(again.is_ok() && again->withdrawal_id == w->withdrawal_id, "a repeat returns the first withdrawal");
        });

        detail::capability_check<Probe>(report, "unknown ids are not found", [&](Probe& p, const auto& expect) {
            expect(code_is(p.venue().order_status("no-such-order"), capability_errc::not_found), "an unknown order");
            expect(code_is(p.venue().withdrawal_status("no-such-withdrawal"), capability_errc::not_found), "an unknown withdrawal");
        });

        detail::capability_check<Probe>(report, "balances show what was bought", [](Probe& p, const auto& expect) {
            auto& v = p.venue();
            expect(v.buy(kit_buy(p, "c-bal", p.ticket())).is_ok(), "the buy fills");
            const auto b = v.balances();
            expect(b.is_ok(), "balances answer");
            bool found = false;
            if (b.is_ok())
                for (const venue_balance& x : *b)
                    found = found || (x.available.asset_of() == p.unit() && !x.available.is_zero());
            expect(found, "the asset bought is in a balance");
        });

        detail::capability_check<Probe>(report, "the statement pages facts that validate", [](Probe& p, const auto& expect) {
            auto& v = p.venue();
            expect(v.buy(kit_buy(p, "c-s1", p.ticket())).is_ok() && v.buy(kit_buy(p, "c-s2", p.ticket())).is_ok(), "two buys fill");
            const auto first = v.statement({onramp_statement_source::fills, {}, 1});
            expect(first.is_ok() && first->facts.size() == 1 && !first->next_cursor.empty(), "the first page holds one fact and a cursor");
            if (!first.is_ok() || first->facts.size() != 1)
                return;
            const auto second = v.statement({onramp_statement_source::fills, first->next_cursor.view(), 1});
            expect(second.is_ok() && second->facts.size() == 1, "the second page holds one fact");
            if (!second.is_ok() || second->facts.size() != 1)
                return;
            expect(first->facts[0].fact_id != second->facts[0].fact_id, "no fact twice");
            for (const onramp_fact& f : {first->facts[0], second->facts[0]}) {
                expect(f.kind == onramp_fact_kind::trade_fill, "a buy is a trade_fill");
                expect(validate(f).is_ok(), "every fact validates");
            }
            const auto w = v.statement({onramp_statement_source::withdrawals, {}, 100});
            expect(w.is_ok(), "the withdrawals statement answers");
            if (w.is_ok())
                for (const onramp_fact& f : w->facts)
                    expect(validate(f).is_ok(), "every fact validates");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [&](Probe& p, const auto& expect) {
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto q = p.venue().quote({p.unit(), p.quote_currency(), p.ticket()});
            expect(code_is(q, capability_errc::unavailable), "the provider's failure is returned");
            if (q.is_err())
                expect(!capability_error_leaks(q.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_onramp_provider_conformance_hpp
