//
//  onramp_provider_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_onramp_provider`: the standard-library reference `onramp_provider`, a venue in memory with
/// prices, balances, fills, withdrawals and facts.
/// \ingroup core
///
/// **Overview.** A venue a test controls: `set_price` fixes the price of an asset in a quote currency (an exact
/// `rate`), `deposit` funds a balance, `set_fee_bps` charges a fee on the quote side, `set_withdrawal_fee` a network
/// fee. `buy` fills at once at the set price (never above `max_price`); `withdraw` checks the chain, the address and
/// the balance, then broadcasts; the test calls `confirm` when it should land. All amounts are exact. Rounding favours
/// the venue, as a real one does: a spend buys `toward_zero` of the asset, a receive costs `ceiling` of the quote
/// currency. `statement` pages one fact per buy and per withdrawal, in the venue's own assets `<code>:MEM` at the same
/// scale. `fail_next(code, response)` makes the next `quote`, `buy` or `withdraw` fail as though the venue had answered
/// `response`.
///
/// \snippet interfaces/onramp_provider_reference_demo.cpp reference
///
/// **Relationships.** Models `onramp_provider` (`concepts/onramp_provider.hpp`); passes
/// `interfaces/onramp_provider_conformance.hpp` in every form. Group: \ref core.
///
/// **Implementations / variants.** One account (`"trading"`); no partial fills.
///
/// **Guarantees.** Not thread-safe. Ids and transaction hashes are deterministic (`mem_ord_1`, `mem_wd_2`, the hash a
/// SHA-256 of the withdrawal id). Errors follow the redaction rule.
///
/// **Errors.** As the contract, and whatever `fail_next` armed.
///
/// **Performance.** Not benchmarked: a test double.
///
/// **Tests and demos.** `tests/interfaces/onramp_provider_reference_test.cpp`,
/// `demos/interfaces/onramp_provider_reference_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<chrono>`, `<cstddef>`, `<cstdint>`, `<functional>`, `<map>`, `<optional>`,
/// `<string>`, `<string_view>`, `<utility>`, `<vector>`, `hash/sha2.hpp`, `interfaces/onramp_provider.hpp`.
///
/// **Known issues.** One account; no partial fills.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_onramp_provider_reference_hpp
#define snicholls_core_interfaces_onramp_provider_reference_hpp

#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/onramp_provider.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The reference on-ramp venue.
    /// \ingroup core
    class memory_onramp_provider {
    public:
        /// \brief Reads the time.
        using clock_source = std::function<utc_s()>;
        /// \brief How long a quote lives.
        static constexpr std::chrono::seconds quote_lifetime{30};
        /// \brief The only account.
        static constexpr std::string_view account = "trading";

        /// \brief A venue delivering `assets`.
        /// \param assets What it can deliver, and on which chains.
        /// \param clock  The clock.
        memory_onramp_provider(std::vector<onramp_asset> assets, clock_source clock)
            : assets_(std::move(assets)), clock_(std::move(clock))
        {
        }

        /// \brief What it delivers.
        /// \return The assets.
        [[nodiscard]] std::vector<onramp_asset> assets() const { return assets_; }

        /// \brief A price for a ticket.
        /// \param q The request.
        /// \return The quote, or the error.
        capability_result<onramp_quote> quote(const onramp_quote_request& q)
        {
            if (auto e = take_injected())
                return err(*e);
            return price(q.unit, q.quote_currency, q.amount);
        }

        /// \brief Buys at the set price, at once, from the quote-currency balance.
        /// \param b The request.
        /// \return The filled order, or the error.
        capability_result<order> buy(const buy_request& b)
        {
            if (b.client_order_id.empty())
                return fail(capability_errc::invalid_argument, {}, "a buy needs a client order id");
            auto client = make_id<client_id>(b.client_order_id);
            if (!client)
                return err(client.unwrap_err());
            if (const auto it = buys_.find(b.client_order_id); it != buys_.end()) {
                if (!it->second.matches(b))
                    return fail(capability_errc::conflict, {}, "client order id reused for another buy");
                return ok(orders_.at(it->second.order_id));
            }
            if (auto e = take_injected())
                return err(*e);
            auto q = price(b.unit, b.quote_currency, b.amount);
            if (!q)
                return err(q.unwrap_err());
            if (b.max_price && greater(q->price, *b.max_price))
                return fail(capability_errc::rejected, {}, "the price is above max_price");
            if (!has(q->spend))
                return fail(capability_errc::insufficient_funds);
            move_balance(q->spend, true);
            move_balance(q->receive, false);
            const std::string n = std::to_string(++next_id_);
            order o{id<provider_id>("mem_ord_" + n), *client, order_status::filled, q->receive, q->spend, q->fee, clock_()};
            orders_.emplace(std::string(o.order_id.view()), o);
            onramp_fact f = fact("mem_trd_" + n, onramp_fact_kind::trade_fill);
            f.debit = venue(*subtract(q->spend, q->fee));
            f.credit = venue(q->receive);
            f.fee = venue(q->fee);
            f.price = q->price;
            f.order_id = o.order_id;
            f.client_order_id = o.client_order_id;
            fill_facts_.push_back(f);
            buys_.emplace(std::string(b.client_order_id),
                          kept_buy{b.unit, b.quote_currency, b.amount, b.max_price, std::string(o.order_id.view())});
            return ok(o);
        }

        /// \brief An order's state.
        /// \param order_id The id.
        /// \return The order, or `not_found`.
        capability_result<order> order_status(std::string_view order_id)
        {
            const auto it = orders_.find(order_id);
            if (it == orders_.end())
                return fail(capability_errc::not_found);
            return ok(it->second);
        }

        /// \brief Withdraws to an address: checks the chain, the address and the balance (the amount plus the fee),
        /// then broadcasts.
        /// \param w The request.
        /// \return The withdrawal (broadcast), or the error.
        capability_result<withdrawal> withdraw(const withdraw_request& w)
        {
            if (w.client_reference.empty())
                return fail(capability_errc::invalid_argument, {}, "a withdrawal needs a client reference");
            auto client = make_id<client_id>(w.client_reference);
            if (!client)
                return err(client.unwrap_err());
            if (const auto it = withdrawals_by_ref_.find(w.client_reference); it != withdrawals_by_ref_.end()) {
                if (!it->second.matches(w))
                    return fail(capability_errc::conflict, {}, "client reference reused for another withdrawal");
                return ok(withdrawals_.at(it->second.withdrawal_id));
            }
            if (auto e = take_injected())
                return err(*e);
            const auto a = std::find_if(assets_.begin(), assets_.end(),
                                        [&](const onramp_asset& x) { return x.unit == w.amount.asset_of(); });
            if (a == assets_.end())
                return fail(capability_errc::invalid_argument, {}, "the venue does not deliver that asset");
            if (std::find(a->chain_ids.begin(), a->chain_ids.end(), w.chain_id) == a->chain_ids.end())
                return fail(capability_errc::invalid_argument, {}, "the asset is not withdrawn on that chain");
            if (!is_evm_address(w.address))
                return fail(capability_errc::invalid_argument, {}, "the address is not an EVM address");
            if (w.amount.is_negative() || w.amount.is_zero())
                return fail(capability_errc::invalid_argument, {}, "a withdrawal is positive");
            const money fee = fee_for(w.amount.asset_of());
            const auto gross = add(w.amount, fee);
            if (!gross || !has(*gross))
                return fail(capability_errc::insufficient_funds);
            move_balance(*gross, true);
            withdrawal out;
            out.withdrawal_id = id<provider_id>("mem_wd_" + std::to_string(++next_id_));
            out.client_reference = *client;
            out.status = withdrawal_status::broadcast;
            out.amount = w.amount;
            out.fee = fee;
            out.chain_id = w.chain_id;
            out.address = id<short_text>(w.address);
            out.transaction_hash = id<provider_id>("0x" + std::string(sha256::hash(out.withdrawal_id.view()).hex().view()));
            out.updated_at = clock_();
            withdrawals_.emplace(std::string(out.withdrawal_id.view()), out);
            onramp_fact f = fact(std::string(out.withdrawal_id.view()), onramp_fact_kind::withdrawal);
            f.debit = venue(w.amount);
            f.credit = w.amount;   // the 1:1 move to the chain asset
            f.fee = venue(fee);
            f.withdrawal_id = out.withdrawal_id;
            f.client_reference = out.client_reference;
            f.chain_id = w.chain_id;
            f.address = out.address;
            f.transaction_hash = out.transaction_hash;
            withdrawal_facts_.push_back(f);
            withdrawals_by_ref_.emplace(std::string(w.client_reference),
                                        kept_withdrawal{w.amount, w.chain_id, std::string(w.address), std::string(w.memo),
                                                        std::string(out.withdrawal_id.view())});
            return ok(out);
        }

        /// \brief A withdrawal's state.
        /// \param withdrawal_id The id.
        /// \return The withdrawal, or `not_found`.
        capability_result<withdrawal> withdrawal_status(std::string_view withdrawal_id)
        {
            const auto it = withdrawals_.find(withdrawal_id);
            if (it == withdrawals_.end())
                return fail(capability_errc::not_found);
            return ok(it->second);
        }

        /// \brief Every balance, by asset code.
        /// \return The balances.
        capability_result<std::vector<venue_balance>> balances()
        {
            std::vector<venue_balance> out;
            for (const auto& [code, m] : balances_)
                out.push_back({id<short_text>(account), m, m});
            return ok(std::move(out));
        }

        /// \brief A page of facts, newest page first; the cursor is the count already read, from the newest.
        /// \param q The page.
        /// \return The facts, oldest first within the page, or `invalid_argument` for a bad limit or cursor.
        capability_result<onramp_statement_page> statement(const onramp_statement_query& q)
        {
            const auto& all = q.source == onramp_statement_source::fills ? fill_facts_ : withdrawal_facts_;
            if (q.limit == 0 || q.limit > 100)
                return fail(capability_errc::invalid_argument, {}, "a page holds 1 to 100 facts");
            std::size_t skip = 0;
            for (const char c : q.cursor) {
                if (c < '0' || c > '9' || skip > all.size())
                    return fail(capability_errc::invalid_argument, {}, "not a cursor this venue gave");
                skip = skip * 10 + static_cast<std::size_t>(c - '0');
            }
            onramp_statement_page page;
            const std::size_t end = all.size() > skip ? all.size() - skip : 0;
            const std::size_t begin = end > q.limit ? end - q.limit : 0;
            page.facts.assign(all.begin() + static_cast<std::ptrdiff_t>(begin), all.begin() + static_cast<std::ptrdiff_t>(end));
            if (begin > 0)
                page.next_cursor = id<provider_id>(std::to_string(skip + (end - begin)));
            return ok(std::move(page));
        }

        /// \brief The venue's own asset for `unit`: `<code>:MEM` at the same scale.
        /// \param unit The asset.
        /// \return The venue asset.
        [[nodiscard]] static asset venue_asset(const asset& unit)
        {
            return *make_asset(std::string(unit.code().substr(0, 20)) + ":MEM", unit.scale());
        }

        /// \brief Sets the price of `p.from()` in `p.to()` (the quote currency), in whole units.
        /// \param p The rate.
        void set_price(const rate& p) { prices_.insert_or_assign(pair_key(p.from(), p.to()), p); }

        /// \brief Funds a balance (a fiat deposit, say).
        /// \param m The amount.
        void deposit(const money& m) { move_balance(m, false); }

        /// \brief The fee on the quote side of a buy, in basis points.
        /// \param bps 0 to 10000.
        void set_fee_bps(std::uint32_t bps) noexcept { fee_bps_ = std::min<std::uint32_t>(bps, 10000); }

        /// \brief The network fee on a withdrawal of `fee`'s asset.
        /// \param fee The fee.
        void set_withdrawal_fee(const money& fee) { withdrawal_fees_.insert_or_assign(std::string(fee.asset_of().code()), fee); }

        /// \brief The withdrawal landed.
        /// \param withdrawal_id The id.
        /// \return The withdrawal, or `not_found`.
        capability_result<withdrawal> confirm(std::string_view withdrawal_id)
        {
            const auto it = withdrawals_.find(withdrawal_id);
            if (it == withdrawals_.end())
                return fail(capability_errc::not_found);
            it->second.status = withdrawal_status::confirmed;
            it->second.updated_at = clock_();
            return ok(it->second);
        }

        /// \brief Makes the next `quote`, `buy` or `withdraw` fail (once) as though the venue answered `response`.
        /// \param code     The error's code.
        /// \param response The venue's response: its first word becomes the provider code when it is one, and
        ///                 nothing else of it is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        // An id the venue makes itself, which always fits.
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

        static money venue(const money& m) { return money::from_units(venue_asset(m.asset_of()), m.magnitude(), m.is_negative()); }

        onramp_fact fact(const std::string& name, onramp_fact_kind k) const
        {
            onramp_fact f;
            f.source = "memory";
            f.provider_account = "memory";
            f.fact_id = id<fact_id>(name);
            f.kind = k;
            f.occurred_at = clock_();
            f.account = id<short_text>(account);
            return f;
        }

        static std::string pair_key(const asset& a, const asset& b)
        {
            return std::string(a.code()) + ":" + std::to_string(a.scale()) + "/" + std::string(b.code()) + ":" +
                   std::to_string(b.scale());
        }

        static bool greater(const rate& a, const rate& b) noexcept
        {
            const auto l = uint256::checked_mul(a.numerator(), b.denominator());
            const auto r = uint256::checked_mul(b.numerator(), a.denominator());
            return !l || (r && *l > *r);   // an overflowing left side counts as greater: refuse rather than fill
        }

        capability_result<onramp_quote> price(const asset& unit, const asset& quote_currency, const money& amount) const
        {
            if (amount.is_negative() || amount.is_zero())
                return fail(capability_errc::invalid_argument, {}, "an amount is positive");
            if (amount.asset_of() != unit && amount.asset_of() != quote_currency)
                return fail(capability_errc::invalid_argument, {}, "the amount is in neither asset");
            const auto p = prices_.find(pair_key(unit, quote_currency));
            if (p == prices_.end())
                return fail(capability_errc::unsupported, {}, "no price for that pair");
            money spend, receive, fee;
            if (amount.asset_of() == quote_currency) {
                spend = amount;
                fee = fee_on(amount);
                const auto per_asset = p->second.inverse();
                const auto net = subtract(amount, fee);
                const auto got = per_asset && net ? convert(*net, *per_asset, rounding::toward_zero)
                                                  : result<conversion, money_errc>(err(money_errc::overflow));
                if (!got)
                    return fail(capability_errc::invalid_argument, {}, "the amount cannot be converted");
                receive = got->converted;
            } else {
                receive = amount;
                const auto cost = convert(amount, p->second, rounding::ceiling);
                if (!cost)
                    return fail(capability_errc::invalid_argument, {}, "the amount cannot be converted");
                fee = fee_on(cost->converted);
                spend = *add(cost->converted, fee);
            }
            if (receive.is_zero())
                return fail(capability_errc::invalid_argument, {}, "the ticket buys nothing");
            const utc_s now = clock_();
            return ok(onramp_quote{p->second, spend, receive, fee, now, now + quote_lifetime});
        }

        money fee_on(const money& m) const
        {
            const auto r = rate::make(m.asset_of(), m.asset_of(), fee_bps_, 10000);
            const auto f = r ? convert(m, *r, rounding::ceiling) : result<conversion, money_errc>(err(money_errc::overflow));
            return f ? f->converted : money::zero(m.asset_of());
        }

        money fee_for(const asset& a) const
        {
            const auto it = withdrawal_fees_.find(a.code());
            return it == withdrawal_fees_.end() || it->second.asset_of() != a ? money::zero(a) : it->second;
        }

        bool has(const money& m) const
        {
            const auto it = balances_.find(m.asset_of().code());
            return it != balances_.end() && it->second.asset_of() == m.asset_of() && it->second.magnitude() >= m.magnitude();
        }

        void move_balance(const money& m, bool out)
        {
            auto it = balances_.find(m.asset_of().code());
            if (it == balances_.end())
                it = balances_.emplace(std::string(m.asset_of().code()), money::zero(m.asset_of())).first;
            it->second = out ? *subtract(it->second, m) : *add(it->second, m);
        }

        // What an idempotency key was first used for: owned copies of the borrowed requests.
        struct kept_buy {
            asset unit;
            asset quote_currency;
            money amount;
            std::optional<rate> max_price;
            std::string order_id;
            [[nodiscard]] bool matches(const buy_request& b) const
            {
                return unit == b.unit && quote_currency == b.quote_currency && amount == b.amount && max_price == b.max_price;
            }
        };
        struct kept_withdrawal {
            money amount;
            std::uint64_t chain_id{1};
            std::string address;
            std::string memo;
            std::string withdrawal_id;
            [[nodiscard]] bool matches(const withdraw_request& w) const
            {
                return amount == w.amount && chain_id == w.chain_id && address == w.address && memo == w.memo;
            }
        };

        std::vector<onramp_asset> assets_;
        clock_source clock_;
        std::uint32_t fee_bps_{0};
        std::uint64_t next_id_{0};
        std::map<std::string, rate, std::less<>> prices_;
        std::map<std::string, money, std::less<>> balances_;
        std::map<std::string, money, std::less<>> withdrawal_fees_;
        std::map<std::string, order, std::less<>> orders_;
        std::map<std::string, kept_buy, std::less<>> buys_;
        std::map<std::string, withdrawal, std::less<>> withdrawals_;
        std::map<std::string, kept_withdrawal, std::less<>> withdrawals_by_ref_;
        std::vector<onramp_fact> fill_facts_;
        std::vector<onramp_fact> withdrawal_facts_;
        std::optional<capability_error> injected_;
    };

    static_assert(onramp_provider<memory_onramp_provider>, "snicholls::memory_onramp_provider must model snicholls::onramp_provider");

} // namespace snicholls

#endif // snicholls_core_interfaces_onramp_provider_reference_hpp
