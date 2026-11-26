//
//  cash_ladder.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::finance::make_cash_ladder`: an opening balance and dated cash flows as a ladder of per-date
/// inflow, outflow, net and cumulative cash, the first date below a threshold, and an overdue bucket.
/// \ingroup core
///
/// **Overview.** A liquidity forecast: a provider that pays twice a week, stock-market settlements in flight and
/// card payouts due all move one cash account, and the question is whether the account stays above its floor on every
/// date ahead. `make_cash_ladder(flows, terms, rungs)` takes
/// - the flows, each a value date (`std::chrono::sys_days`) and a signed `money` (positive in, negative out), in any
///   order;
/// - the terms: today (`as_of`), the opening balance, the threshold and the overdue policy;
/// - the caller's span of rungs, which it fills: one rung per date that has a flow, in date order, each with that
///   date's inflow, outflow, net and the cumulative position at its close.
///
/// It returns the ladder: the filled rungs, the position at the start, the closing and the lowest position, the first
/// date the position is below the threshold, and the overdue bucket. A flow dated before `as_of` is overdue; what
/// happens to it is the caller's policy, set separately for inflows and outflows (`overdue_policy`):
/// - `overdue_treatment::count`: trusted to have happened, counted in the starting position;
/// - `overdue_treatment::report_only`: in the overdue bucket only, not in any position;
/// - `overdue_treatment::due_today`: moved to the `as_of` rung, as though it falls due today.
///
/// Every overdue flow is in the bucket whatever its treatment. The default `overdue_policy{}` is the prudent one,
/// `overdue_policy::prudent()`: overdue inflows reported but not counted (money late in is not money in), overdue
/// outflows due today (money late out is owed now).
///
/// \snippet finance/cash_ladder_demo.cpp ladder
///
/// Use it for a forecast over flows already known; it does not model, estimate or net across accounts. Not for more
/// than one asset: a ladder is one account's currency, and a flow, an opening or a threshold of another asset is
/// `money_errc::asset_mismatch` (convert first, with `money/rate.hpp`, and say at which rate).
///
/// **Relationships.** `money/money.hpp` (`money`, `add`, `compare`, `money_errc`); a ledger's value-dated pending
/// settlements feed it, and its buffers and breaks (a liquidity shortfall) are the ledger's. `finance/business_day.hpp`
/// moves a value date to a business day before it reaches the ladder. Group: \ref core.
///
/// **Implementations / variants.** One function; the overdue policy is its variation.
///
/// **Guarantees.**
/// - Exact: every sum is `money` arithmetic over `uint256` minor units; nothing is rounded or converted.
/// - `constexpr`, `noexcept`, and no allocation: the rungs are written into the caller's span, and the returned
///   ladder's `rungs` is the filled prefix of it. `flows.size() + 1` rungs are always enough.
/// - Rungs are in strictly increasing date order, every one on or after `as_of`. Each rung's `inflow` and `outflow`
///   are not negative, `net == inflow - outflow`, and `cumulative` is the start plus every net up to and including it.
/// - `first_below` is `as_of` when the starting position is already below the threshold, else the first rung whose
///   cumulative is below it, else empty. "Below" is strict: a position equal to the threshold is not a breach.
/// - On an error the span's contents are unspecified and nothing else is changed.
/// - O(n log k + n k) for n flows on k dates in the worst case (an insertion shifts later rungs), O(n log k) when the
///   flows come in date order.
///
/// **Errors.** `money_errc`:
/// - `asset_mismatch`: a flow, the threshold or the opening is of another asset;
/// - `overflow`: a sum exceeds 2^256 - 1 minor units;
/// - `buffer_too_small`: the flows fall on more dates than the span has rungs.
///
/// **Performance.** Not benchmarked: one pass of `money` additions per flow and one per rung.
///
/// **Tests and demos.** `tests/finance/cash_ladder_test.cpp` (no flows, every flow overdue under each treatment,
/// crossing the threshold, exact arithmetic past 64 bits, unsorted input, each error, at compile time),
/// `demos/finance/cash_ladder_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<compare>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<span>`, `money/money.hpp`,
/// `result.hpp`.
///
/// **Known issues.** A zero-amount flow still makes a rung for its date (an inflow of zero). Dates are value dates as
/// given: a flow on a weekend stays on the weekend unless the caller adjusts it.
///
/// **Status.** New in 0.25.0. Experimental.

#ifndef snicholls_core_finance_cash_ladder_hpp
#define snicholls_core_finance_cash_ladder_hpp

#include <snicholls/core/money/money.hpp>
#include <snicholls/core/result.hpp>

#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace snicholls::finance {

    /// \brief One dated cash flow: positive in, negative out.
    /// \ingroup core
    struct cash_flow {
        std::chrono::sys_days on{};  ///< the value date
        money amount{};              ///< the signed amount, of the ladder's asset

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const cash_flow&, const cash_flow&) noexcept = default;
    };

    /// \brief What a ladder does with a flow dated before `as_of`.
    /// \ingroup core
    enum class overdue_treatment : std::uint8_t {
        count = 1,        ///< trusted to have happened: counted in the starting position
        report_only = 2,  ///< in the overdue bucket only, never in a position
        due_today = 3,    ///< moved to the `as_of` rung
    };

    /// \brief The overdue treatment, set separately for inflows and outflows. The default is the prudent policy.
    /// \ingroup core
    struct overdue_policy {
        overdue_treatment inflows = overdue_treatment::report_only;  ///< for an overdue flow in
        overdue_treatment outflows = overdue_treatment::due_today;   ///< for an overdue flow out

        /// \brief The prudent policy: overdue inflows reported but not counted, overdue outflows due today.
        /// \return `{report_only, due_today}`, the same as `overdue_policy{}`.
        [[nodiscard]] static constexpr overdue_policy prudent() noexcept { return {}; }

        /// \brief One treatment for both directions.
        /// \param t The treatment.
        /// \return `{t, t}`.
        [[nodiscard]] static constexpr overdue_policy both(overdue_treatment t) noexcept { return {t, t}; }

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const overdue_policy&, const overdue_policy&) noexcept = default;
    };

    /// \brief What a ladder is built against: today, the opening balance, the floor and the overdue policy.
    /// \ingroup core
    struct cash_ladder_terms {
        std::chrono::sys_days as_of{};  ///< today: a flow dated before it is overdue
        money opening{};                ///< the balance at the start of `as_of`; its asset is the ladder's
        money threshold{};              ///< the floor: a position strictly below it is a breach
        overdue_policy overdue{};       ///< what overdue flows do; prudent by default
    };

    /// \brief One date of the ladder.
    /// \ingroup core
    struct cash_rung {
        std::chrono::sys_days on{};  ///< the date
        money inflow{};              ///< the sum of the flows in, not negative
        money outflow{};             ///< the sum of the flows out, as a magnitude, not negative
        money net{};                 ///< `inflow - outflow`
        money cumulative{};          ///< the position at the close of the date

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const cash_rung&, const cash_rung&) noexcept = default;
    };

    /// \brief The overdue flows: every flow dated before `as_of`, whatever its treatment.
    /// \ingroup core
    struct overdue_bucket {
        money inflow{};           ///< the sum of the overdue flows in
        money outflow{};          ///< the sum of the overdue flows out, as a magnitude
        std::size_t inflows = 0;  ///< how many flows in are overdue
        std::size_t outflows = 0; ///< how many flows out are overdue

        /// \brief Whether nothing is overdue.
        /// \return `inflows == 0 && outflows == 0`.
        [[nodiscard]] constexpr bool empty() const noexcept { return inflows == 0 && outflows == 0; }

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const overdue_bucket&, const overdue_bucket&) noexcept = default;
    };

    /// \brief A built ladder. `rungs` is the filled prefix of the caller's span.
    /// \ingroup core
    struct cash_ladder {
        std::span<cash_rung> rungs;                          ///< one per date with a flow, in date order
        money start{};                                       ///< the opening plus the overdue flows counted
        money closing{};                                     ///< the last rung's cumulative, or `start`
        money lowest{};                                      ///< the lowest of `start` and every cumulative
        std::optional<std::chrono::sys_days> first_below{};  ///< the first date below the threshold, if any
        overdue_bucket overdue{};                            ///< every overdue flow

        /// \brief Whether the position stays at or above the threshold on every date.
        /// \return `!first_below`.
        [[nodiscard]] constexpr bool stays_above() const noexcept { return !first_below.has_value(); }
    };

    namespace detail {
        // a += b, keeping the first error
        constexpr bool ladder_add(money& a, const money& b, money_errc& e) noexcept
        {
            auto s = add(a, b);
            if (!s) {
                e = s.unwrap_err();
                return false;
            }
            a = *s;
            return true;
        }

        constexpr bool ladder_below(const money& a, const money& b) noexcept { return *compare(a, b) < 0; }
    } // namespace detail

    /// \brief Builds the ladder of `flows` against `terms`, writing the rungs into `rungs`.
    /// \ingroup core
    /// \param flows The dated flows, in any order, every one of `terms.opening`'s asset.
    /// \param terms Today, the opening balance, the threshold and the overdue policy.
    /// \param rungs Where the rungs go; `flows.size() + 1` is always enough.
    /// \return The ladder, or `asset_mismatch`, `overflow` or `buffer_too_small`.
    [[nodiscard]] constexpr result<cash_ladder, money_errc> make_cash_ladder(std::span<const cash_flow> flows,
                                                                            const cash_ladder_terms& terms,
                                                                            std::span<cash_rung> rungs) noexcept
    {
        const asset& a = terms.opening.asset_of();
        if (terms.threshold.asset_of() != a)
            return err(money_errc::asset_mismatch);
        const money zero = money::zero(a);
        cash_ladder out;
        out.start = terms.opening;
        out.overdue = {zero, zero, 0, 0};
        money_errc e{};
        std::size_t used = 0;

        // the rung for date d, inserted in order when it is new; nullptr when the span is full
        const auto rung_for = [&](std::chrono::sys_days d) -> cash_rung* {
            std::size_t lo = 0, hi = used;
            while (lo < hi) {
                const std::size_t mid = lo + (hi - lo) / 2;
                if (rungs[mid].on < d)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo < used && rungs[lo].on == d)
                return &rungs[lo];
            if (used == rungs.size())
                return nullptr;
            for (std::size_t i = used; i > lo; --i)
                rungs[i] = rungs[i - 1];
            rungs[lo] = cash_rung{d, zero, zero, zero, zero};
            ++used;
            return &rungs[lo];
        };

        for (const cash_flow& f : flows) {
            if (f.amount.asset_of() != a)
                return err(money_errc::asset_mismatch);
            const bool in = !f.amount.is_negative();
            std::chrono::sys_days on = f.on;
            if (on < terms.as_of) {
                if (!detail::ladder_add(in ? out.overdue.inflow : out.overdue.outflow, in ? f.amount : -f.amount, e))
                    return err(e);
                ++(in ? out.overdue.inflows : out.overdue.outflows);
                const overdue_treatment t = in ? terms.overdue.inflows : terms.overdue.outflows;
                if (t == overdue_treatment::report_only)
                    continue;
                if (t == overdue_treatment::count) {
                    if (!detail::ladder_add(out.start, f.amount, e))
                        return err(e);
                    continue;
                }
                on = terms.as_of;   // due today
            }
            cash_rung* r = rung_for(on);
            if (r == nullptr)
                return err(money_errc::buffer_too_small);
            if (!detail::ladder_add(in ? r->inflow : r->outflow, in ? f.amount : -f.amount, e))
                return err(e);
        }

        money position = out.start;
        out.lowest = position;
        if (detail::ladder_below(position, terms.threshold))
            out.first_below = terms.as_of;
        for (std::size_t i = 0; i < used; ++i) {
            cash_rung& r = rungs[i];
            const auto net = subtract(r.inflow, r.outflow);
            if (!net)
                return err(net.unwrap_err());
            r.net = *net;
            if (!detail::ladder_add(position, r.net, e))
                return err(e);
            r.cumulative = position;
            if (detail::ladder_below(position, out.lowest))
                out.lowest = position;
            if (!out.first_below && detail::ladder_below(position, terms.threshold))
                out.first_below = r.on;
        }
        out.closing = position;
        out.rungs = rungs.first(used);
        return ok(out);
    }

} // namespace snicholls::finance

#endif // snicholls_core_finance_cash_ladder_hpp
