//
//  returns.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Performance measures: time-weighted return (chained sub-periods), modified Dietz, money-weighted return and
/// the internal rate of return, exact where the inputs allow and `double` where they cannot.
/// \ingroup core
///
/// **Overview.** A fund, a wealth account or a mandate reports how it did. Two questions, two families of measure:
///
/// - **How did the manager do?** The time-weighted return removes the effect of the investor's flows: the period is cut
///   at every external flow into sub-periods, each sub-period's growth is `end / (start + flow)`, and the growths are
///   chained, `TWR = prod(growth) - 1`. `period_return` and `time_weighted_return` compute it **exactly**, as a
///   `finance::ratio`, from `money` valuations; `time_weighted_return(std::span<const double>)` chains returns already
///   in `double`. The **modified Dietz** return approximates it without a valuation at each flow:
///   `(V1 - V0 - sum F) / (V0 + sum w F)`, each flow weighted by the share of the period it was invested
///   (`w = (days in period - days before the flow) / days in period`), also exact.
/// - **How did the investor do?** The money-weighted return is the rate that discounts the investor's flows (the
///   opening value and every contribution in, the closing value out) to zero: an internal rate of return.
///   `money_weighted_return` gives it annualised (ACT/365 Fixed exponents, as a spreadsheet's XIRR); `xirr` takes dated
///   flows; `irr` takes flows one period apart (numpy-financial's `irr`). A rate is a root of a polynomial in
///   `1 / (1 + r)`, irrational in general: these are `double`, found with Brent's method (`math/roots.hpp`).
///
/// **The root finder.** The flows' sign changes bound the number of rates (Descartes' rule of signs, which holds for
/// real exponents too). With none there is no rate (`no_root`). With one there is exactly one in `(-1, inf)`: it is
/// bracketed between `r = 0` and the end of the range where the earliest or the latest flow dominates, and Brent's
/// method closes on it. With more, the whole range is scanned for sign changes (1,024 steps in `1 / (1 + r)` for
/// `r >= 0` and in `1 + r` below) and each is closed with Brent; more than one is `multiple_roots`, and `irr_roots`
/// lists them. The range is `-0.999999 <= r <= 999,999`.
///
/// \snippet finance/returns_demo.cpp twr
///
/// \snippet finance/returns_demo.cpp irr
///
/// **Relationships.** `finance/ratio.hpp` (the exact answers), `money/money.hpp` (the inputs), `math/roots.hpp`
/// (`brent`), `math/constexpr_math.hpp` (`pow`). Above core: the fund library's performance reports. Group: \ref core.
///
/// **Implementations / variants.** Exact: `period_return`, `time_weighted_return` over valuations, `modified_dietz`.
/// In `double`: `time_weighted_return` over returns, `irr`, `irr_roots`, `xirr`, `money_weighted_return`.
///
/// **Guarantees.** `constexpr`, `noexcept`, no allocation (the flows are spans; `irr_roots` writes into the caller's
/// span). The exact functions return the exact value or an error, never a rounded one; amounts are minor units of one
/// asset, so the asset's scale cancels.
///
/// **Errors.** `returns_errc`, through `result`: `empty` (no periods or flows); `asset_mismatch` (amounts in two
/// assets); `zero_capital` (a sub-period, or Dietz's weighted capital, with nothing invested: zero or negative);
/// `out_of_range` (an amount beyond 64 bits of minor units, or an exact answer that does not fit a `ratio`: use the
/// `double` form); `invalid_dates` (an end before its start, a flow outside the period, dates out of order);
/// `no_root` (no rate in range: flows of one sign, or a root beyond the range); `multiple_roots` (more than one rate:
/// the IRR is not defined; read them with `irr_roots`).
///
/// **Performance.** Not benchmarked. One sign change (the usual case) costs two Brent searches over the flows; more
/// cost a scan of 2,048 evaluations over the flows.
///
/// **Tests and demos.** `tests/finance/returns_test.cpp`, against numpy-financial 1.0.0's `irr` and an exact
/// independent reference in Python's `fractions` and `decimal` (`scripts/generate_returns_fixtures.py`,
/// `tests/fixtures/finance/returns.txt`, CTest `generated.returns_fixtures_up_to_date`); `demos/finance/returns_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstddef>`, `<cstdint>`, `<initializer_list>`, `<limits>`, `<optional>`, `<span>`, `finance/ratio.hpp`,
/// `math/constexpr_math.hpp`, `math/roots.hpp`, `money/money.hpp`, `result.hpp`.
///
/// **Known issues.** A double root (the NPV touches zero without crossing) has no sign change and is not found. Two
/// rates closer together than a scan step are missed. A rate beyond the range is `no_root`. numpy-financial's `irr`
/// returns one of several roots where this header says `multiple_roots`: deliberately, since the IRR is then not a
/// return. The time-weighted return's flow is at the start of its sub-period; a flow at the end belongs to the next.
///
/// **Status.** New in 0.27.0. Experimental.

#ifndef snicholls_core_finance_returns_hpp
#define snicholls_core_finance_returns_hpp

#include <snicholls/core/finance/ratio.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/math/roots.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/result.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>

namespace snicholls::finance {

    /// \brief Why a return could not be computed.
    /// \ingroup core
    enum class returns_errc : std::uint8_t {
        empty = 1,           ///< no periods, or no flows
        asset_mismatch = 2,  ///< amounts in more than one asset
        zero_capital = 3,    ///< nothing invested in a sub-period: its capital is zero or negative
        out_of_range = 4,    ///< an amount beyond 64 bits of minor units, or an exact answer that does not fit a ratio
        invalid_dates = 5,   ///< an end before its start, a flow outside the period, or dates out of order
        no_root = 6,         ///< no rate in range: flows of one sign, or a rate beyond -0.999999 to 999,999
        multiple_roots = 7,  ///< more than one rate: the IRR is not defined
    };

    /// \brief One sub-period of a time-weighted return: the value at its start, the external flow at its start (in
    /// positive, out negative), and the value at its end, all of one asset.
    /// \ingroup core
    struct valuation {
        money start;  ///< the value before the flow
        money flow;   ///< the external flow at the start of the sub-period
        money end;    ///< the value at the end

        /// \brief Equality.
        friend constexpr bool operator==(const valuation&, const valuation&) noexcept = default;
    };

    /// \brief An external flow on a day, exact: in positive, out negative.
    /// \ingroup core
    struct money_flow {
        std::chrono::sys_days on{};  ///< the day
        money amount;                ///< the amount

        /// \brief Equality.
        friend constexpr bool operator==(const money_flow&, const money_flow&) noexcept = default;
    };

    /// \brief A flow on a day, in `double`, for `xirr`: received positive, paid negative.
    /// `money_flow` is the exact one.
    /// \ingroup core
    struct xirr_flow {
        std::chrono::sys_days on{};  ///< the day
        double amount = 0;           ///< the amount

        /// \brief Equality.
        friend constexpr bool operator==(const xirr_flow&, const xirr_flow&) noexcept = default;
    };

    namespace detail {
        // A money amount as signed minor units, when it fits in 64 bits.
        constexpr std::optional<std::int64_t> minor_units(const money& m) noexcept
        {
            const auto u = m.magnitude().to_uint64();
            if (!u || *u > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                return std::nullopt;
            const auto v = static_cast<std::int64_t>(*u);
            return m.is_negative() ? -v : v;
        }

        // A money amount as a double of minor units (the asset's scale cancels in a rate).
        constexpr double minor_units_double(const money& m) noexcept
        {
            double v = 0;
            for (std::size_t i = 4; i-- > 0;)
                v = v * 18446744073709551616.0 + static_cast<double>(m.magnitude().limb(i));
            return m.is_negative() ? -v : v;
        }

        constexpr std::optional<std::int64_t> mul(std::int64_t a, std::int64_t b) noexcept { return checked_mul(a, b); }
        constexpr std::optional<std::int64_t> sum(std::int64_t a, std::int64_t b) noexcept { return checked_add(a, b); }

        // The rate search's ends: 1 / (1 + r) and 1 + r from this up to 1.
        inline constexpr double rate_floor = 1e-6;
        inline constexpr int rate_steps = 1024;

        // The net present value's sign at x = 1 / (1 + r) in (0, 1] (r >= 0), as sum c x^t, or at y = 1 + r in (0, 1]
        // (r <= 0), as y^T times the value, sum c y^(T - t): both bounded, with the value's sign.
        template <class Flows>
        constexpr double npv_scaled(const Flows& f, bool below_zero, double v) noexcept
        {
            double s = 0;
            for (std::size_t i = 0; i < f.size(); ++i)
                s += f.amount(i) * math::pow(v, below_zero ? f.span() - f.time(i) : f.time(i));
            return s;
        }

        constexpr double to_rate(bool below_zero, double v) noexcept { return below_zero ? v - 1 : 1 / v - 1; }

        template <class Flows>
        constexpr double close(const Flows& f, bool below_zero, double lo, double hi) noexcept
        {
            const auto g = [&](double v) noexcept { return npv_scaled(f, below_zero, v); };
            return to_rate(below_zero, math::brent(g, lo, hi).x);
        }

        // Every rate in range, ascending; the count found (written up to out.size()).
        template <class Flows>
        constexpr std::size_t all_rates(const Flows& f, std::span<double> out) noexcept
        {
            std::size_t found = 0;
            const auto keep = [&](double r) noexcept {   // the smallest out.size() rates, ascending
                std::size_t j = found < out.size() ? found : out.size();
                if (j == out.size() && (j == 0 || out[j - 1] <= r)) {
                    ++found;
                    return;
                }
                if (j == out.size())
                    --j;   // the largest kept makes way
                for (; j > 0 && out[j - 1] > r; --j)
                    out[j] = out[j - 1];
                out[j] = r;
                ++found;
            };
            for (const bool below_zero : {true, false}) {
                double prev_v = rate_floor, prev_g = npv_scaled(f, below_zero, rate_floor);
                if (prev_g == 0)
                    keep(to_rate(below_zero, prev_v));
                for (int k = 1; k <= rate_steps; ++k) {
                    const double v = k == rate_steps ? 1.0 : rate_floor + (1 - rate_floor) * k / rate_steps;
                    const double g = npv_scaled(f, below_zero, v);
                    if (g == 0) {
                        if (!(below_zero && k == rate_steps))   // r = 0 is counted once, from above
                            keep(to_rate(below_zero, v));
                    } else if (prev_g != 0 && (prev_g < 0) != (g < 0)) {
                        keep(close(f, below_zero, prev_v, v));
                    }
                    prev_v = v;
                    prev_g = g;
                }
            }
            return found;
        }

        // The one rate, or why there is not one.
        template <class Flows>
        constexpr result<double, returns_errc> one_rate(const Flows& f) noexcept
        {
            int changes = 0, last = 0;
            for (std::size_t i = 0; i < f.size(); ++i) {
                const double a = f.amount(i);
                const int s = a > 0 ? 1 : a < 0 ? -1 : 0;
                if (s != 0 && last != 0 && s != last)
                    ++changes;
                if (s != 0)
                    last = s;
            }
            if (changes == 0)
                return err(returns_errc::no_root);
            if (changes == 1) {   // exactly one rate in (-1, inf): find which side of zero, and close on it
                const double at_zero = npv_scaled(f, false, 1);
                if (at_zero == 0)
                    return ok(0.0);
                for (const bool below_zero : {false, true}) {
                    const double edge = npv_scaled(f, below_zero, rate_floor);
                    if (edge != 0 && (edge < 0) != (at_zero < 0))
                        return ok(close(f, below_zero, rate_floor, 1.0));
                    if (edge == 0)
                        return ok(to_rate(below_zero, rate_floor));
                }
                return err(returns_errc::no_root);
            }
            double rates[2]{};
            const std::size_t n = all_rates(f, rates);
            if (n == 0)
                return err(returns_errc::no_root);
            if (n > 1)
                return err(returns_errc::multiple_roots);
            return ok(rates[0]);
        }

        struct periodic_flows {
            std::span<const double> flows;
            constexpr std::size_t size() const noexcept { return flows.size(); }
            constexpr double amount(std::size_t i) const noexcept { return flows[i]; }
            constexpr double time(std::size_t i) const noexcept { return static_cast<double>(i); }
            constexpr double span() const noexcept { return flows.empty() ? 0 : static_cast<double>(flows.size() - 1); }
        };

        struct dated_flows {
            std::span<const xirr_flow> flows;
            constexpr std::size_t size() const noexcept { return flows.size(); }
            constexpr double amount(std::size_t i) const noexcept { return flows[i].amount; }
            constexpr double time(std::size_t i) const noexcept
            {
                return static_cast<double>((flows[i].on - flows.front().on).count()) / 365.0;
            }
            constexpr double span() const noexcept { return time(flows.size() - 1); }
        };

        // The investor's flows: the opening value paid in, each contribution paid in, the closing value received.
        struct investor_flows {
            double start_value, end_value;
            std::span<const money_flow> flows;
            std::chrono::sys_days start, end;
            constexpr std::size_t size() const noexcept { return flows.size() + 2; }
            constexpr double amount(std::size_t i) const noexcept
            {
                return i == 0 ? -start_value : i == size() - 1 ? end_value : -minor_units_double(flows[i - 1].amount);
            }
            constexpr double time(std::size_t i) const noexcept
            {
                const std::chrono::sys_days on = i == 0 ? start : i == size() - 1 ? end : flows[i - 1].on;
                return static_cast<double>((on - start).count()) / 365.0;
            }
            constexpr double span() const noexcept { return time(size() - 1); }
        };

        // Flows within [start, end], in order, all of one asset.
        constexpr std::optional<returns_errc> check_period(const money& v0, const money& v1, std::span<const money_flow> flows,
                                                           std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            if (end <= start)
                return returns_errc::invalid_dates;
            if (v0.asset_of() != v1.asset_of())
                return returns_errc::asset_mismatch;
            std::chrono::sys_days previous = start;
            for (const money_flow& f : flows) {
                if (f.amount.asset_of() != v0.asset_of())
                    return returns_errc::asset_mismatch;
                if (f.on < previous || f.on > end)
                    return returns_errc::invalid_dates;
                previous = f.on;
            }
            return std::nullopt;
        }
    } // namespace detail

    /// \brief One sub-period's return, exact: `(end - start - flow) / (start + flow)`.
    /// \ingroup core
    /// \param v The sub-period.
    /// \return The return, or `asset_mismatch`, `out_of_range` or `zero_capital`.
    [[nodiscard]] constexpr result<ratio, returns_errc> period_return(const valuation& v) noexcept
    {
        if (v.flow.asset_of() != v.start.asset_of() || v.end.asset_of() != v.start.asset_of())
            return err(returns_errc::asset_mismatch);
        const auto s = detail::minor_units(v.start), f = detail::minor_units(v.flow), e = detail::minor_units(v.end);
        const auto capital = s && f ? detail::sum(*s, *f) : std::nullopt;
        const auto gain = capital && e ? checked_sub(*e, *capital) : std::nullopt;
        if (!gain)
            return err(returns_errc::out_of_range);
        if (*capital <= 0)
            return err(returns_errc::zero_capital);
        return ok(ratio{*gain, *capital});
    }

    /// \brief The time-weighted return of chained sub-periods, exact: `prod(end / (start + flow)) - 1`.
    /// \ingroup core
    /// \param periods The sub-periods, in order, all of one asset.
    /// \return The return; `empty`, `asset_mismatch`, `zero_capital`, or `out_of_range` when the exact chain leaves 64
    /// bits (then chain `period_return(v)->to_double()` with the `double` overload).
    [[nodiscard]] constexpr result<ratio, returns_errc> time_weighted_return(std::span<const valuation> periods) noexcept
    {
        if (periods.empty())
            return err(returns_errc::empty);
        ratio growth{1, 1};
        for (const valuation& v : periods) {
            if (v.start.asset_of() != periods.front().start.asset_of())
                return err(returns_errc::asset_mismatch);
            const auto r = period_return(v);
            if (!r)
                return err(r.unwrap_err_unchecked());
            const auto step = add(*r, ratio{1, 1});
            const auto chained = step ? multiply(growth, *step) : std::nullopt;
            if (!chained)
                return err(returns_errc::out_of_range);
            growth = *chained;
        }
        const auto twr = subtract(growth, ratio{1, 1});
        if (!twr)
            return err(returns_errc::out_of_range);
        return ok(*twr);
    }

    /// \brief The time-weighted return of sub-period returns already in `double`: `prod(1 + r) - 1`.
    /// \ingroup core
    /// \param period_returns The sub-periods' returns, in order.
    /// \return The chained return; zero for none.
    [[nodiscard]] constexpr double time_weighted_return(std::span<const double> period_returns) noexcept
    {
        double growth = 1;
        for (const double r : period_returns)
            growth *= 1 + r;
        return growth - 1;
    }

    /// \brief The modified Dietz return, exact: `(V1 - V0 - sum F) / (V0 + sum w F)`, `w = (D - d) / D` for a flow `d`
    /// days into a period of `D` days (a flow on the first day is invested all period, one on the last day not at all).
    /// \ingroup core
    /// \param start_value The value at the start of `start`.
    /// \param end_value   The value at `end`.
    /// \param flows       The external flows (in positive), in date order, within `[start, end]`.
    /// \param start       The first day.
    /// \param end         The last day, after `start`.
    /// \return The return; `invalid_dates`, `asset_mismatch`, `zero_capital` or `out_of_range`.
    [[nodiscard]] constexpr result<ratio, returns_errc> modified_dietz(const money& start_value, const money& end_value,
                                                                       std::span<const money_flow> flows,
                                                                       std::chrono::sys_days start,
                                                                       std::chrono::sys_days end) noexcept
    {
        if (const auto bad = detail::check_period(start_value, end_value, flows, start, end))
            return err(*bad);
        const std::int64_t days = (end - start).count();
        const auto v0 = detail::minor_units(start_value), v1 = detail::minor_units(end_value);
        if (!v0 || !v1)
            return err(returns_errc::out_of_range);
        auto gain = checked_sub(*v1, *v0);                // V1 - V0 - sum F
        auto capital = detail::mul(*v0, days);            // D V0 + sum (D - d) F
        for (const money_flow& f : flows) {
            const auto a = detail::minor_units(f.amount);
            const auto weighted = a ? detail::mul(*a, days - (f.on - start).count()) : std::nullopt;
            gain = gain && a ? checked_sub(*gain, *a) : std::nullopt;
            capital = capital && weighted ? detail::sum(*capital, *weighted) : std::nullopt;
        }
        const auto scaled = gain ? detail::mul(*gain, days) : std::nullopt;
        if (!scaled || !capital)
            return err(returns_errc::out_of_range);
        if (*capital <= 0)
            return err(returns_errc::zero_capital);
        return ok(ratio{*scaled, *capital});
    }

    /// \brief The internal rate of return of flows one period apart (numpy-financial's `irr`): the `r > -1` at which
    /// `sum c_t / (1 + r)^t == 0`.
    /// \ingroup core
    /// \param flows The flows, `flows[t]` at the end of period `t` (paid negative, received positive).
    /// \return The rate a period; `empty`, `no_root` or `multiple_roots`.
    [[nodiscard]] constexpr result<double, returns_errc> irr(std::span<const double> flows) noexcept
    {
        if (flows.empty())
            return err(returns_errc::empty);
        return detail::one_rate(detail::periodic_flows{flows});
    }

    /// \brief Every rate in range at which flows one period apart are worth zero, ascending.
    /// \ingroup core
    /// \param flows The flows, one period apart.
    /// \param out   Where to write the rates: the smallest `out.size()` of them.
    /// \return How many rates were found (more than `out.size()` when it was too small).
    [[nodiscard]] constexpr std::size_t irr_roots(std::span<const double> flows, std::span<double> out) noexcept
    {
        return flows.empty() ? 0 : detail::all_rates(detail::periodic_flows{flows}, out);
    }

    /// \brief The annual internal rate of return of dated flows (a spreadsheet's XIRR): the `r > -1` at which
    /// `sum c_i / (1 + r)^((d_i - d_0) / 365) == 0`.
    /// \ingroup core
    /// \param flows The flows, in date order.
    /// \return The annual rate; `empty`, `invalid_dates`, `no_root` or `multiple_roots`.
    [[nodiscard]] constexpr result<double, returns_errc> xirr(std::span<const xirr_flow> flows) noexcept
    {
        if (flows.empty())
            return err(returns_errc::empty);
        for (std::size_t i = 1; i < flows.size(); ++i)
            if (flows[i].on < flows[i - 1].on)
                return err(returns_errc::invalid_dates);
        return detail::one_rate(detail::dated_flows{flows});
    }

    /// \brief The money-weighted return, annualised: the XIRR of the investor's flows (the start value paid in on
    /// `start`, each external flow paid in on its day, the end value received on `end`).
    /// \ingroup core
    /// \param start_value The value at the start.
    /// \param end_value   The value at the end.
    /// \param flows       The external flows (in positive), in date order, within `[start, end]`.
    /// \param start       The first day.
    /// \param end         The last day, after `start`.
    /// \return The annual rate (the period's is `(1 + r)^(days / 365) - 1`); `invalid_dates`, `asset_mismatch`,
    /// `no_root` or `multiple_roots`.
    [[nodiscard]] constexpr result<double, returns_errc> money_weighted_return(const money& start_value, const money& end_value,
                                                                               std::span<const money_flow> flows,
                                                                               std::chrono::sys_days start,
                                                                               std::chrono::sys_days end) noexcept
    {
        if (const auto bad = detail::check_period(start_value, end_value, flows, start, end))
            return err(*bad);
        return detail::one_rate(detail::investor_flows{detail::minor_units_double(start_value),
                                                       detail::minor_units_double(end_value), flows, start, end});
    }

} // namespace snicholls::finance

#endif // snicholls_core_finance_returns_hpp
