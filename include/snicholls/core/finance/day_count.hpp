//
//  day_count.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Day-count conventions as types: ACT/360, ACT/365 Fixed, 30/360, 30E/360 and ACT/ACT ISDA, each with an
/// exact year fraction.
/// \ingroup core
///
/// **Overview.** Each convention of the 2006 ISDA Definitions (section 4.16) that a money-market or swap leg
/// names is an empty type modelling `snicholls::day_count_convention`:
/// - `act_360` (4.16(e)): actual days / 360;
/// - `act_365_fixed` (4.16(d)): actual days / 365;
/// - `thirty_360` (4.16(f), "30/360, 360/360, Bond Basis"): a start on the 31st counts from the 30th; an end on
///   the 31st counts to the 30th when the start is the 30th or 31st;
/// - `thirty_e_360` (4.16(g), "30E/360, Eurobond Basis"): either date on the 31st counts as the 30th;
/// - `act_act_isda` (4.16(b)): the days in common years / 365 plus the days in leap years / 366.
///
/// `day_count(start, end)` is the days the convention counts, `year_fraction(start, end)` the exact
/// `finance::year_fraction`. A convention is chosen at compile time by naming its type, or at run time through
/// `day_count_interface` (`interfaces/day_count.hpp`). Data that names a convention (a repo's terms in a
/// `security_fact`, a bond's) holds a `day_count_basis`, one stable enumerator per type, and `accrual_fraction(basis,
/// start, end)` gives that type's fraction.
///
/// \snippet finance/day_count_demo.cpp day_count
///
/// \snippet finance/day_count_demo.cpp basis
///
/// **Relationships.** `concepts/day_count.hpp`, `interfaces/day_count.hpp`, `finance/year_fraction.hpp`,
/// `finance/business_day.hpp` (rolling a date onto a business day before counting), `time/civil.hpp`. Group:
/// \ref core.
///
/// **Implementations.** The five above. Not provided: 30E/360 ISDA and 30/360 US, which need the maturity date
/// or an end-of-February rule; ACT/ACT ICMA, which needs the coupon schedule; BUS/252, which needs a calendar.
///
/// **Guarantees.** constexpr, `noexcept`, O(1), no allocation; every convention is an empty, trivially copyable
/// type. `day_count(a, a)` and `year_fraction(a, a)` are zero. The ACT conventions are antisymmetric
/// (`f(b, a) == -f(a, b)`); the 30/360 family applies its rules to the dates as given, start first, as ISDA and
/// QuantLib do, and so is not.
///
/// **Errors.** None.
///
/// **Performance.** `benchmarks/finance/day_count_benchmark.cpp`: 4,096 date pairs up to 30 years apart, Release,
/// Apple clang 17, Xeon W-3245, 2026-11-23: `act_act_isda::year_fraction` 273 us against 260 us for the same
/// arithmetic written by hand, `thirty_360::year_fraction` 193 us against 203 us (about 65 and 47 ns a pair).
///
/// **Tests and demos.** `tests/finance/day_count_test.cpp`, against fixtures from QuantLib 1.44
/// (`scripts/generate_finance_fixtures.py`, `tests/fixtures/finance/day_count.txt`); `demos/finance/day_count_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstdint>`, `<string_view>`, `concepts/day_count.hpp`, `finance/year_fraction.hpp`,
/// `time/civil.hpp`.
///
/// **Known issues.** QuantLib's `Thirty360(ISDA)` is 30E/360 ISDA, not this header's `thirty_360`; QuantLib's
/// `Thirty360(BondBasis)` is.
///
/// **Status.** New in 0.23.0; `day_count_basis` and `accrual_fraction` new in 0.27.0.

#ifndef snicholls_core_finance_day_count_hpp
#define snicholls_core_finance_day_count_hpp

#include <snicholls/core/concepts/day_count.hpp>
#include <snicholls/core/finance/year_fraction.hpp>
#include <snicholls/core/time/civil.hpp>

#include <chrono>
#include <cstdint>
#include <string_view>

namespace snicholls::finance {

    namespace detail {
        constexpr std::int64_t serial(std::chrono::sys_days d) noexcept
        {
            return static_cast<std::int64_t>(d.time_since_epoch().count());
        }
        // 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1), after a convention has adjusted d1 and d2
        constexpr std::int64_t thirty(const civil_date& a, std::int64_t d1, const civil_date& b, std::int64_t d2) noexcept
        {
            return 360 * (b.year - a.year) + 30 * (static_cast<std::int64_t>(b.month) - static_cast<std::int64_t>(a.month)) + (d2 - d1);
        }
    } // namespace detail

    /// \brief ACT/360 (ISDA 2006 4.16(e)): actual days / 360.
    /// \ingroup core
    struct act_360 {
        /// \brief The convention's name.
        /// \return `"ACT/360"`.
        static constexpr std::string_view name() noexcept { return "ACT/360"; }
        /// \brief The days from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `end - start`, negative when `end` is earlier.
        static constexpr std::int64_t day_count(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            return detail::serial(end) - detail::serial(start);
        }
        /// \brief The fraction of a year from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `day_count / 360`, exactly.
        static constexpr finance::year_fraction year_fraction(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            return {day_count(start, end), 360};
        }
    };

    /// \brief ACT/365 Fixed (ISDA 2006 4.16(d)): actual days / 365.
    /// \ingroup core
    struct act_365_fixed {
        /// \brief The convention's name.
        /// \return `"ACT/365F"`.
        static constexpr std::string_view name() noexcept { return "ACT/365F"; }
        /// \brief The days from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `end - start`, negative when `end` is earlier.
        static constexpr std::int64_t day_count(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            return detail::serial(end) - detail::serial(start);
        }
        /// \brief The fraction of a year from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `day_count / 365`, exactly.
        static constexpr finance::year_fraction year_fraction(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            return {day_count(start, end), 365};
        }
    };

    /// \brief 30/360, Bond Basis (ISDA 2006 4.16(f)): a day of 31 counts as 30, at the end only after a start on the
    /// 30th or 31st.
    /// \ingroup core
    struct thirty_360 {
        /// \brief The convention's name.
        /// \return `"30/360"`.
        static constexpr std::string_view name() noexcept { return "30/360"; }
        /// \brief The days the convention counts from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `360 (Y2 - Y1) + 30 (M2 - M1) + (D2 - D1)` with D1 = min(D1, 30), and D2 = 30 when it is 31 and
        /// D1 is then 30.
        static constexpr std::int64_t day_count(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            const civil_date a = civil_from_days(detail::serial(start)), b = civil_from_days(detail::serial(end));
            const std::int64_t d1 = a.day == 31 ? 30 : a.day;
            const std::int64_t d2 = b.day == 31 && d1 == 30 ? 30 : b.day;
            return detail::thirty(a, d1, b, d2);
        }
        /// \brief The fraction of a year from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `day_count / 360`, exactly.
        static constexpr finance::year_fraction year_fraction(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            return {day_count(start, end), 360};
        }
    };

    /// \brief 30E/360, Eurobond Basis (ISDA 2006 4.16(g)): a day of 31 counts as 30, at either end.
    /// \ingroup core
    struct thirty_e_360 {
        /// \brief The convention's name.
        /// \return `"30E/360"`.
        static constexpr std::string_view name() noexcept { return "30E/360"; }
        /// \brief The days the convention counts from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `360 (Y2 - Y1) + 30 (M2 - M1) + (D2 - D1)` with D1 = min(D1, 30) and D2 = min(D2, 30).
        static constexpr std::int64_t day_count(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            const civil_date a = civil_from_days(detail::serial(start)), b = civil_from_days(detail::serial(end));
            return detail::thirty(a, a.day == 31 ? 30 : a.day, b, b.day == 31 ? 30 : b.day);
        }
        /// \brief The fraction of a year from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `day_count / 360`, exactly.
        static constexpr finance::year_fraction year_fraction(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            return {day_count(start, end), 360};
        }
    };

    /// \brief ACT/ACT ISDA (ISDA 2006 4.16(b)): the days in common years / 365 plus the days in leap years / 366.
    /// \ingroup core
    struct act_act_isda {
        /// \brief The convention's name.
        /// \return `"ACT/ACT ISDA"`.
        static constexpr std::string_view name() noexcept { return "ACT/ACT ISDA"; }
        /// \brief The days from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return `end - start`, negative when `end` is earlier.
        static constexpr std::int64_t day_count(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            return detail::serial(end) - detail::serial(start);
        }
        /// \brief The fraction of a year from `start` to `end`, over the common denominator 365 * 366.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return The exact fraction; `-year_fraction(end, start)` when `end` is earlier.
        static constexpr finance::year_fraction year_fraction(std::chrono::sys_days start, std::chrono::sys_days end) noexcept
        {
            std::int64_t a = detail::serial(start), b = detail::serial(end), sign = 1;
            if (b < a) {
                const std::int64_t t = a;
                a = b;
                b = t;
                sign = -1;
            }
            // 365 * 366 = 133590: a common-year day is 366 / 133590 of a year, a leap-year day 365 / 133590
            const auto weight = [](std::int64_t year) noexcept { return is_leap_year(year) ? std::int64_t{365} : std::int64_t{366}; };
            const std::int64_t y1 = civil_from_days(a).year, y2 = civil_from_days(b).year;
            if (y1 == y2)
                return {sign * (b - a) * weight(y1), 133590};
            const std::int64_t first = days_from_civil(y1 + 1, 1, 1) - a;    // to the end of the first year
            const std::int64_t last = b - days_from_civil(y2, 1, 1);         // from the start of the last year
            return {sign * ((y2 - y1 - 1) * 133590 + first * weight(y1) + last * weight(y2)), 133590};
        }
    };

    static_assert(day_count_convention<act_360> && day_count_convention<act_365_fixed> && day_count_convention<thirty_360> &&
                      day_count_convention<thirty_e_360> && day_count_convention<act_act_isda>,
                  "snicholls::finance: every day-count convention must model snicholls::day_count_convention");

    /// \brief A day-count convention as a value, for data that names one: each enumerator is one of the types above.
    /// Values are stable; new ones are appended.
    /// \ingroup core
    enum class day_count_basis : std::uint8_t {
        act_360 = 1,        ///< `act_360`
        act_365_fixed = 2,  ///< `act_365_fixed`
        thirty_360 = 3,     ///< `thirty_360`
        thirty_e_360 = 4,   ///< `thirty_e_360`
        act_act_isda = 5,   ///< `act_act_isda`
    };

    /// \brief The exact year fraction from `start` to `end` under the convention `basis` names.
    /// \ingroup core
    /// \param basis The convention.
    /// \param start The first day.
    /// \param end   The last day.
    /// \return That convention's `year_fraction(start, end)`; zero for a value that is not an enumerator.
    [[nodiscard]] constexpr finance::year_fraction accrual_fraction(day_count_basis basis, std::chrono::sys_days start,
                                                                    std::chrono::sys_days end) noexcept
    {
        switch (basis) {
        case day_count_basis::act_360: return act_360::year_fraction(start, end);
        case day_count_basis::act_365_fixed: return act_365_fixed::year_fraction(start, end);
        case day_count_basis::thirty_360: return thirty_360::year_fraction(start, end);
        case day_count_basis::thirty_e_360: return thirty_e_360::year_fraction(start, end);
        case day_count_basis::act_act_isda: return act_act_isda::year_fraction(start, end);
        }
        return {};
    }

} // namespace snicholls::finance

#endif // snicholls_core_finance_day_count_hpp
