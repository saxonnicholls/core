//
//  business_day.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Business-day adjustment rules (following, modified following, preceding, modified preceding,
/// unadjusted) over a holiday calendar the caller supplies.
/// \ingroup core
///
/// **Overview.** A payment or fixing date that falls on a day the market is closed moves to a business day by a
/// rule the trade names (ISDA 2006 section 4.12):
/// - `following`: the first business day on or after the date;
/// - `modified_following`: following, unless that is in the next month, then preceding;
/// - `preceding`: the last business day on or before the date;
/// - `modified_preceding`: preceding, unless that is in the previous month, then following;
/// - `unadjusted`: the date as it is.
///
/// `adjust(day, convention, calendar)` applies one to any `holiday_calendar`. Core holds no holidays, which are
/// run-time data: `weekend_calendar` (Saturday and Sunday closed) is the reference calendar, and
/// `weekends_and_holidays{predicate}` makes a calendar from the caller's own holiday test, a set or a table
/// loaded at start-up.
///
/// \snippet finance/business_day_demo.cpp business_day
///
/// **Relationships.** `concepts/holiday_calendar.hpp`, `interfaces/holiday_calendar.hpp` (a calendar chosen at run
/// time), `time/business_days.hpp` (counting Monday-to-Friday days), `finance/day_count.hpp`. Group: \ref core.
///
/// **Implementations.** The five rules; `weekend_calendar`; `weekends_and_holidays<F>`.
///
/// **Guarantees.** constexpr, no allocation, `noexcept` when the calendar's `is_business_day` is. O(g) calls of
/// the calendar for a run of `g` closed days. A business day is returned unchanged by every rule.
///
/// **Errors.** `business_day_errc::no_business_day` when the search passes `max_business_day_search` closed days
/// in a row: a calendar with no open day in a year is broken, and the search stops rather than loop forever.
///
/// **Performance.** A weekend-only calendar adjusts in at most three tests; not benchmarked.
///
/// **Tests and demos.** `tests/finance/business_day_test.cpp`, against fixtures from QuantLib 1.44 over its
/// `WeekendsOnly` and `TARGET` calendars (`scripts/generate_finance_fixtures.py`,
/// `tests/fixtures/finance/business_day.txt`); `demos/finance/business_day_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstdint>`, `<string_view>`, `<utility>`, `concepts/holiday_calendar.hpp`,
/// `result.hpp`, `time/business_days.hpp`.
///
/// **Known issues.** QuantLib's `HalfMonthModifiedFollowing` and `Nearest` are not provided.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_finance_business_day_hpp
#define snicholls_core_finance_business_day_hpp

#include <snicholls/core/concepts/holiday_calendar.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/time/business_days.hpp>

#include <chrono>
#include <cstdint>
#include <string_view>
#include <utility>

namespace snicholls::finance {

    /// \brief How a date that is not a business day moves to one (ISDA 2006 4.12).
    /// \ingroup core
    enum class business_day_convention : std::uint8_t {
        unadjusted,           ///< the date as it is
        following,            ///< the first business day on or after
        modified_following,   ///< following, unless that is in the next month, then preceding
        preceding,            ///< the last business day on or before
        modified_preceding,   ///< preceding, unless that is in the previous month, then following
    };

    /// \brief Why an adjustment failed.
    /// \ingroup core
    enum class business_day_errc : std::uint8_t {
        no_business_day = 1,   ///< more than `max_business_day_search` closed days in a row
    };

    /// \brief The most closed days in a row an adjustment passes before it gives up: a year.
    /// \ingroup core
    inline constexpr int max_business_day_search = 366;

    /// \brief The reference calendar: Monday to Friday open, Saturday and Sunday closed, no holidays.
    /// \ingroup core
    struct weekend_calendar {
        /// \brief The calendar's name.
        /// \return `"weekends"`.
        static constexpr std::string_view name() noexcept { return "weekends"; }
        /// \brief Whether `day` is Monday to Friday.
        /// \param day The day.
        /// \return `true` for Monday to Friday.
        static constexpr bool is_business_day(std::chrono::sys_days day) noexcept { return snicholls::is_business_day(day); }
    };

    /// \brief A calendar from a holiday predicate: open Monday to Friday unless `is_holiday(day)`.
    /// \ingroup core
    /// \tparam F A callable `bool(std::chrono::sys_days)`: a lambda over the caller's set or table of holidays.
    template <typename F>
    struct weekends_and_holidays {
        F is_holiday;   ///< `true` for a weekday the market is closed

        /// \brief Whether `day` is a weekday and not a holiday.
        /// \param day The day.
        /// \return `true` when the market is open.
        constexpr bool is_business_day(std::chrono::sys_days day) const noexcept(noexcept(static_cast<bool>(is_holiday(day))))
        {
            return snicholls::is_business_day(day) && !static_cast<bool>(is_holiday(day));
        }
    };

    /// \brief Deduce the predicate's type: `weekends_and_holidays{[&](auto d) { return set.contains(d); }}`.
    template <typename F>
    weekends_and_holidays(F) -> weekends_and_holidays<F>;

    namespace detail {
        // The first business day at or after `day` stepping by `step` (+1 or -1) days, within the search limit.
        template <typename C>
        constexpr result<std::chrono::sys_days, business_day_errc> roll(std::chrono::sys_days day, int step, const C& calendar)
        {
            for (int i = 0; i <= max_business_day_search; ++i, day += std::chrono::days(step))
                if (static_cast<bool>(calendar.is_business_day(day)))
                    return ok(day);
            return err(business_day_errc::no_business_day);
        }

        constexpr std::chrono::month month_of(std::chrono::sys_days day) noexcept
        {
            return std::chrono::year_month_day(day).month();
        }

        // Roll by `step`; if that leaves the month, roll the other way.
        template <typename C>
        constexpr result<std::chrono::sys_days, business_day_errc> roll_modified(std::chrono::sys_days day, int step, const C& calendar)
        {
            const auto r = roll(day, step, calendar);
            if (r && month_of(*r) != month_of(day))
                return roll(day, -step, calendar);
            return r;
        }
    } // namespace detail

    /// \brief Move `day` to a business day of `calendar` by `convention`.
    /// \ingroup core
    /// \tparam C A `holiday_calendar`.
    /// \param day        The date.
    /// \param convention The rule.
    /// \param calendar   The calendar.
    /// \return The adjusted date (`day` itself when it is a business day or the rule is `unadjusted`), or
    /// `no_business_day`.
    template <holiday_calendar C>
    [[nodiscard]] constexpr result<std::chrono::sys_days, business_day_errc>
    adjust(std::chrono::sys_days day, business_day_convention convention, const C& calendar) noexcept(noexcept(calendar.is_business_day(day)))
    {
        switch (convention) {
        case business_day_convention::following: return detail::roll(day, 1, calendar);
        case business_day_convention::modified_following: return detail::roll_modified(day, 1, calendar);
        case business_day_convention::preceding: return detail::roll(day, -1, calendar);
        case business_day_convention::modified_preceding: return detail::roll_modified(day, -1, calendar);
        case business_day_convention::unadjusted: break;
        }
        return ok(day);
    }

    static_assert(holiday_calendar<weekend_calendar>, "snicholls::finance::weekend_calendar must model snicholls::holiday_calendar");

} // namespace snicholls::finance

#endif // snicholls_core_finance_business_day_hpp
