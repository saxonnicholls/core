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
/// - `unadjusted`: the date as it is;
/// - `half_month_modified_following`: modified following, and also not across the 15th (QuantLib);
/// - `nearest`: the nearer business day, following on a tie (QuantLib).
///
/// Over any calendar too, from crtp's `calendar` (QuantLib's `Calendar`, 0.27.0): `is_end_of_month` and
/// `end_of_month` (the last business day of a month), `advance_business_days`, `advance_months` (calendar months,
/// then the convention, with the end-of-month rule), `count_business_days` (QuantLib's flags), `holidays_between`,
/// `joint_calendar` (holidays of either, or business days of either) and `adjusted_calendar` (a calendar with holidays
/// added and removed, without changing the one it wraps).
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
/// **Implementations.** The seven rules; `weekend_calendar`; `weekends_and_holidays<F>`; `joint_calendar<A, B>`;
/// `adjusted_calendar<C>`.
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
/// **Dependencies.** `<algorithm>`, `<chrono>`, `<cstdint>`, `<string_view>`, `<utility>`, `<vector>`,
/// `concepts/holiday_calendar.hpp`,
/// `result.hpp`, `time/business_days.hpp`.
///
/// **Known issues.** `holidays_between` allocates its vector. The two QuantLib rules added in 0.27.0 are checked
/// against hand-computed dates, not against QuantLib's fixtures.
///
/// **Status.** New in 0.23.0; the QuantLib rules and calendar operations in 0.27.0.

#ifndef snicholls_core_finance_business_day_hpp
#define snicholls_core_finance_business_day_hpp

#include <snicholls/core/concepts/holiday_calendar.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/time/business_days.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls::finance {

    /// \brief How a date that is not a business day moves to one (ISDA 2006 4.12).
    /// \ingroup core
    enum class business_day_convention : std::uint8_t {
        unadjusted,           ///< the date as it is
        following,            ///< the first business day on or after
        modified_following,   ///< following, unless that is in the next month, then preceding
        preceding,            ///< the last business day on or before
        modified_preceding,   ///< preceding, unless that is in the previous month, then following
        half_month_modified_following,   ///< modified following, and not from the first half of a month into the second
        nearest,              ///< the nearer business day; following on a tie
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
        case business_day_convention::half_month_modified_following: {
            const auto r = detail::roll_modified(day, 1, calendar);
            if (r && static_cast<unsigned>(std::chrono::year_month_day(day).day()) <= 15 &&
                static_cast<unsigned>(std::chrono::year_month_day(*r).day()) > 15)
                return detail::roll(day, -1, calendar);
            return r;
        }
        case business_day_convention::nearest:
            for (int i = 0; i <= max_business_day_search; ++i) {
                if (static_cast<bool>(calendar.is_business_day(day + std::chrono::days(i))))
                    return ok(day + std::chrono::days(i));
                if (static_cast<bool>(calendar.is_business_day(day - std::chrono::days(i))))
                    return ok(day - std::chrono::days(i));
            }
            return err(business_day_errc::no_business_day);
        case business_day_convention::unadjusted: break;
        }
        return ok(day);
    }

    /// \brief The last business day of `day`'s month.
    /// \ingroup core
    /// \param day      Any day of the month.
    /// \param calendar The calendar.
    /// \return It, or `no_business_day`.
    template <holiday_calendar C>
    [[nodiscard]] constexpr result<std::chrono::sys_days, business_day_errc> end_of_month(std::chrono::sys_days day, const C& calendar)
    {
        const std::chrono::year_month_day ymd(day);
        const std::chrono::sys_days last{ymd.year() / ymd.month() / std::chrono::last};
        return detail::roll(last, -1, calendar);
    }

    /// \brief Whether `day` is the last business day of its month (QuantLib's `isEndOfMonth`).
    /// \ingroup core
    /// \param day      The day.
    /// \param calendar The calendar.
    /// \return `true` when it is a business day and none follows in its month.
    template <holiday_calendar C>
    [[nodiscard]] constexpr bool is_end_of_month(std::chrono::sys_days day, const C& calendar)
    {
        const auto e = end_of_month(day, calendar);
        return e && *e == day;
    }

    /// \brief `day` moved by `n` business days (negative: backwards); `n == 0` is `day` itself.
    /// \ingroup core
    /// \param day      The start.
    /// \param n        Business days.
    /// \param calendar The calendar.
    /// \return The day, or `no_business_day`.
    template <holiday_calendar C>
    [[nodiscard]] constexpr result<std::chrono::sys_days, business_day_errc> advance_business_days(std::chrono::sys_days day, int n,
                                                                                                   const C& calendar)
    {
        const int step = n < 0 ? -1 : 1;
        for (; n != 0; n -= step) {
            const auto next = detail::roll(day + std::chrono::days(step), step, calendar);
            if (!next)
                return next;
            day = *next;
        }
        return ok(day);
    }

    /// \brief `day` plus `months` calendar months (clamped to the month's end), adjusted by `convention`; with the
    /// end-of-month rule, a start on the last business day of its month ends on the last business day of the target's
    /// (QuantLib's `Calendar::advance` for months and years).
    /// \ingroup core
    /// \param day               The start.
    /// \param months            The months (12 per year; negative: backwards).
    /// \param convention        The adjustment.
    /// \param end_of_month_rule Whether the end-of-month rule applies.
    /// \param calendar          The calendar.
    /// \return The day, or `no_business_day`.
    template <holiday_calendar C>
    [[nodiscard]] constexpr result<std::chrono::sys_days, business_day_errc>
    advance_months(std::chrono::sys_days day, int months, business_day_convention convention, bool end_of_month_rule, const C& calendar)
    {
        const std::chrono::sys_days target{add_months(std::chrono::year_month_day(day), months)};
        if (end_of_month_rule && is_end_of_month(day, calendar))
            return end_of_month(target, calendar);
        return adjust(target, convention, calendar);
    }

    /// \brief Business days between `from` and `to`, QuantLib's `businessDaysBetween`: `[from, to)` by default, negative
    /// when `to < from`.
    /// \ingroup core
    /// \param from          The first day.
    /// \param to            The last day.
    /// \param calendar      The calendar.
    /// \param include_first Whether `from` counts.
    /// \param include_last  Whether `to` counts.
    /// \return The count.
    template <holiday_calendar C>
    [[nodiscard]] constexpr std::int64_t count_business_days(std::chrono::sys_days from, std::chrono::sys_days to, const C& calendar,
                                                             bool include_first = true, bool include_last = false)
    {
        const auto count = [&calendar](std::chrono::sys_days a, std::chrono::sys_days b, bool first, bool last) {
            std::int64_t n = last && static_cast<bool>(calendar.is_business_day(b)) ? 1 : 0;
            for (auto x = first ? a : a + std::chrono::days(1); x < b; x += std::chrono::days(1))
                n += static_cast<bool>(calendar.is_business_day(x)) ? 1 : 0;
            return n;
        };
        if (from < to)
            return count(from, to, include_first, include_last);
        if (to < from)
            return -count(to, from, include_last, include_first);
        return include_first && include_last && static_cast<bool>(calendar.is_business_day(from)) ? 1 : 0;
    }

    /// \brief The days of `[from, to]` the calendar is closed on, weekends included only when asked (QuantLib's
    /// `holidayList`).
    /// \ingroup core
    /// \param from             The first day.
    /// \param to               The last day.
    /// \param calendar         The calendar.
    /// \param include_weekends Whether closed Saturdays and Sundays are listed.
    /// \return The days, in order.
    template <holiday_calendar C>
    [[nodiscard]] std::vector<std::chrono::sys_days> holidays_between(std::chrono::sys_days from, std::chrono::sys_days to,
                                                                      const C& calendar, bool include_weekends = false)
    {
        std::vector<std::chrono::sys_days> out;
        for (auto d = from; d <= to; d += std::chrono::days(1))
            if (!static_cast<bool>(calendar.is_business_day(d)) && (include_weekends || snicholls::is_business_day(d)))
                out.push_back(d);
        return out;
    }

    /// \brief How a `joint_calendar` combines two calendars (QuantLib's `JointCalendarRule`).
    /// \ingroup core
    enum class joint_rule : std::uint8_t {
        join_holidays,       ///< closed when either is closed (a settlement needing both markets)
        join_business_days,  ///< open when either is open
    };

    /// \brief Two calendars as one.
    /// \ingroup core
    /// \tparam A The first calendar.
    /// \tparam B The second.
    template <holiday_calendar A, holiday_calendar B>
    struct joint_calendar {
        A first;                                ///< the first calendar
        B second;                               ///< the second
        joint_rule rule{joint_rule::join_holidays};  ///< how they combine

        /// \brief Whether `day` is a business day of the pair.
        /// \param day The day.
        /// \return As the rule says.
        [[nodiscard]] constexpr bool is_business_day(std::chrono::sys_days day) const
        {
            const bool a = static_cast<bool>(first.is_business_day(day));
            const bool b = static_cast<bool>(second.is_business_day(day));
            return rule == joint_rule::join_holidays ? (a && b) : (a || b);
        }
    };

    /// \brief Deduces the calendars: `joint_calendar{sydney, new_york}`.
    template <typename A, typename B>
    joint_calendar(A, B) -> joint_calendar<A, B>;
    /// \brief Deduces the calendars: `joint_calendar{sydney, new_york, joint_rule::join_business_days}`.
    template <typename A, typename B>
    joint_calendar(A, B, joint_rule) -> joint_calendar<A, B>;

    /// \brief A calendar with holidays added and removed (QuantLib's `addHoliday` / `removeHoliday`), leaving the one
    /// it wraps unchanged, so calendars stay immutable and safe to share.
    /// \ingroup core
    /// \tparam C The calendar.
    template <holiday_calendar C>
    class adjusted_calendar {
    public:
        /// \brief `base` with `added` closed and `removed` open.
        /// \param base    The calendar.
        /// \param added   Days to close.
        /// \param removed Days to open (a removal wins over an addition of the same day).
        adjusted_calendar(C base, std::vector<std::chrono::sys_days> added, std::vector<std::chrono::sys_days> removed = {})
            : base_(std::move(base)), added_(std::move(added)), removed_(std::move(removed))
        {
            std::sort(added_.begin(), added_.end());
            std::sort(removed_.begin(), removed_.end());
        }

        /// \brief Whether `day` is a business day.
        /// \param day The day.
        /// \return `false` for an added holiday, `true` for a removed one, else the base calendar's answer.
        [[nodiscard]] bool is_business_day(std::chrono::sys_days day) const
        {
            if (std::binary_search(removed_.begin(), removed_.end(), day))
                return true;
            if (std::binary_search(added_.begin(), added_.end(), day))
                return false;
            return static_cast<bool>(base_.is_business_day(day));
        }

        /// \brief The calendar it wraps.
        /// \return It.
        [[nodiscard]] const C& base() const noexcept { return base_; }

    private:
        C base_;
        std::vector<std::chrono::sys_days> added_;
        std::vector<std::chrono::sys_days> removed_;
    };

    static_assert(holiday_calendar<weekend_calendar>, "snicholls::finance::weekend_calendar must model snicholls::holiday_calendar");
    static_assert(holiday_calendar<joint_calendar<weekend_calendar, weekend_calendar>>);
    static_assert(holiday_calendar<adjusted_calendar<weekend_calendar>>);

} // namespace snicholls::finance

#endif // snicholls_core_finance_business_day_hpp
