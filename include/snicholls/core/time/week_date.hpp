//
//  week_date.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief ISO 8601 week dates: the week-based year, the week number and the weekday of a day, both ways, constexpr.
/// \ingroup core
///
/// **Overview.** ISO 8601 numbers weeks from Monday, and week 1 of a year is the week with the year's first
/// Thursday in it (equivalently, the week of 4 January). A year has 52 or 53 weeks, and the days of late December
/// and early January can belong to the neighbouring week-based year: 2024-12-30 is 2025-W01-1. Settlement cycles,
/// reporting calendars and futures expiries are written this way.
/// - `to_iso_week_date(day)` gives the `iso_week_date` of a `std::chrono::sys_days`;
/// - `to_sys_days(week_date)` goes back, for a valid week date;
/// - `make_iso_week_date(year, week, weekday)` validates first, and refuses week 53 of a 52-week year;
/// - `iso_weeks_in_year(year)` is 52 or 53.
///
/// \snippet time/week_date_demo.cpp week_date
///
/// **Relationships.** `time/civil.hpp` (the day arithmetic underneath), `time/names.hpp` (weekday names),
/// `time/business_days.hpp`, `time/utc.hpp` (`utc<D>::days()` gives the `sys_days`). Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** constexpr, `noexcept`, O(1), no allocation. `to_sys_days(to_iso_week_date(d)) == d` for every
/// day (tested over 1600 to 2400 in full), and a week date orders as its day does.
///
/// **Errors.** `make_iso_week_date` returns `week_date_errc::invalid_week` or `invalid_weekday`.
///
/// **Performance.** Two `days_from_civil` and one `civil_from_days`; not benchmarked.
///
/// **Tests and demos.** `tests/time/week_date_test.cpp`, against Python's `datetime.date.isocalendar()` and
/// `fromisocalendar()` (`scripts/generate_week_date_fixtures.py`, `tests/fixtures/time/week_date.txt`);
/// `demos/time/week_date_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<compare>`, `<cstdint>`, `result.hpp`, `time/civil.hpp`.
///
/// **Known issues.** The year is a `std::chrono::sys_days` year: within +-32767, the range of `std::chrono::year`.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_time_week_date_hpp
#define snicholls_core_time_week_date_hpp

#include <snicholls/core/result.hpp>
#include <snicholls/core/time/civil.hpp>

#include <chrono>
#include <compare>
#include <cstdint>

namespace snicholls {

    /// \brief An ISO 8601 week date: week-based year, week 1 to 53, weekday 1 (Monday) to 7 (Sunday).
    /// \ingroup core
    struct iso_week_date {
        std::int32_t year = 1970;   ///< The week-based year, which can differ from the calendar year near 1 January.
        unsigned week = 1;          ///< The week, 1 to 52 or 53.
        unsigned weekday = 4;       ///< The weekday, 1 (Monday) to 7 (Sunday), as `std::chrono::weekday::iso_encoding()`.

        /// \brief Equality.
        /// \return Whether the two are the same week date.
        friend constexpr bool operator==(const iso_week_date&, const iso_week_date&) noexcept = default;
        /// \brief Order: by year, then week, then weekday, which is the order of the days.
        /// \return The order.
        friend constexpr std::strong_ordering operator<=>(const iso_week_date&, const iso_week_date&) noexcept = default;
    };

    /// \brief Why a week date was refused.
    /// \ingroup core
    enum class week_date_errc : std::uint8_t {
        invalid_week = 1,   ///< the week is 0, or past the year's 52 or 53 weeks
        invalid_weekday,    ///< the weekday is not 1 to 7
    };

    namespace detail {
        // Days since 1970-01-01 of the Monday of week 1 of week-based year y: the Monday on or before 4 January.
        constexpr std::int64_t iso_week_one(std::int64_t y) noexcept
        {
            const std::int64_t jan4 = days_from_civil(y, 1, 4);
            const unsigned wd = weekday_from_days(jan4);   // 0 = Sunday
            return jan4 - (wd == 0 ? 6 : wd - 1);
        }
    } // namespace detail

    /// \brief The number of ISO weeks in week-based year `year`: 53 when it starts on a Thursday, or is a leap year
    /// starting on a Wednesday; otherwise 52.
    /// \ingroup core
    /// \param year The week-based year.
    /// \return 52 or 53.
    [[nodiscard]] constexpr unsigned iso_weeks_in_year(std::int32_t year) noexcept
    {
        return static_cast<unsigned>((detail::iso_week_one(year + std::int64_t{1}) - detail::iso_week_one(year)) / 7);
    }

    /// \brief The ISO week date of `day`.
    /// \ingroup core
    /// \param day The day.
    /// \return Its week-based year, week and weekday.
    [[nodiscard]] constexpr iso_week_date to_iso_week_date(std::chrono::sys_days day) noexcept
    {
        const auto z = static_cast<std::int64_t>(day.time_since_epoch().count());
        std::int64_t y = civil_from_days(z).year;
        if (z >= detail::iso_week_one(y + 1))
            ++y;
        else if (z < detail::iso_week_one(y))
            --y;
        const unsigned wd = weekday_from_days(z);
        return {static_cast<std::int32_t>(y), static_cast<unsigned>((z - detail::iso_week_one(y)) / 7 + 1), wd == 0 ? 7U : wd};
    }

    /// \brief The day of a valid ISO week date.
    /// \ingroup core
    /// \pre `w` is valid: `make_iso_week_date` accepts it.
    /// \param w The week date.
    /// \return The day.
    [[nodiscard]] constexpr std::chrono::sys_days to_sys_days(const iso_week_date& w) noexcept
    {
        const std::int64_t z = detail::iso_week_one(w.year) + (static_cast<std::int64_t>(w.week) - 1) * 7 + (static_cast<std::int64_t>(w.weekday) - 1);
        return std::chrono::sys_days{std::chrono::days{static_cast<std::chrono::days::rep>(z)}};
    }

    /// \brief A validated ISO week date.
    /// \ingroup core
    /// \param year    The week-based year.
    /// \param week    The week, 1 to `iso_weeks_in_year(year)`.
    /// \param weekday The weekday, 1 (Monday) to 7 (Sunday).
    /// \return The week date, or `invalid_week` or `invalid_weekday`.
    [[nodiscard]] constexpr result<iso_week_date, week_date_errc> make_iso_week_date(std::int32_t year, unsigned week, unsigned weekday) noexcept
    {
        if (week < 1 || week > iso_weeks_in_year(year))
            return err(week_date_errc::invalid_week);
        if (weekday < 1 || weekday > 7)
            return err(week_date_errc::invalid_weekday);
        return ok(iso_week_date{year, week, weekday});
    }

} // namespace snicholls

#endif // snicholls_core_time_week_date_hpp
