//
//  business_days.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Weekday arithmetic on `std::chrono` calendar types.
/// \ingroup core
///
/// Weekday arithmetic on std::chrono calendar types: a Monday-to-Friday calendar, O(1), constexpr, noexcept.
///
/// \code
///   snicholls::is_business_day(t.days())                       // Monday..Friday
///   snicholls::add_business_days(d, 3)                         // Friday + 3 -> Wednesday
///   snicholls::business_days_between(a, b)                     // business days in [a, b) (negative if b < a)
///   snicholls::previous_weekday(d, std::chrono::Friday)        // strictly before d ("last Friday")
///   snicholls::add_months(2024y / January / 31d, 1)            // 2024-02-29: clamped to the end of the month
/// \endcode
///
/// Holidays are not modelled: a holiday calendar needs data per market (and time zones), which belong in the consuming library.

#ifndef snicholls_core_time_business_days_hpp
#define snicholls_core_time_business_days_hpp

#include <chrono>
#include <cstdint>

#include <snicholls/core/time/civil.hpp>

namespace snicholls {

/// \brief Whether the value is weekend.
/// \param wd the wd.
/// \return `true` if the value is weekend.
constexpr bool is_weekend(std::chrono::weekday wd) noexcept
{
    return wd == std::chrono::Saturday || wd == std::chrono::Sunday;
}

/// \brief Whether `d` is Monday to Friday.
/// \ingroup core
///
/// \overview
/// Weekday arithmetic on `std::chrono` calendar types: a Monday-to-Friday calendar (no holidays). See the file description above.
///
/// \relations
/// - `snicholls::add_business_days`, `snicholls::business_days_between`, `snicholls::next_weekday`, `snicholls::add_months`.
/// - Group: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - O(1), `constexpr`, `noexcept`.
///
/// \errors
/// None.
///
/// \examples
/// \snippet business_days_demo.cpp demo
///
/// \perf
/// Not benchmarked.
///
/// \testing
/// - Tests: tests/time/business_days_test.cpp.
/// - Demo: demos/time/business_days_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only .
///
/// \gotchas
/// - No holiday calendars: weekends only.
///
/// \status
/// Stable, pre-1.0.
///
/// \param d the day.
/// \return `true` for Monday to Friday.
constexpr bool is_business_day(std::chrono::sys_days d) noexcept
{
    return !is_weekend(std::chrono::weekday(d));
}

/// \brief The n-th business day after d (n > 0) or before d (n < 0).
///
/// D itself never counts and may be a weekend day.
///
/// n == 0 returns d unchanged.
/// \ingroup core
/// \param d the d.
/// \param n the count.
/// \return the n-th business day after d (n > 0) or before d (n < 0).
constexpr std::chrono::sys_days add_business_days(std::chrono::sys_days d, std::int64_t n) noexcept
{
    using std::chrono::days;
    if (n == 0) return d;
    const unsigned iso = std::chrono::weekday(d).iso_encoding();   // Monday 1 .. Sunday 7
    if (n > 0) {
        if (iso >= 6) d -= days(iso - 5);   // weekend: count from the Friday before
        const auto w = static_cast<std::int64_t>(std::chrono::weekday(d).iso_encoding() - 1);   // 0..4
        const std::int64_t rem = n % 5;
        return d + days(n / 5 * 7 + rem + (w + rem > 4 ? 2 : 0));
    }
    if (iso >= 6) d += days(8 - iso);   // weekend: count from the Monday after
    const auto w = static_cast<std::int64_t>(std::chrono::weekday(d).iso_encoding() - 1);
    const std::int64_t m = -n;
    const std::int64_t rem = m % 5;
    return d - days(m / 5 * 7 + rem + (w - rem < 0 ? 2 : 0));
}

/// \brief Business days in [a, b); negative when b < a.
///
/// business_days_between(a, add_business_days(a, n)) == n for every business day a.
/// \ingroup core
/// \param a the left operand.
/// \param b the right operand.
/// \return the result.
constexpr std::int64_t business_days_between(std::chrono::sys_days a, std::chrono::sys_days b) noexcept
{
    // f(x) = business days in [M, x) for a fixed Monday M (1970-01-05): 5 per whole week + min(rest, 5).
    constexpr std::int64_t monday = 4;
    const auto f = [](std::int64_t x) noexcept {
        const std::int64_t k = x - monday;
        const std::int64_t q = k >= 0 ? k / 7 : -((-k + 6) / 7);
        const std::int64_t r = k - q * 7;
        return q * 5 + (r < 5 ? r : 5);
    };
    return f(b.time_since_epoch().count()) - f(a.time_since_epoch().count());
}

/// \brief The latest day strictly before d that is a wd (d - 7 if d is one).
/// \ingroup core
/// \param d the d.
/// \param wd the wd.
/// \return the latest day strictly before d that is a wd (d - 7 if d is one).
constexpr std::chrono::sys_days previous_weekday(std::chrono::sys_days d, std::chrono::weekday wd) noexcept
{
    const auto back = std::chrono::weekday(d) - wd;   // [0, 6]
    return d - (back.count() == 0 ? std::chrono::days(7) : back);
}

/// \brief The earliest day strictly after d that is a wd (d + 7 if d is one).
/// \ingroup core
/// \param d the d.
/// \param wd the wd.
/// \return the earliest day strictly after d that is a wd (d + 7 if d is one).
constexpr std::chrono::sys_days next_weekday(std::chrono::sys_days d, std::chrono::weekday wd) noexcept
{
    const auto ahead = wd - std::chrono::weekday(d);   // [0, 6]
    return d + (ahead.count() == 0 ? std::chrono::days(7) : ahead);
}

/// \brief d + n months, the day clamped to the last day of the target month (Jan 31 + 1 month = Feb 28/29).
///
/// Precondition: d.ok().
/// \ingroup core
/// \param d the d.
/// \param n the count.
/// \return the result.
constexpr std::chrono::year_month_day add_months(std::chrono::year_month_day d, int n) noexcept
{
    const std::chrono::year_month ym = std::chrono::year_month(d.year(), d.month()) + std::chrono::months(n);
    const unsigned last = last_day_of_month(static_cast<int>(ym.year()), static_cast<unsigned>(ym.month()));
    const unsigned day = static_cast<unsigned>(d.day());
    return std::chrono::year_month_day(ym.year(), ym.month(), std::chrono::day(day < last ? day : last));
}

} // namespace snicholls

#endif /* snicholls_core_time_business_days_hpp */
