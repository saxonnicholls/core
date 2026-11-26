//
//  holiday_calendar.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::holiday_calendar`: anything that says whether a day is a business day.
/// \ingroup core
///
/// **Overview.** Which days a market is open is run-time data: holidays are declared by governments and
/// exchanges, a year or two ahead, and change by decree. Core holds none of it. It holds the rules that use it
/// (`finance/business_day.hpp`: following, modified following, preceding, ...), written over this concept, so the
/// caller's calendar plugs in: anything whose `is_business_day(std::chrono::sys_days)` answers `bool`. It is the
/// concept half of a pair; the pure virtual half is `snicholls::holiday_calendar_interface`
/// (`interfaces/holiday_calendar.hpp`), for a calendar chosen at run time (by market, from a trade's terms).
///
/// \snippet concepts/holiday_calendar_demo.cpp holiday_calendar
///
/// **Relationships.** `finance/business_day.hpp` (the adjustment rules, `finance::weekend_calendar`, the reference
/// calendar, and `finance::weekends_and_holidays`, a calendar from a predicate; since 0.27.0 also crtp's calendar
/// operations over any model: `end_of_month`, `advance_business_days`, `advance_months`, `count_business_days`,
/// `holidays_between`, `joint_calendar`, `adjusted_calendar`), `interfaces/holiday_calendar.hpp`,
/// `interfaces/holiday_calendar_conformance.hpp`, `time/business_days.hpp` (Monday to Friday arithmetic).
/// Group: \ref core.
///
/// **Implementations.** `finance::weekend_calendar`, `finance::weekends_and_holidays<F>`,
/// `holiday_calendar_interface` and its adapter; the caller's market calendars.
///
/// **Guarantees.** A concept: no code, no cost.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/holiday_calendar_test.cpp`, `demos/concepts/holiday_calendar_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<concepts>`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0; the calendar operations over it in 0.27.0.

#ifndef snicholls_core_concepts_holiday_calendar_hpp
#define snicholls_core_concepts_holiday_calendar_hpp

#include <chrono>
#include <concepts>

namespace snicholls {

    /// \brief A calendar of business days: `is_business_day(day)` answers `bool`.
    template <typename T>
    concept holiday_calendar = requires(const T& c, std::chrono::sys_days d) {
        { c.is_business_day(d) } -> std::convertible_to<bool>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_holiday_calendar_hpp
