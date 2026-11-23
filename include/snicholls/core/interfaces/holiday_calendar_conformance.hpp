//
//  holiday_calendar_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_holiday_calendar`: the conformance kit of the `holiday_calendar` capability.
/// \ingroup core
///
/// **Overview.** Drives any `holiday_calendar`, a type or an object behind `holiday_calendar_interface`, over a
/// window of days and says which check failed:
/// - the answer for a day is the same on every call;
/// - every day of the window has a business day within `finance::max_business_day_search` days on both sides, so
///   no adjustment rule of `finance/business_day.hpp` fails on it.
///
/// No test framework is included; a test asserts that the result is empty.
///
/// \snippet interfaces/holiday_calendar_conformance_demo.cpp conformance
///
/// **Relationships.** `concepts/holiday_calendar.hpp`, `interfaces/holiday_calendar.hpp`,
/// `finance/business_day.hpp`. Group: \ref core.
///
/// **Implementations.** Not applicable.
///
/// **Guarantees.** constexpr. O(n g) calls for a window of `n` days whose longest run of closed days is `g`.
///
/// **Errors.** The returned message names the failed check.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/interfaces/holiday_calendar_conformance_test.cpp`,
/// `demos/interfaces/holiday_calendar_conformance_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<optional>`, `<string_view>`, `concepts/holiday_calendar.hpp`,
/// `finance/business_day.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_holiday_calendar_conformance_hpp
#define snicholls_core_interfaces_holiday_calendar_conformance_hpp

#include <snicholls/core/concepts/holiday_calendar.hpp>
#include <snicholls/core/finance/business_day.hpp>

#include <chrono>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief Check `c` against the `holiday_calendar` contract for every day in `[from, from + days)`.
    /// \param c    The implementation.
    /// \param from The first day of the window.
    /// \param days The window's length in days.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <holiday_calendar T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_holiday_calendar(const T& c, std::chrono::sys_days from, int days)
    {
        for (int i = 0; i < days; ++i) {
            const std::chrono::sys_days d = from + std::chrono::days(i);
            if (static_cast<bool>(c.is_business_day(d)) != static_cast<bool>(c.is_business_day(d)))
                return "holiday_calendar: is_business_day() changes between calls";
            if (!finance::adjust(d, finance::business_day_convention::following, c) ||
                !finance::adjust(d, finance::business_day_convention::preceding, c))
                return "holiday_calendar: no business day within finance::max_business_day_search days";
        }
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_holiday_calendar_conformance_hpp
