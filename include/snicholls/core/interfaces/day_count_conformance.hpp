//
//  day_count_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_day_count`: the conformance kit of the `day_count_convention` capability.
/// \ingroup core
///
/// **Overview.** Drives any `day_count_convention`, a type or an object behind `day_count_interface`, through the
/// contract every convention keeps, over every start and end in a window of days, and says which check failed:
/// - the name is not empty;
/// - a period of no days counts zero days and is zero years;
/// - for a fixed start, the count and the fraction never fall as the end moves later;
/// - the fraction is between 1/400 and 1/300 of a year per counted day, so a convention's denominator is a
///   year's worth of days (360 to 366).
///
/// No test framework is included; a test asserts that the result is empty.
///
/// \snippet interfaces/day_count_conformance_demo.cpp conformance
///
/// **Relationships.** `concepts/day_count.hpp`, `interfaces/day_count.hpp`, `finance/day_count.hpp`. Group:
/// \ref core.
///
/// **Implementations.** Not applicable.
///
/// **Guarantees.** constexpr. O(n^2) calls for a window of `n` days.
///
/// **Errors.** The returned message names the failed check.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/interfaces/day_count_conformance_test.cpp`,
/// `demos/interfaces/day_count_conformance_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstdint>`, `<optional>`, `<string_view>`, `concepts/day_count.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_day_count_conformance_hpp
#define snicholls_core_interfaces_day_count_conformance_hpp

#include <snicholls/core/concepts/day_count.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief Check `t` against the `day_count_convention` contract for every start and end in `[from, from + days)`.
    /// \param t    The implementation.
    /// \param from The first day of the window.
    /// \param days The window's length in days.
    /// \return `std::nullopt` if it conforms, or the failed check.
    template <day_count_convention T>
    [[nodiscard]] constexpr std::optional<std::string_view> check_day_count(const T& t, std::chrono::sys_days from, int days)
    {
        if (std::string_view{t.name()}.empty())
            return "day_count: name() is empty";
        for (int i = 0; i < days; ++i) {
            const std::chrono::sys_days a = from + std::chrono::days(i);
            if (t.day_count(a, a) != 0 || t.year_fraction(a, a) != finance::year_fraction{})
                return "day_count: a period of no days is not zero";
            std::int64_t count = 0;
            finance::year_fraction fraction{};
            for (int j = i + 1; j < days; ++j) {
                const std::chrono::sys_days b = from + std::chrono::days(j);
                const std::int64_t c = t.day_count(a, b);
                const finance::year_fraction f = t.year_fraction(a, b);
                if (c < count || f < fraction)
                    return "day_count: the count or the fraction falls as the end moves later";
                if (f < finance::year_fraction{c, 400} || f > finance::year_fraction{c, 300})
                    return "day_count: the fraction is not a year's worth of days per counted day";
                count = c;
                fraction = f;
            }
        }
        return std::nullopt;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_day_count_conformance_hpp
