//
//  day_count.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::day_count_convention`: anything that counts the days between two dates and turns them into an
/// exact fraction of a year.
/// \ingroup core
///
/// **Overview.** Interest accrues over a fraction of a year, and the market agrees how to count it: ACT/360,
/// ACT/365 Fixed, 30/360, 30E/360, ACT/ACT ISDA and others. The concept is what an accrual, a coupon schedule
/// or a curve needs from any of them: `name()`, `day_count(start, end)` (the days the convention counts) and
/// `year_fraction(start, end)` (an exact `finance::year_fraction`). It is the concept half of a pair; the pure
/// virtual half is `snicholls::day_count_interface` (`interfaces/day_count.hpp`), for a convention chosen at run
/// time from a trade's terms.
///
/// \snippet concepts/day_count_demo.cpp day_count
///
/// **Relationships.** `concepts/named.hpp` (it refines `has_name`), `finance/day_count.hpp` (the conventions),
/// `interfaces/day_count.hpp` (the virtual twin and the adapters), `interfaces/day_count_conformance.hpp` (the
/// kit). Group: \ref core.
///
/// **Implementations.** `finance::act_360`, `act_365_fixed`, `thirty_360`, `thirty_e_360`, `act_act_isda`;
/// `day_count_interface` and its adapter.
///
/// **Guarantees.** A concept: no code, no cost.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/day_count_test.cpp`, `demos/concepts/day_count_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<concepts>`, `<cstdint>`, `concepts/named.hpp`, `finance/year_fraction.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_concepts_day_count_hpp
#define snicholls_core_concepts_day_count_hpp

#include <snicholls/core/concepts/named.hpp>
#include <snicholls/core/finance/year_fraction.hpp>

#include <chrono>
#include <concepts>
#include <cstdint>

namespace snicholls {

    /// \brief A day-count convention: `name()`, `day_count(start, end)` and an exact `year_fraction(start, end)`.
    template <typename T>
    concept day_count_convention = has_name<T> && requires(const T& t, std::chrono::sys_days a, std::chrono::sys_days b) {
        { t.day_count(a, b) } -> std::same_as<std::int64_t>;
        { t.year_fraction(a, b) } -> std::same_as<finance::year_fraction>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_day_count_hpp
