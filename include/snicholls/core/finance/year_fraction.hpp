//
//  year_fraction.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `finance::year_fraction`: a fraction of a year as an exact ratio of two integers; the name day counts
/// return, for `finance::ratio`.
/// \ingroup core
///
/// **Overview.** A day-count convention's answer is a ratio of whole numbers: 91 / 360 under ACT/360, 92 / 365
/// under ACT/365 Fixed, `(a * 366 + b * 365) / 133590` under ACT/ACT ISDA for `a` days in common years and `b` in
/// leap years. `year_fraction` keeps it exactly, normalised (lowest terms, positive denominator), so two
/// conventions' answers compare exactly and an accrual is `notional * numerator / denominator` with one rounding
/// at the end. `to_double()` is for display and for code that wants a `double` anyway.
///
/// Since 0.27.0 it is an alias of `finance::ratio` (`finance/ratio.hpp`), the one exact rational of `finance/`: a year
/// fraction multiplies a repo rate or a return with `multiply` and compares with a split ratio, with no conversion.
/// Every use that compiled before compiles unchanged: the constructor, `numerator()`, `denominator()`, `to_double()`,
/// `==` and `<=>`.
///
/// \snippet finance/year_fraction_demo.cpp year_fraction
///
/// **Relationships.** `finance/ratio.hpp` (the type), `finance/day_count.hpp` (the conventions that return it),
/// `concepts/day_count.hpp`, `money/rate.hpp` (the same exact-ratio style over `uint256`). Group: \ref core.
///
/// **Implementations.** One: `finance::ratio`.
///
/// **Guarantees.** As `finance::ratio`: constexpr, `noexcept`, trivially copyable, 16 bytes, always normalised, so `==`
/// is member-wise. `<=>` is exact for every pair (before 0.27.0 it cross-multiplied in 64 bits, exact only below
/// 2^63).
///
/// **Errors.** None. The constructor's denominator must not be zero, and neither part `INT64_MIN`: a precondition,
/// `assert`ed (so a compile error in constant evaluation and a failed assertion in a debug build).
///
/// **Performance.** One `std::gcd` on construction. Not benchmarked on its own; see `finance/day_count.hpp`.
///
/// **Tests and demos.** `tests/finance/year_fraction_test.cpp`, `demos/finance/year_fraction_demo.cpp`.
///
/// **Dependencies.** `finance/ratio.hpp`.
///
/// **Known issues.** A sum of two conventions' fractions means nothing, though `add` will compute one: scale an
/// amount by `numerator` and `denominator`, or `multiply` by a rate.
///
/// **Status.** New in 0.23.0; an alias of `finance::ratio` since 0.27.0.

#ifndef snicholls_core_finance_year_fraction_hpp
#define snicholls_core_finance_year_fraction_hpp

#include <snicholls/core/finance/ratio.hpp>

namespace snicholls::finance {

    /// \brief A fraction of a year, `numerator / denominator`, exact and normalised: a `finance::ratio`.
    /// \ingroup core
    using year_fraction = ratio;

} // namespace snicholls::finance

#endif // snicholls_core_finance_year_fraction_hpp
