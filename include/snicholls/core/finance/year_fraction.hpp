//
//  year_fraction.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `finance::year_fraction`: a fraction of a year as an exact ratio of two integers.
/// \ingroup core
///
/// **Overview.** A day-count convention's answer is a ratio of whole numbers: 91 / 360 under ACT/360, 92 / 365
/// under ACT/365 Fixed, `(a * 366 + b * 365) / 133590` under ACT/ACT ISDA for `a` days in common years and `b` in
/// leap years. `year_fraction` keeps it exactly, normalised (lowest terms, positive denominator), so two
/// conventions' answers compare exactly and an accrual is `notional * numerator / denominator` with one rounding
/// at the end. `to_double()` is for display and for code that wants a `double` anyway.
///
/// \snippet finance/year_fraction_demo.cpp year_fraction
///
/// **Relationships.** `finance/day_count.hpp` (the conventions that return it), `concepts/day_count.hpp`,
/// `money/rate.hpp` (the same exact-ratio style over `uint256`). Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** constexpr, `noexcept`, trivially copyable, 16 bytes. Always normalised: equal values have
/// equal members, so `==` is member-wise. `<=>` cross-multiplies in 64 bits, exact while `|numerator| *
/// denominator` of either side stays below 2^63; every day count of `std::chrono::sys_days` (spans below 2^32
/// days, denominators at most 133,590) does.
///
/// **Errors.** None. The constructor's denominator must not be zero: a precondition, `assert`ed (so a compile error
/// in constant evaluation and a failed assertion in a debug build).
///
/// **Performance.** One `std::gcd` on construction. Not benchmarked on its own; see `finance/day_count.hpp`.
///
/// **Tests and demos.** `tests/finance/year_fraction_test.cpp`, `demos/finance/year_fraction_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<compare>`, `<cstdint>`, `<numeric>`.
///
/// **Known issues.** Not an arithmetic type: there is no `+` or `*`, since a sum of two conventions' fractions
/// means nothing; scale by hand from `numerator` and `denominator`.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_finance_year_fraction_hpp
#define snicholls_core_finance_year_fraction_hpp

#include <cassert>
#include <compare>
#include <cstdint>
#include <numeric>

namespace snicholls::finance {

    /// \brief A fraction of a year, `numerator / denominator`, exact and normalised.
    /// \ingroup core
    class year_fraction {
    public:
        /// \brief Zero.
        constexpr year_fraction() noexcept = default;

        /// \brief `numerator / denominator`, reduced to lowest terms with a positive denominator.
        /// \param numerator   The numerator.
        /// \param denominator The denominator, not zero.
        constexpr year_fraction(std::int64_t numerator, std::int64_t denominator) noexcept
        {
            assert(denominator != 0);
            if (denominator < 0) {
                numerator = -numerator;
                denominator = -denominator;
            }
            const std::int64_t g = std::gcd(numerator, denominator);
            numerator_ = g == 0 ? 0 : numerator / g;
            denominator_ = g == 0 ? 1 : denominator / g;
        }

        /// \brief The numerator, in lowest terms; negative when the fraction is.
        /// \return It.
        [[nodiscard]] constexpr std::int64_t numerator() const noexcept { return numerator_; }
        /// \brief The denominator, in lowest terms; always positive.
        /// \return It.
        [[nodiscard]] constexpr std::int64_t denominator() const noexcept { return denominator_; }
        /// \brief The nearest `double`, for display: one rounding, of the quotient.
        /// \return `numerator / denominator`.
        [[nodiscard]] constexpr double to_double() const noexcept
        {
            return static_cast<double>(numerator_) / static_cast<double>(denominator_);
        }

        /// \brief Equality: the same number.
        /// \return Whether the two are equal.
        friend constexpr bool operator==(const year_fraction&, const year_fraction&) noexcept = default;
        /// \brief Order by value.
        /// \param a The left operand.
        /// \param b The right operand.
        /// \return The order of `a` and `b`.
        friend constexpr std::strong_ordering operator<=>(const year_fraction& a, const year_fraction& b) noexcept
        {
            return a.numerator_ * b.denominator_ <=> b.numerator_ * a.denominator_;
        }

    private:
        std::int64_t numerator_ = 0;
        std::int64_t denominator_ = 1;
    };

} // namespace snicholls::finance

#endif // snicholls_core_finance_year_fraction_hpp
