//
//  year_fraction_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/year_fraction.hpp>: an exact fraction of a year, normalised, ordered, and an
//  accrual computed from it with one rounding at the end. Exits non-zero if a value is not the one documented.
//

#include <snicholls/core/finance/year_fraction.hpp>

#include <cstdint>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    std::printf("snicholls-core year_fraction demo\n");
    using snicholls::finance::year_fraction;

    //! [year_fraction]
    const year_fraction quarter{91, 360};                       // ACT/360 over 91 days
    const year_fraction half{-180, -360};                       // normalised: 1/2
    static_assert(year_fraction{180, 360} == year_fraction{1, 2});

    // interest on 1,000,000.00 (in cents) at 5%: notional * rate * numerator / denominator, rounded once
    const std::int64_t cents = 100'000'000;
    const std::int64_t interest = cents * 5 * quarter.numerator() / (100 * quarter.denominator());   // 1,263,888 cents
    //! [year_fraction]

    std::printf("  91/360 = %.10f, interest %lld cents\n", quarter.to_double(), static_cast<long long>(interest));
    check(half.numerator() == 1 && half.denominator() == 2, "-180/-360 is stored as 1/2");
    check(quarter < half && half > year_fraction{}, "fractions order by value");
    check(interest == 1'263'888, "the accrual is exact until the one division");
    check(year_fraction{} == year_fraction{0, 9}, "zero has one form");
    return failures == 0 ? 0 : 1;
}
