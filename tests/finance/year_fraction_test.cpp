//
//  year_fraction_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/year_fraction.hpp>: normalisation, order, conversion, layout.
//

#include <snicholls/core/finance/year_fraction.hpp>

#include <gtest/gtest.h>

#include <compare>
#include <cstdint>
#include <type_traits>

namespace {

using snicholls::finance::year_fraction;

static_assert(sizeof(year_fraction) == 16 && alignof(year_fraction) == alignof(std::int64_t));
static_assert(std::is_trivially_copyable_v<year_fraction>);
static_assert(std::is_nothrow_constructible_v<year_fraction, std::int64_t, std::int64_t>);
static_assert(year_fraction{} == year_fraction{0, 7});
static_assert(year_fraction{180, 360} == year_fraction{1, 2});
static_assert(year_fraction{3, -6}.numerator() == -1 && year_fraction{3, -6}.denominator() == 2);
static_assert(year_fraction{0, -5}.denominator() == 1);
static_assert(year_fraction{1, 3} < year_fraction{1, 2});
static_assert(year_fraction{-1, 2} < year_fraction{});
static_assert(year_fraction{91, 360}.to_double() == 91.0 / 360.0);

TEST(FinanceYearFraction, IsReducedToLowestTermsWithAPositiveDenominator)
{
    const year_fraction f{-133590, -267180};
    EXPECT_EQ(f.numerator(), 1);
    EXPECT_EQ(f.denominator(), 2);
    EXPECT_EQ((year_fraction{366 * 10, 133590}), (year_fraction{10, 365}));
}

TEST(FinanceYearFraction, OrdersByValueAcrossDenominators)
{
    EXPECT_EQ(year_fraction(1, 365) <=> year_fraction(1, 366), std::strong_ordering::greater);
    EXPECT_EQ(year_fraction(2, 4) <=> year_fraction(1, 2), std::strong_ordering::equal);
    EXPECT_LT(year_fraction(-3, 360), year_fraction(-1, 360));
    // the largest span of sys_days at the largest denominator still compares exactly
    const std::int64_t big = std::int64_t{1} << 32;
    EXPECT_LT(year_fraction(big * 365, 133590), year_fraction(big * 365 + 1, 133590));
}

TEST(FinanceYearFraction, ConvertsToTheNearestDouble)
{
    EXPECT_DOUBLE_EQ(year_fraction(1, 4).to_double(), 0.25);
    EXPECT_EQ(year_fraction().to_double(), 0.0);
    EXPECT_EQ(year_fraction(-1, 8).to_double(), -0.125);
}

} // namespace
