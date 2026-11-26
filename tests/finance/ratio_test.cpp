//
//  ratio_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/ratio.hpp>: normalisation, checked construction, parsing, exact order at the
//  edges of 64 bits, the checked arithmetic, layout, and year_fraction being the same type.
//

#include <snicholls/core/finance/ratio.hpp>
#include <snicholls/core/finance/year_fraction.hpp>

#include <gtest/gtest.h>

#include <compare>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using snicholls::finance::ratio;
using snicholls::finance::ratio_errc;
constexpr std::int64_t max64 = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t min64 = std::numeric_limits<std::int64_t>::min();

static_assert(std::is_same_v<snicholls::finance::year_fraction, ratio>);
static_assert(sizeof(ratio) == 16 && std::is_trivially_copyable_v<ratio>);
static_assert(std::is_nothrow_constructible_v<ratio, std::int64_t, std::int64_t>);
static_assert(ratio{} == ratio{0, -9} && ratio{}.denominator() == 1);
static_assert(ratio{6, -4}.numerator() == -3 && ratio{6, -4}.denominator() == 2);
static_assert(ratio{1, 3} < ratio{1, 2} && ratio{-1, 2} < ratio{} && ratio{3, 2} > ratio{1, 1});
static_assert(-ratio{3, 4} == ratio{-3, 4});
static_assert(*ratio::parse("13.24%") == ratio{331, 2500});
static_assert(*multiply(ratio{3, 2}, ratio{2, 3}) == ratio{1, 1});
static_assert(noexcept(ratio::parse(std::declval<std::string_view>())) && noexcept(multiply(ratio{}, ratio{})) &&
              noexcept(ratio{} <=> ratio{}));

TEST(FinanceRatio, MakeChecksWhatTheConstructorAsserts)
{
    EXPECT_EQ(ratio::make(1, 0).unwrap_err(), ratio_errc::zero_denominator);
    EXPECT_EQ(ratio::make(min64, 1).unwrap_err(), ratio_errc::out_of_range);
    EXPECT_EQ(ratio::make(1, min64).unwrap_err(), ratio_errc::out_of_range);
    EXPECT_EQ(*ratio::make(-max64, -max64), (ratio{1, 1}));
    EXPECT_EQ(*ratio::make(0, -5), ratio{});
}

TEST(FinanceRatio, ParsesIntegersFractionsDecimalsAndPercentages)
{
    EXPECT_EQ(*ratio::parse("7"), (ratio{7, 1}));
    EXPECT_EQ(*ratio::parse("3/2"), (ratio{3, 2}));
    EXPECT_EQ(*ratio::parse("-6/4"), (ratio{-3, 2}));
    EXPECT_EQ(*ratio::parse("1.25"), (ratio{5, 4}));
    EXPECT_EQ(*ratio::parse("-0.001"), (ratio{-1, 1000}));
    EXPECT_EQ(*ratio::parse("4.35%"), (ratio{87, 2000}));
    EXPECT_EQ(*ratio::parse("100%"), (ratio{1, 1}));
    EXPECT_EQ(*ratio::parse("0/9"), ratio{});
}

TEST(FinanceRatio, RefusesTextThatIsNotARatio)
{
    for (const std::string_view bad : {"", "-", "/2", "3/", "1.", ".5", "1.5/2", "3/2%", "1e3", "1,5", " 1", "1 ", "--1", "1%%"})
        EXPECT_EQ(ratio::parse(bad).unwrap_err(), ratio_errc::malformed) << bad;
    EXPECT_EQ(ratio::parse("1/0").unwrap_err(), ratio_errc::zero_denominator);
    EXPECT_EQ(ratio::parse("9223372036854775808").unwrap_err(), ratio_errc::out_of_range);
    EXPECT_EQ(ratio::parse("0.0000000000000000001").unwrap_err(), ratio_errc::out_of_range) << "10^19 does not fit";
    EXPECT_EQ(*ratio::parse("9223372036854775807"), (ratio{max64, 1}));
}

TEST(FinanceRatio, OrdersExactlyWhereCrossMultiplicationWouldOverflow)
{
    const ratio a{max64 - 1, max64}, b{max64 - 2, max64 - 1};   // differ by about 1e-37
    EXPECT_EQ(a <=> b, std::strong_ordering::greater);
    EXPECT_EQ(b <=> a, std::strong_ordering::less);
    EXPECT_EQ(-a <=> -b, std::strong_ordering::less);
    EXPECT_EQ((ratio{max64, 3} <=> ratio{max64, 3}), std::strong_ordering::equal);
    EXPECT_LT((ratio{-max64, 2}), (ratio{-max64 + 1, 2}));
    EXPECT_LT((ratio{1, max64}), (ratio{1, max64 - 1}));
    EXPECT_GT(ratio{}, (ratio{-1, max64}));
    EXPECT_EQ((ratio{7, 2} <=> ratio{7, 2}), std::strong_ordering::equal);
    EXPECT_LT((ratio{7, 2}), (ratio{11, 3}));   // same whole part, the remainders decide
}

TEST(FinanceRatio, ArithmeticIsExactAndRefusesOverflow)
{
    EXPECT_EQ(*add(ratio{1, 6}, ratio{1, 3}), (ratio{1, 2}));
    EXPECT_EQ(*subtract(ratio{1, 6}, ratio{1, 3}), (ratio{-1, 6}));
    EXPECT_EQ(*multiply(ratio{-3, 2}, ratio{4, 9}), (ratio{-2, 3}));
    EXPECT_EQ(*multiply(ratio{max64, 3}, ratio{3, max64}), (ratio{1, 1})) << "cancelled across before multiplying";
    EXPECT_EQ(*multiply(ratio{}, ratio{max64, 1}), ratio{});
    EXPECT_FALSE(multiply(ratio{max64, 1}, ratio{2, 1}).has_value());
    EXPECT_FALSE(add(ratio{max64, 1}, ratio{1, 1}).has_value());
    EXPECT_FALSE(add(ratio{1, max64}, ratio{1, max64 - 1}).has_value());
    EXPECT_EQ(*add(ratio{max64 - 1, 1}, ratio{1, 1}), (ratio{max64, 1}));
    EXPECT_FALSE(subtract(ratio{-max64, 1}, ratio{1, 1}).has_value()) << "INT64_MIN is never a member";
}

TEST(FinanceRatio, InvertsAndConverts)
{
    EXPECT_EQ(*ratio(-2, 3).inverse(), (ratio{-3, 2}));
    EXPECT_FALSE(ratio{}.inverse().has_value());
    EXPECT_DOUBLE_EQ(ratio(1, 8).to_double(), 0.125);
    EXPECT_TRUE(ratio(-1, 8).is_negative());
    EXPECT_TRUE(ratio(0, 8).is_zero());
}

} // namespace
