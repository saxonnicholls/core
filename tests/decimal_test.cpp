//
//  decimal_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/decimal.hpp>.
//

// Tests for <snicholls/core/decimal.hpp>: exact decimals with one canonical text.

#include <snicholls/core/decimal.hpp>

#include <gtest/gtest.h>

#include <array>
#include <string_view>
#include <type_traits>

namespace {

using snicholls::decimal;
using snicholls::decimal_syntax;

constexpr bool round_trips(std::string_view text)
{
    const auto d = decimal::parse(text);
    return d && d->text().view() == text;
}

// Compile-time round trips of canonical texts.
static_assert(round_trips("0"));
static_assert(round_trips("6930"));
static_assert(round_trips("72.5"));
static_assert(round_trips("-5.25"));
static_assert(round_trips("0.0000001"));
static_assert(round_trips("100.5"));
static_assert(round_trips("120000"));
static_assert(round_trips("999999999999999999"));
static_assert(round_trips("0.000000000000000001"));
static_assert(round_trips("1000000000000000000"));
static_assert(round_trips("-123456789012345678000000000000000000"));

// Non-canonical spellings are rejected in canonical mode.
static_assert(!decimal::parse("72.50"));
static_assert(!decimal::parse("+72.5"));
static_assert(!decimal::parse("-0"));
static_assert(!decimal::parse("007"));
static_assert(!decimal::parse("1e5"));
static_assert(!decimal::parse(".5"));
static_assert(!decimal::parse("5."));
static_assert(!decimal::parse(""));
static_assert(!decimal::parse("-"));
static_assert(!decimal::parse("1,5"));
static_assert(!decimal::parse(" 1"));

static_assert(std::is_trivially_copyable_v<decimal>);
static_assert(sizeof(decimal) == 16);
static_assert(noexcept(decimal::parse(std::string_view{})));   // the function (a const char* argument adds a conversion that is not noexcept)

TEST(DecimalParse, CanonicalTextsRoundTrip)
{
    for (const std::string_view t : {"0", "1", "-1", "6930", "72.5", "0.0000001", "-0.5", "10.01", "1.000001"}) {
        const auto d = decimal::parse(t);
        ASSERT_TRUE(d) << t;
        EXPECT_EQ(d->text().view(), t);
        EXPECT_EQ(d->to_string(), t);
    }
}

TEST(DecimalParse, PlainSyntaxAcceptsTrailingZerosAndPlus)
{
    EXPECT_EQ(decimal::parse("72.50", decimal_syntax::plain), decimal::parse("72.5"));
    EXPECT_EQ(decimal::parse("+6930.0", decimal_syntax::plain), decimal::parse("6930"));
    EXPECT_EQ(decimal::parse("-0.0", decimal_syntax::plain), decimal{});
    EXPECT_EQ(decimal::parse("0.000", decimal_syntax::plain), decimal{});
    EXPECT_FALSE(decimal::parse("1e2", decimal_syntax::plain));
    EXPECT_FALSE(decimal::parse("01.5", decimal_syntax::plain));
}

TEST(DecimalParse, TooManyDigitsOrOutOfRangeIsRejected)
{
    EXPECT_FALSE(decimal::parse("1234567890123456789"));     // 19 significant digits
    EXPECT_FALSE(decimal::parse("0.0000000000000000001"));   // exponent -19
    EXPECT_FALSE(decimal::parse("10000000000000000000"));    // 10^19: exponent 19
    EXPECT_TRUE(decimal::parse("123456789012345678"));
}

TEST(DecimalParts, FromPartsNormalises)
{
    EXPECT_EQ(decimal::from_parts(69300, -1), decimal::parse("6930"));
    EXPECT_EQ(decimal::from_parts(725, -1)->mantissa(), 725);
    EXPECT_EQ(decimal::from_parts(725, -1)->exponent(), -1);
    EXPECT_EQ(decimal::from_parts(6930, 0)->mantissa(), 693);
    EXPECT_EQ(decimal::from_parts(6930, 0)->exponent(), 1);
    EXPECT_EQ(decimal::from_parts(0, 7), decimal{});
    EXPECT_EQ(decimal::from_parts(0, 7)->exponent(), 0);
    EXPECT_FALSE(decimal::from_parts(INT64_MIN, 0));
    EXPECT_FALSE(decimal::from_parts(1'000'000'000'000'000'001, 0));
    EXPECT_FALSE(decimal::from_parts(1, 19));
    EXPECT_FALSE(decimal::from_parts(1, -19));
    EXPECT_EQ(decimal::from_integer(-42)->text().view(), "-42");
}

TEST(DecimalCompare, OrderingIsExact)
{
    const auto d = [](std::string_view t) { return *decimal::parse(t); };
    EXPECT_LT(d("72.5"), d("72.51"));
    EXPECT_LT(d("-5"), d("-4.99"));
    EXPECT_LT(d("-0.1"), d("0"));
    EXPECT_LT(d("0"), d("0.000000000000000001"));
    EXPECT_LT(d("999999999999999999"), d("1000000000000000000"));
    EXPECT_GT(d("100"), d("99.999999999999999"));
    EXPECT_EQ(d("6930") <=> d("6930"), std::strong_ordering::equal);
    EXPECT_GT(d("-0.5"), d("-0.51"));
    static_assert(*decimal::parse("0.1") < *decimal::parse("0.2"));
}

TEST(DecimalAccessors, SignAndDouble)
{
    EXPECT_TRUE(decimal{}.is_zero());
    EXPECT_FALSE(decimal{}.is_negative());
    EXPECT_TRUE(decimal::parse("-2.5")->is_negative());
    EXPECT_DOUBLE_EQ(decimal::parse("72.5")->to_double(), 72.5);
    EXPECT_DOUBLE_EQ(decimal::parse("6930")->to_double(), 6930.0);
}

TEST(DecimalFormat, WritesIntoASpanOrReportsTooSmall)
{
    std::array<char, 8> buf{};
    const auto n = decimal::parse("-72.5")->format(buf);
    ASSERT_TRUE(n);
    EXPECT_EQ(std::string_view(buf.data(), *n), "-72.5");
    std::array<char, 2> small{};
    EXPECT_FALSE(decimal::parse("-72.5")->format(small));
}

} // namespace
