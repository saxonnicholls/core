//
//  checked_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/checked.hpp>: every boundary of the checked and saturating arithmetic
//  at compile time, exact conversions, and parsing.
//

#include <snicholls/core/checked.hpp>

#include <gtest/gtest.h>

#include <charconv>
#include <cmath>
#include <string>
#include <string_view>

#include <cstdint>
#include <limits>
#include <system_error>

namespace {

using std::int64_t;
constexpr int64_t big = std::numeric_limits<int64_t>::max(), small = std::numeric_limits<int64_t>::min();

static_assert(snicholls::checked_add<int64_t>(big, 1) == std::nullopt && snicholls::checked_add<int64_t>(big - 1, 1) == big);
static_assert(snicholls::checked_sub<int64_t>(small, 1) == std::nullopt && snicholls::checked_sub<int64_t>(0, small) == std::nullopt);
static_assert(snicholls::checked_mul<int64_t>(small, -1) == std::nullopt && snicholls::checked_mul<int64_t>(3037000499, 3037000499).has_value());
static_assert(snicholls::checked_mul<int64_t>(3037000500, 3037000500) == std::nullopt);
static_assert(snicholls::checked_add<std::uint32_t>(4000000000u, 400000000u) == std::nullopt);
static_assert(snicholls::saturating_add<int64_t>(big, 5) == big && snicholls::saturating_sub<int64_t>(small, 5) == small);
static_assert(snicholls::saturating_mul<int64_t>(big, -2) == small && snicholls::saturating_mul<int64_t>(-big, -2) == big);
static_assert(snicholls::numeric_cast<std::uint8_t>(255) == 255 && !snicholls::numeric_cast<std::uint8_t>(256));
static_assert(!snicholls::numeric_cast<std::uint32_t>(-1) && snicholls::numeric_cast<int>(3.0) == 3 && !snicholls::numeric_cast<int>(3.5));
static_assert(!snicholls::numeric_cast<int>(1e10) && snicholls::numeric_cast<float>(0.5) == 0.5f && !snicholls::numeric_cast<float>(0.1));
static_assert(!snicholls::numeric_cast<int64_t>(9.3e18) && snicholls::numeric_cast<double>(int64_t{1} << 53) == 9007199254740992.0);

TEST(ParseNumber, WholeStringsOnly)
{
    EXPECT_EQ(snicholls::parse_number<int>("42").unwrap(), 42);
    EXPECT_EQ(snicholls::parse_number<double>("1.25").unwrap(), 1.25);
    EXPECT_EQ(snicholls::parse_number<int>("42x").unwrap_err(), std::errc::invalid_argument);
    EXPECT_EQ(snicholls::parse_number<int>(" 42").unwrap_err(), std::errc::invalid_argument);
    EXPECT_EQ(snicholls::parse_number<std::int8_t>("300").unwrap_err(), std::errc::result_out_of_range);
    EXPECT_EQ(snicholls::parse_number<int>("").unwrap_err(), std::errc::invalid_argument);
}

// The floating-point fallback (used where std::from_chars has none) accepts exactly from_chars' syntax.
namespace parse = snicholls::detail::parse;
static_assert(parse::float_syntax("1.25") && parse::float_syntax("-1e-300") && parse::float_syntax(".5") && parse::float_syntax("5."));
static_assert(parse::float_syntax("INF") && parse::float_syntax("-infinity") && parse::float_syntax("nan") && parse::float_syntax("NaN(abc_1)"));
static_assert(!parse::float_syntax("") && !parse::float_syntax("+1") && !parse::float_syntax(" 1") && !parse::float_syntax("1 "));
static_assert(!parse::float_syntax(".") && !parse::float_syntax("1e") && !parse::float_syntax("0x1p3") && !parse::float_syntax("infx"));
static_assert(!parse::float_syntax("nan(") && !parse::float_syntax("-") && !parse::float_syntax("1.2.3"));

TEST(CheckedParseFallback, AgreesWithFromCharsWhereTheLibraryHasIt)
{
    for (const std::string_view t : {"0", "1.25", "-0.1", "3.141592653589793", "1e300", "2.2250738585072014e-308", "4.9e-324", "1e-400",
                                     "123456789012345678901234567890", ".5", "5.", "-1E+10", "inf", "-Infinity"}) {
        const auto ours = parse::parse_float<double>(t);
        ASSERT_TRUE(ours.is_ok()) << t;
        if constexpr (parse::has_from_chars<double>) {
            double want = 0;
            std::from_chars(t.data(), t.data() + t.size(), want);
            EXPECT_EQ(ours.unwrap(), want) << t;
        }
    }
    EXPECT_TRUE(std::isnan(parse::parse_float<double>("nan").unwrap()));
    EXPECT_EQ(parse::parse_float<double>("1e400").unwrap_err(), std::errc::result_out_of_range);
    EXPECT_EQ(parse::parse_float<double>("-1e400").unwrap_err(), std::errc::result_out_of_range);
    EXPECT_EQ(parse::parse_float<double>(" 1").unwrap_err(), std::errc::invalid_argument);
    EXPECT_EQ(parse::parse_float<double>("0x10").unwrap_err(), std::errc::invalid_argument);
    EXPECT_EQ(parse::parse_float<float>("1.5").unwrap(), 1.5f);
    EXPECT_EQ(parse::parse_float<float>("1e39").unwrap_err(), std::errc::result_out_of_range);
    EXPECT_EQ(parse::parse_float<long double>("0.5").unwrap(), 0.5L);
    EXPECT_EQ(parse::parse_float<double>(std::string(256, '1')).unwrap_err(), std::errc::invalid_argument);
}

} // namespace
