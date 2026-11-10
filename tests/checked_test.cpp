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

} // namespace
