//
//  enums_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/enums.hpp>: names, values, parsing, gaps, negatives, every
//  underlying type, a widened range, and what cannot be reflected.
//

#include <snicholls/core/meta/enums.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string_view>

namespace trading {
enum class side : std::uint8_t { buy, sell };
enum class tenor : int { overnight = -1, week = 7, month = 30, quarter = 91 };
enum venue : unsigned short { lse = 1, nyse = 2, cme = 900 };
enum unfixed { first, second };   // no fixed underlying type
} // namespace trading

template <>
struct snicholls::enum_range<trading::venue> {
    static constexpr long long min = 0;
    static constexpr long long max = 1000;
};

namespace {

using namespace trading;
using snicholls::enum_cast;
using snicholls::enum_name;

static_assert(snicholls::reflectable_enum<side> && snicholls::reflectable_enum<venue>);
static_assert(!snicholls::reflectable_enum<unfixed> && !snicholls::reflectable_enum<int>);

static_assert(enum_name(side::buy) == "buy" && enum_name(side::sell) == "sell");
static_assert(enum_name<side::sell>() == "sell");
static_assert(snicholls::enum_count_v<side> == 2);
static_assert(snicholls::enum_values_v<side> == std::array{side::buy, side::sell});

// Gaps and a negative value, in order of value
static_assert(snicholls::enum_values_v<tenor> == std::array{tenor::overnight, tenor::week, tenor::month, tenor::quarter});
static_assert(snicholls::enum_names_v<tenor>[0] == "overnight");
static_assert(enum_name(static_cast<tenor>(8)).empty());
static_assert(!snicholls::enum_contains(static_cast<tenor>(8)) && snicholls::enum_contains(tenor::month));
static_assert(snicholls::enum_index(tenor::month) == 2u && !snicholls::enum_index(static_cast<tenor>(3)));

// Parsing, by name or by value
static_assert(enum_cast<tenor>("quarter") == tenor::quarter && !enum_cast<tenor>("year") && !enum_cast<tenor>("Quarter"));
static_assert(enum_cast<tenor>(30) == tenor::month && !enum_cast<tenor>(31));

// An unscoped enumeration with a fixed type, and a widened range
static_assert(enum_name(cme) == "cme" && snicholls::enum_count_v<venue> == 3);
static_assert(enum_cast<venue>("nyse") == nyse);

// Every name round-trips
consteval bool round_trips()
{
    for (const tenor t : snicholls::enum_values_v<tenor>)
        if (enum_cast<tenor>(enum_name(t)) != t)
            return false;
    return true;
}
static_assert(round_trips());
// An enumeration in an anonymous namespace, whose qualified spelling starts with a parenthesis on Clang
enum class hidden : int { only = 4 };
static_assert(enum_name(hidden::only) == "only" && enum_name(static_cast<hidden>(5)).empty() && snicholls::enum_count_v<hidden> == 1);

static_assert(noexcept(enum_name(side::buy)) && noexcept(enum_cast<side>(std::string_view{})));

TEST(EnumReflection, NamesAreViewsIntoStaticStorage)
{
    const std::string_view a = enum_name(side::buy);
    const std::string_view b = enum_name(side::buy);
    EXPECT_EQ(a.data(), b.data());
    EXPECT_EQ(enum_name(static_cast<side>(200)), "");
}

} // namespace
