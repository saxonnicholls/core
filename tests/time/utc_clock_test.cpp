//
//  utc_clock_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/utc_clock.hpp>.
//

#include <snicholls/core/time/utc_clock.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <type_traits>
#include <utility>

namespace {

using namespace snicholls;
using namespace std::chrono_literals;

struct seconds_clock {   // any utc<D> will do
    [[nodiscard]] constexpr utc_s now() const noexcept { return utc_s::from_unix_seconds(7); }
};
struct not_utc {   // a now() that is not a utc
    [[nodiscard]] long now() const noexcept { return 0; }
};
struct mutable_only {   // a clock must read on a const object
    utc_s now() { return {}; }
};

static_assert(utc_clock<manual_utc_clock> && utc_clock<seconds_clock>);
static_assert(!utc_clock<not_utc> && !utc_clock<mutable_only> && !utc_clock<int>);
static_assert(std::is_trivially_copyable_v<manual_utc_clock> && std::is_aggregate_v<manual_utc_clock>);
static_assert(noexcept(manual_utc_clock{}.now()) &&
              noexcept(std::declval<manual_utc_clock&>().advance(std::declval<std::chrono::milliseconds>())));
static_assert(manual_utc_clock{}.now() == utc_ms{}, "it starts at the epoch unless told");
static_assert([] {
    manual_utc_clock c{utc_ms::from_unix_millis(1'000)};
    c.advance(250ms);
    c.advance(-5s);   // ignored: it never runs backwards
    return c.now().unix_millis();
}() == 1'250);
static_assert(detail::clock_now<std::chrono::seconds>(manual_utc_clock{utc_ms::from_unix_millis(-1)}) ==
              utc_s::from_unix_seconds(-1), "floored, before 1970 too");

TEST(TimeUtcClock, AManualClockMovesOnlyWhenTold)
{
    manual_utc_clock c{utc_s::from_unix_seconds(1'800'000'000)};
    EXPECT_EQ(c.now().unix_seconds(), 1'800'000'000);
    c.advance(std::chrono::days{2});
    EXPECT_EQ(c.now().unix_seconds(), 1'800'000'000 + 2 * 86'400);
    c.advance(0ms);
    EXPECT_EQ(c.now().unix_seconds(), 1'800'000'000 + 2 * 86'400);
    c.t = utc_ms::from_unix_millis(5);   // or set outright
    EXPECT_EQ(c.now().unix_millis(), 5);
}

TEST(TimeUtcClock, AnyPrecisionIsReadAtTheReadersPrecision)
{
    EXPECT_EQ(detail::clock_now<std::chrono::seconds>(seconds_clock{}).unix_seconds(), 7);
    EXPECT_EQ(detail::clock_now<std::chrono::milliseconds>(manual_utc_clock{utc_ms::from_unix_millis(1'999)}).unix_millis(), 1'999);
    EXPECT_EQ(detail::clock_now<std::chrono::seconds>(manual_utc_clock{utc_ms::from_unix_millis(1'999)}).unix_seconds(), 1);
}

} // namespace
