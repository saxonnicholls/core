//
//  utc_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/utc.hpp>.
//

// <snicholls/core/time/utc.hpp>: snicholls::utc - layout, construction, std::chrono interop, Unix numbers, calendar,
// arithmetic, comparison, hashing.

#include <chrono>
#include <compare>
#include <cstdint>
#include <ctime>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <gtest/gtest.h>

#include <snicholls/core/time/utc.hpp>

#include "../support/time_fixtures.hpp"

namespace {

using namespace std::chrono;
using namespace std::chrono_literals;
using snicholls::utc;

// ---- layout: zero overhead --------------------------------------------------------------------------------

template <typename D>
constexpr bool zero_overhead =
    sizeof(utc<D>) == sizeof(sys_time<D>) && alignof(utc<D>) == alignof(sys_time<D>) &&
    std::is_trivially_copyable_v<utc<D>> && std::is_trivially_destructible_v<utc<D>> &&
    std::is_standard_layout_v<utc<D>> && std::is_nothrow_default_constructible_v<utc<D>>;

static_assert(zero_overhead<nanoseconds> && zero_overhead<microseconds> && zero_overhead<milliseconds>);
static_assert(zero_overhead<seconds> && zero_overhead<minutes> && zero_overhead<days>);
static_assert(sizeof(utc<>) == 8);
static_assert(std::is_same_v<utc<>, snicholls::utc_ns> && std::is_same_v<utc<>::duration, nanoseconds>);
static_assert(std::is_same_v<utc<>::time_point, sys_time<nanoseconds>> && std::is_same_v<utc<>::clock, system_clock>);

// ---- which durations are accepted -------------------------------------------------------------------------

static_assert(snicholls::utc_duration<nanoseconds> && snicholls::utc_duration<duration<std::int64_t, std::ratio<1, 10000000>>>);
static_assert(snicholls::utc_duration<hours> && snicholls::utc_duration<days> && snicholls::utc_duration<duration<int, std::ratio<1, 100>>>);
static_assert(!snicholls::utc_duration<duration<double>>);                         // floating point
static_assert(!snicholls::utc_duration<duration<std::uint64_t, std::nano>>);       // unsigned
static_assert(!snicholls::utc_duration<duration<std::int64_t, std::ratio<1, 3>>>); // not decimal
static_assert(!snicholls::utc_duration<duration<std::int64_t, std::ratio<3, 10>>>);
static_assert(!snicholls::utc_duration<duration<std::int64_t, std::pico>>);        // finer than ns
static_assert(!snicholls::utc_duration<int> && !snicholls::utc_duration<sys_seconds>);
static_assert(utc<nanoseconds>::fraction_digits == 9 && utc<microseconds>::fraction_digits == 6);
static_assert(utc<milliseconds>::fraction_digits == 3 && utc<seconds>::fraction_digits == 0);
static_assert(utc<days>::fraction_digits == 0 && utc<duration<std::int64_t, std::centi>>::fraction_digits == 2);

// ---- construction -----------------------------------------------------------------------------------------

static_assert(utc<>{}.time_since_epoch() == 0ns);   // the epoch, never "now"
static_assert(utc<>::epoch() == utc<>{});
static_assert(std::is_convertible_v<sys_time<nanoseconds>, utc<>>);
static_assert(std::is_convertible_v<sys_seconds, utc<>> && std::is_convertible_v<sys_days, utc<>>);   // lossless
static_assert(!std::is_constructible_v<utc<milliseconds>, sys_time<nanoseconds>>);                     // lossy
static_assert(std::is_convertible_v<utc<milliseconds>, utc<>> && !std::is_constructible_v<utc<milliseconds>, utc<>>);
static_assert(std::is_convertible_v<utc<>, sys_time<nanoseconds>>);
static_assert(!std::is_constructible_v<utc<>, std::int64_t> && !std::is_convertible_v<utc<>, std::int64_t>);
static_assert(utc<>::min().time_since_epoch() == nanoseconds::min() && utc<>::max().time_since_epoch() == nanoseconds::max());

TEST(UtcConstruction, FromStdChronoCalendarExpression)
{
    constexpr utc<> t = sys_days{2024y / April / 30} + 14h + 23min + 5s + 123456789ns;
    static_assert(t.unix_nanos() == 1714486985123456789);
    const sys_time<nanoseconds> back = t;
    EXPECT_EQ(back.time_since_epoch().count(), 1714486985123456789);
    EXPECT_EQ(t.sys_time(), back);
    const utc<> widened = utc<milliseconds>::from_unix_millis(5);
    EXPECT_EQ(widened.unix_nanos(), 5000000);
}

TEST(UtcConstruction, NowIsBetweenTwoSystemClockReadsAndFlooredToDuration)
{
    const auto before = floor<nanoseconds>(system_clock::now());
    const utc<> t = utc<>::now();
    const auto after = system_clock::now();
    EXPECT_LE(before, t.sys_time());
    EXPECT_LE(t.sys_time(), after);
    const utc<seconds> s = utc<seconds>::now();
    EXPECT_EQ(s.sys_time(), floor<seconds>(s.sys_time()));
    static_assert(noexcept(utc<>::now()));
}

// ---- Unix numbers -----------------------------------------------------------------------------------------

static_assert(utc<>::from_unix_millis(1714486985123).unix_millis() == 1714486985123);
static_assert(utc<>::from_unix_seconds(1).unix_nanos() == 1000000000);
static_assert(utc<>::from_unix_micros(-1).unix_nanos() == -1000);
static_assert(utc<>::from_unix_nanos(-1).unix_seconds() == -1);   // floored, not truncated
static_assert(utc<>::from_unix_nanos(-1).unix_millis() == -1 && utc<>::from_unix_nanos(-1).unix_micros() == -1);
static_assert(utc<milliseconds>::from_unix_nanos(-1).unix_millis() == -1);   // floored into the coarser type
static_assert(utc<milliseconds>::from_unix_nanos(1999999).unix_millis() == 1);
static_assert(utc<>::from_unix(hours(1)).unix_seconds() == 3600);
static_assert(noexcept(utc<>::from_unix_millis(0)) && noexcept(utc<>{}.unix_millis()));

TEST(UtcUnix, MassiveMillisecondTimestampsRoundTripAtEveryResolution)
{
    constexpr std::int64_t ms = 1714486985123;
    EXPECT_EQ(utc<>::from_unix_millis(ms).unix_millis(), ms);
    EXPECT_EQ(utc<microseconds>::from_unix_millis(ms).unix_millis(), ms);
    EXPECT_EQ(utc<milliseconds>::from_unix_millis(ms).unix_millis(), ms);
    EXPECT_EQ(utc<seconds>::from_unix_millis(ms).unix_millis(), 1714486985000);
    EXPECT_EQ(utc<>::from_unix_millis(ms).to_unix<microseconds>(), microseconds(ms * 1000));
}

TEST(UtcUnix, FlooringMatchesStdChronoFloorForRandomValues)
{
    std::mt19937_64 rng(42);
    for (int i = 0; i < 100000; ++i) {
        const auto ns = static_cast<std::int64_t>(rng());
        const utc<> t = utc<>::from_unix_nanos(ns);
        ASSERT_EQ(t.unix_seconds(), floor<seconds>(nanoseconds(ns)).count());
        ASSERT_EQ(t.unix_millis(), floor<milliseconds>(nanoseconds(ns)).count());
        ASSERT_EQ(t.unix_micros(), floor<microseconds>(nanoseconds(ns)).count());
        ASSERT_EQ(utc<milliseconds>(utc<milliseconds>::from_unix_nanos(ns)).unix_millis(), t.unix_millis());
    }
}

TEST(UtcUnix, IsRepresentableIsExactAtTheBoundaries)
{
    // utc<> holds +-9223372036854775807 ns: +-9223372036854 ms (floored) fit, one more does not.
    EXPECT_TRUE(utc<>::is_representable(milliseconds(9223372036854)));
    EXPECT_FALSE(utc<>::is_representable(milliseconds(9223372036855)));
    EXPECT_TRUE(utc<>::is_representable(milliseconds(-9223372036854)));
    EXPECT_FALSE(utc<>::is_representable(milliseconds(-9223372036855)));
    EXPECT_TRUE(utc<>::is_representable(nanoseconds::max()) && utc<>::is_representable(nanoseconds::min()));
    EXPECT_TRUE(utc<seconds>::is_representable(nanoseconds::min()));
    // days has a 32-bit rep in libc++ (then 2^63 s is too many days) and a 64-bit rep in libstdc++.
    EXPECT_EQ(utc<days>::is_representable(seconds::max()), sizeof(days::rep) >= sizeof(seconds::rep));
    static_assert(utc<>::is_representable(seconds(9223372036)) && !utc<>::is_representable(seconds(9223372037)));
}

TEST(UtcUnix, TimespecIsFlooredWithNanosecondsInRange)
{
    const std::timespec a = utc<>::from_unix_nanos(-1).to_timespec();
    EXPECT_EQ(a.tv_sec, -1);
    EXPECT_EQ(a.tv_nsec, 999999999);
    const std::timespec b = utc<>::from_unix_nanos(1714486985123456789).to_timespec();
    EXPECT_EQ(b.tv_sec, 1714486985);
    EXPECT_EQ(b.tv_nsec, 123456789);
    EXPECT_EQ(utc<>::from_timespec(a), utc<>::from_unix_nanos(-1));
    EXPECT_EQ(utc<>::from_timespec(b).unix_nanos(), 1714486985123456789);
    EXPECT_EQ(utc<milliseconds>::from_timespec(b).unix_millis(), 1714486985123);
    EXPECT_EQ(utc<seconds>::from_timespec(a).unix_seconds(), -1);
    EXPECT_EQ(utc<milliseconds>::from_unix_millis(-1).to_timespec().tv_nsec, 999000000);
}

// ---- calendar ---------------------------------------------------------------------------------------------

TEST(UtcCalendar, AccessorsAgreeWithStdChrono)
{
    constexpr utc<> t = sys_days{2024y / April / 30} + 14h + 23min + 5s + 123456789ns;
    static_assert(t.date() == 2024y / April / 30);
    static_assert(t.weekday() == Tuesday);
    static_assert(t.days() == sys_days{2024y / April / 30});
    static_assert(t.time_of_day().hours() == 14h && t.time_of_day().minutes() == 23min);
    static_assert(t.time_of_day().seconds() == 5s && t.time_of_day().subseconds() == 123456789ns);

    constexpr utc<> before_epoch = utc<>::from_unix_nanos(-1);   // 1969-12-31T23:59:59.999999999
    static_assert(before_epoch.date() == 1969y / December / 31);
    static_assert(before_epoch.weekday() == Wednesday);
    static_assert(before_epoch.time_of_day().hours() == 23h && before_epoch.time_of_day().subseconds() == 999999999ns);

    std::mt19937_64 rng(7);
    for (int i = 0; i < 100000; ++i) {
        const utc<> r = utc<>::from_unix_nanos(static_cast<std::int64_t>(rng()));
        const auto day = floor<days>(r.sys_time());
        ASSERT_EQ(r.days(), day);
        ASSERT_EQ(r.date(), year_month_day(day));
        ASSERT_EQ(r.weekday(), weekday(day));
        ASSERT_EQ(r.time_of_day().to_duration(), r.sys_time() - day);
    }
}

TEST(UtcCalendar, FloorCeilRoundKeepTheType)
{
    constexpr utc<> t = sys_days{2024y / April / 30} + 14h + 30min;
    static_assert(std::is_same_v<decltype(t.floor<days>()), utc<>>);
    static_assert(t.floor<days>() == utc<>(sys_days{2024y / April / 30}));
    static_assert(t.ceil<days>() == utc<>(sys_days{2024y / May / 1}));
    static_assert(t.round<hours>() == utc<>(sys_days{2024y / April / 30} + 14h));   // a tie rounds to even: 14
    static_assert(t.floor<hours>() == utc<>(sys_days{2024y / April / 30} + 14h));
    constexpr utc<> neg = utc<>::from_unix_nanos(-1);
    static_assert(neg.floor<seconds>().unix_nanos() == -1000000000 && neg.ceil<seconds>().unix_nanos() == 0);
    static_assert(utc<>::from_unix_millis(1500).round<seconds>().unix_seconds() == 2);   // ties to even
    static_assert(utc<>::from_unix_millis(2500).round<seconds>().unix_seconds() == 2);
}

// ---- arithmetic -------------------------------------------------------------------------------------------

TEST(UtcArithmetic, FollowsStdChronoCommonTypes)
{
    constexpr utc<milliseconds> ms = utc<milliseconds>::from_unix_millis(1000);
    static_assert(std::is_same_v<decltype(ms + 1ns), utc<nanoseconds>>);
    static_assert(std::is_same_v<decltype(ms + 1s), utc<milliseconds>>);
    static_assert(std::is_same_v<decltype(1h + ms), utc<milliseconds>>);
    static_assert(std::is_same_v<decltype(ms - 1us), utc<microseconds>>);
    static_assert((ms + 1ns).unix_nanos() == 1000000001 && (1h + ms).unix_millis() == 3601000);
    static_assert((ms - 1us).unix_micros() == 999999);
    static_assert(std::is_same_v<decltype(utc<>{} - ms), nanoseconds> && (utc<>{} - ms) == -1s);
    static_assert(noexcept(ms + std::declval<nanoseconds>()) && noexcept(utc<>{} - ms));

    utc<> t{};
    t += 1500ms;
    t -= 1us;
    EXPECT_EQ(t.unix_nanos(), 1499999000);
    utc<seconds> s{};
    s += 1min;   // coarser converts losslessly
    EXPECT_EQ(s.unix_seconds(), 60);
}

template <typename T, typename D>
concept can_add_assign = requires(T t, D d) { t += d; };

static_assert(can_add_assign<utc<>, milliseconds> && !can_add_assign<utc<seconds>, milliseconds>);   // would lose ms
static_assert(!can_add_assign<utc<>, duration<double>>);

// ---- comparison and hashing -------------------------------------------------------------------------------

TEST(UtcCompare, StrongOrderingWithinAndAcrossResolutions)
{
    constexpr utc<> a = utc<>::from_unix_nanos(1);
    constexpr utc<> b = utc<>::from_unix_nanos(2);
    static_assert(std::is_same_v<decltype(a <=> b), std::strong_ordering>);
    static_assert(a < b && b > a && a != b && a == a && (a <=> b) == std::strong_ordering::less);
    constexpr utc<milliseconds> m = utc<milliseconds>::from_unix_millis(1);
    static_assert(m == utc<>::from_unix_millis(1) && m < utc<>::from_unix_nanos(1000001) && m > a);
    static_assert(std::is_same_v<decltype(m <=> a), std::strong_ordering>);
    static_assert(utc<>::min() < utc<>{} && utc<>{} < utc<>::max());
    // Implicit conversion from sys_time joins the hidden friends.
    EXPECT_TRUE(a == sys_time<nanoseconds>(1ns));
    EXPECT_TRUE(sys_time<nanoseconds>(3ns) > b);
}

TEST(UtcHash, HashesTheTickCount)
{
    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    EXPECT_EQ(std::hash<utc<>>{}(t), std::hash<std::int64_t>{}(1714486985123456789));
    static_assert(noexcept(std::hash<utc<>>{}(t)));
    std::unordered_set<utc<>> set{t, t, t + 1ns};
    EXPECT_EQ(set.size(), 2U);
}

// ---- portability: what the standard libraries do differently, and what utc makes identical ----------------------
//
// libc++ (macOS) ticks std::chrono::system_clock in microseconds, libstdc++ (Linux) in nanoseconds and MSVC in
// 100 ns; utc<> is nanoseconds on all of them, so now() carries the platform's resolution in the low digits and
// nothing else differs. Ticks are signed 64-bit everywhere, time_t is 64-bit on every supported platform, and
// timespec::tv_nsec (a long: 32 bits on Windows) only ever holds [0, 1e9).

static_assert(sizeof(utc<>::rep) == 8 && std::is_signed_v<utc<>::rep>);
static_assert(sizeof(utc<seconds>::rep) == 8 && sizeof(utc<milliseconds>::rep) == 8 && sizeof(utc<microseconds>::rep) == 8);
static_assert(sizeof(std::time_t) == 8, "utc::to_timespec needs a 64-bit time_t (32-bit time_t ends in 2038)");
static_assert(std::numeric_limits<long>::max() >= 999'999'999, "tv_nsec must hold [0, 1e9)");

TEST(UtcPortability, NowCarriesTheSystemClocksResolutionAndNothingElse)
{
    using period = system_clock::period;
    static_assert(period::num == 1 && 1'000'000'000 % period::den == 0, "system_clock ticks a decimal fraction of 1 s");
    constexpr std::int64_t ns_per_tick = 1'000'000'000 / period::den;   // libc++ 1000, libstdc++ 1, MSVC 100
    RecordProperty("system_clock_ns_per_tick", std::to_string(ns_per_tick));
    for (int i = 0; i < 1000; ++i) {
        const utc<> t = utc<>::now();
        ASSERT_EQ(t.unix_nanos() % ns_per_tick, 0);   // the clock's own ticks, never interpolated
        ASSERT_GT(t.unix_seconds(), 1'700'000'000);   // after 2023-11-14: the epoch is 1970 everywhere
    }
    // Coarser resolutions floor what the clock returns, as now() does, on any platform's system_clock::duration:
    const system_clock::time_point one_tick_before_epoch{system_clock::duration(-1)};
    EXPECT_EQ(utc<seconds>(floor<seconds>(one_tick_before_epoch)).unix_seconds(), -1);
    EXPECT_EQ(utc<milliseconds>(floor<milliseconds>(one_tick_before_epoch)).unix_millis(), -1);
    EXPECT_EQ(utc<microseconds>(floor<microseconds>(one_tick_before_epoch)).unix_micros(), -1);
    EXPECT_EQ(utc<>(floor<nanoseconds>(one_tick_before_epoch)).unix_nanos(), -ns_per_tick);
}

TEST(UtcPortability, TimespecAtTheExtremesOfUtcNs)
{
    const std::timespec lo = utc<>::min().to_timespec();
    EXPECT_EQ(static_cast<std::int64_t>(lo.tv_sec), -9223372037);   // floor(INT64_MIN ns / 1e9)
    EXPECT_EQ(static_cast<std::int64_t>(lo.tv_nsec), 145224192);
    EXPECT_EQ(utc<>::from_timespec(lo), utc<>::min());
    const std::timespec hi = utc<>::max().to_timespec();
    EXPECT_EQ(static_cast<std::int64_t>(hi.tv_sec), 9223372036);
    EXPECT_EQ(static_cast<std::int64_t>(hi.tv_nsec), 854775807);
    EXPECT_EQ(utc<>::from_timespec(hi), utc<>::max());
    // Beyond 2038 and before 1901 (the 32-bit time_t limits), at second resolution.
    const auto far = utc<seconds>::from_unix_seconds(253402300799);   // 9999-12-31T23:59:59Z
    EXPECT_EQ(static_cast<std::int64_t>(far.to_timespec().tv_sec), 253402300799);
    EXPECT_EQ(utc<seconds>::from_timespec(utc<seconds>::from_unix_seconds(-62135596800).to_timespec()).unix_seconds(),
              -62135596800);   // 0001-01-01T00:00:00Z
}

// Portability fixture (tests/fixtures/time/timespec.tsv): timespec <-> ticks at every resolution, floored.
TEST(UtcPortability, TimespecMatchesPythonAtEveryResolution)
{
    const auto rows = snicholls::test::read_time_fixture("timespec.tsv");
    for (const auto& row : rows) {
        const auto& f = row.fields;
        ASSERT_EQ(f.size(), 4U) << "line " << row.line;
        std::timespec ts{};
        ts.tv_sec = static_cast<std::time_t>(snicholls::test::to_i64(f[0]));
        ts.tv_nsec = static_cast<long>(snicholls::test::to_i64(f[1]));
        const std::int64_t ticks = snicholls::test::to_i64(f[3]);
        const auto check = [&]<typename D>(std::type_identity<D>) {
            const utc<D> t = utc<D>::from_timespec(ts);
            EXPECT_EQ(t.time_since_epoch().count(), ticks) << "line " << row.line;
            if constexpr (std::is_same_v<D, nanoseconds>) {   // exact at ns: the round trip is the identity
                const std::timespec back = t.to_timespec();
                EXPECT_EQ(static_cast<std::int64_t>(back.tv_sec), static_cast<std::int64_t>(ts.tv_sec)) << "line " << row.line;
                EXPECT_EQ(static_cast<std::int64_t>(back.tv_nsec), static_cast<std::int64_t>(ts.tv_nsec)) << "line " << row.line;
            }
        };
        if (f[2] == "s") check(std::type_identity<seconds>{});
        else if (f[2] == "ms") check(std::type_identity<milliseconds>{});
        else if (f[2] == "us") check(std::type_identity<microseconds>{});
        else if (f[2] == "ns") check(std::type_identity<nanoseconds>{});
        else ADD_FAILURE() << "unknown resolution at line " << row.line;
    }
    EXPECT_GT(rows.size(), 500U);
}

} // namespace
