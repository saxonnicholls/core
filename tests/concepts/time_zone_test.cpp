//
//  time_zone_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/time_zone.hpp>: the models and the non-models, from_local's policies, to_local,
//  offset_at and RFC 3339 text with an offset, at compile time where the language allows.
//

#include <snicholls/core/concepts/time_zone.hpp>
#include <snicholls/core/interfaces/time_zone_reference.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <string_view>

using namespace snicholls;
using namespace std::chrono_literals;   // not all of std::chrono: libstdc++'s std::chrono::time_zone and local_info
                                        // would collide with core's
using std::chrono::sys_seconds;
using std::chrono::local_seconds;
using std::chrono::sys_days;
using std::chrono::local_days;
using std::chrono::local_time;
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;
using std::chrono::year;
using std::chrono::March;
using std::chrono::July;
using std::chrono::November;
using std::chrono::December;
using std::chrono::milliseconds;

namespace {

static_assert(time_zone<fixed_offset_zone>);
static_assert(time_zone<two_rule_zone>);
static_assert(time_zone<time_zone_interface>);
static_assert(time_zone<time_zone_adapter<two_rule_zone>>);
static_assert(time_zone<time_zone_adapter<const two_rule_zone&>>);
static_assert(time_zone<time_zone_ref>);

struct no_local_info {
    std::string_view name() const;
    zone_info info(sys_seconds) const;
};
struct name_as_string {
    const char* name() const;
    zone_info info(sys_seconds) const;
    local_info info(local_seconds) const;
};
static_assert(!time_zone<no_local_info>);
static_assert(!time_zone<name_as_string>);
static_assert(!time_zone<int>);

constexpr two_rule_zone ny = new_york_like_zone();
// 2026-03-08 02:30 New York does not exist; 2026-11-01 01:30 happens twice.
constexpr local_seconds gap_time = local_days{year{2026} / March / 8} + hours{2} + minutes{30};
constexpr local_seconds fold_time = local_days{year{2026} / November / 1} + hours{1} + minutes{30};

static_assert(ny.info(gap_time).kind == local_time_kind::nonexistent);
static_assert(ny.info(fold_time).kind == local_time_kind::ambiguous);
static_assert(from_local(ny, gap_time).unwrap_err() == time_zone_errc::nonexistent_local_time);
static_assert(from_local(ny, fold_time).unwrap_err() == time_zone_errc::ambiguous_local_time);

TEST(TimeZoneConcept, FromLocalResolvesAGapOnlyByPolicy)
{
    const auto next = from_local(ny, gap_time, {fold_policy::reject, gap_policy::next_valid});
    EXPECT_EQ(to_text(*next).view(), std::string_view{"2026-03-08T07:00:00Z"});   // 03:00 EDT
    const auto shifted = from_local(ny, gap_time, {fold_policy::reject, gap_policy::shift_forward});
    EXPECT_EQ(to_text(*shifted).view(), std::string_view{"2026-03-08T07:30:00Z"});   // 03:30 EDT
}

TEST(TimeZoneConcept, FromLocalResolvesAFoldOnlyByPolicy)
{
    const auto early = from_local(ny, fold_time, {fold_policy::earliest, gap_policy::reject});
    const auto late = from_local(ny, fold_time, {fold_policy::latest, gap_policy::reject});
    EXPECT_EQ(to_text(*early).view(), std::string_view{"2026-11-01T05:30:00Z"});   // 01:30 EDT
    EXPECT_EQ(to_text(*late).view(), std::string_view{"2026-11-01T06:30:00Z"});    // 01:30 EST
}

TEST(TimeZoneConcept, ToLocalAndOffsetAt)
{
    const utc_ms noon = utc_ms(sys_days{year{2026} / July / 1} + hours{16} + milliseconds{250});
    EXPECT_EQ(offset_at(ny, noon), hours{-4});
    const auto l = to_local(ny, noon);
    static_assert(std::is_same_v<decltype(l), const local_time<milliseconds>>);
    EXPECT_EQ(l.time_since_epoch(), (local_days{year{2026} / July / 1} + hours{12} + milliseconds{250}).time_since_epoch());
    EXPECT_EQ(*from_local(ny, l), noon);
}

TEST(TimeZoneConcept, TextWithAnOffset)
{
    const utc_ms t = utc_ms(sys_days{year{2026} / November / 1} + hours{6} + minutes{30});
    EXPECT_EQ(to_text(t, hours{-5}).view(), std::string_view{"2026-11-01T01:30:00.000-05:00"});
    EXPECT_EQ(to_text(t, minutes{330}, 0).view(), std::string_view{"2026-11-01T12:00:00+05:30"});
    EXPECT_EQ(to_text(t, seconds{0}, 0).view(), std::string_view{"2026-11-01T06:30:00Z"});
    const auto lmt = to_text(t, seconds{-17762});   // New York's local mean time, -04:56:02
    EXPECT_FALSE(lmt);
    EXPECT_EQ(lmt.ec, std::errc::not_supported);
    EXPECT_EQ(to_text(t, hours{24}).ec, std::errc::not_supported);
    EXPECT_EQ(to_text(utc_s(sys_days{year{9999} / December / 31} + hours{23}), hours{2}).ec, std::errc::value_too_large);
}

TEST(TimeZoneConcept, ZoneInfoSaysWhetherDaylightTimeIsInForce)
{
    constexpr zone_info standard{};
    static_assert(!standard.is_dst());
    zone_info d = standard;
    d.save = minutes{60};
    EXPECT_TRUE(d.is_dst());
    EXPECT_NE(d, standard);
}

} // namespace
