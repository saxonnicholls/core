//
//  time_zone_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/time_zone_reference.hpp>: the fixed-offset zone, and the two-rule zone's
//  transitions in both hemispheres against the published dates of 2026, at compile time where possible.
//

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
using std::chrono::seconds;
using std::chrono::minutes;
using std::chrono::hours;
using std::chrono::year;
using std::chrono::last;
using std::chrono::January;
using std::chrono::March;
using std::chrono::April;
using std::chrono::June;
using std::chrono::July;
using std::chrono::October;
using std::chrono::November;
using std::chrono::Sunday;

namespace {

constexpr two_rule_zone sydney = sydney_like_zone();
constexpr two_rule_zone ny = new_york_like_zone();

// Sydney 2026: AEDT ends 2026-04-05 03:00 AEDT (16:00Z on the 4th), begins 2026-10-04 02:00 AEST (16:00Z on the 3rd).
constexpr sys_seconds sydney_end = sys_days{year{2026} / April / 4} + hours{16};
constexpr sys_seconds sydney_start = sys_days{year{2026} / October / 3} + hours{16};
static_assert(sydney.info(sydney_end - seconds{1}).offset == hours{11});
static_assert(sydney.info(sydney_end).offset == hours{10});
static_assert(sydney.info(sydney_end).begin == sydney_end && sydney.info(sydney_end).end == sydney_start);
static_assert(sydney.info(sydney_start).offset == hours{11} && sydney.info(sydney_start).is_dst());

// New York 2026: EDT from 2026-03-08 07:00Z to 2026-11-01 06:00Z.
constexpr sys_seconds ny_start = sys_days{year{2026} / March / 8} + hours{7};
constexpr sys_seconds ny_end = sys_days{year{2026} / November / 1} + hours{6};
static_assert(ny.info(ny_start).begin == ny_start && ny.info(ny_start).end == ny_end);
static_assert(ny.info(ny_start - seconds{1}).offset == hours{-5});

TEST(FixedOffsetZone, OneOffsetForever)
{
    constexpr fixed_offset_zone utc;
    static_assert(utc.name() == "UTC" && utc.info(sys_seconds{}).offset == seconds{0});
    const fixed_offset_zone india{"Asia/Kolkata-like", minutes{330}, "IST"};
    EXPECT_EQ(india.info(sys_days{year{1900} / 1 / 1}).offset, minutes{330});
    EXPECT_EQ(india.info(sys_seconds{}).begin, sys_seconds::min());
    EXPECT_EQ(india.info(sys_seconds{}).end, sys_seconds::max());
    EXPECT_EQ(india.info(local_seconds{}).kind, local_time_kind::unique);
    EXPECT_EQ(india.info(sys_seconds{}).abbrev, std::string_view{"IST"});
    EXPECT_FALSE(india.info(sys_seconds{}).is_dst());
}

TEST(TwoRuleZone, AbbreviationsFollowTheRules)
{
    EXPECT_EQ(sydney.info(sys_days{year{2026} / January / 1}).abbrev, std::string_view{"AEDT"});
    EXPECT_EQ(sydney.info(sys_days{year{2026} / July / 1}).abbrev, std::string_view{"AEST"});
    EXPECT_EQ(ny.info(sys_days{year{2026} / July / 1}).abbrev, std::string_view{"EDT"});
    EXPECT_EQ(ny.info(sys_days{year{2026} / July / 1}).save, minutes{60});
    EXPECT_EQ(sydney.name(), "Sydney-like");
}

TEST(TwoRuleZone, TheGapAndTheFoldInTheSouthernHemisphere)
{
    // 2026-10-04 02:30 does not exist in Sydney; 2026-04-05 02:30 happens twice.
    const local_info gap = sydney.info(local_days{year{2026} / October / 4} + hours{2} + minutes{30});
    EXPECT_EQ(gap.kind, local_time_kind::nonexistent);
    EXPECT_EQ(gap.first.offset, hours{10});
    EXPECT_EQ(gap.second.offset, hours{11});
    EXPECT_EQ(gap.second.begin, sydney_start);
    const local_info fold = sydney.info(local_days{year{2026} / April / 5} + hours{2} + minutes{30});
    EXPECT_EQ(fold.kind, local_time_kind::ambiguous);
    EXPECT_EQ(fold.first.offset, hours{11});
    EXPECT_EQ(fold.second.offset, hours{10});
    EXPECT_EQ(sydney.info(local_days{year{2026} / April / 5} + hours{3}).kind, local_time_kind::unique);
    EXPECT_EQ(sydney.info(local_days{year{2026} / October / 4} + hours{3}).kind, local_time_kind::unique);
}

TEST(TwoRuleZone, TheLastWeekdayOfAMonth)
{
    using namespace std::chrono;
    // Europe-like: last Sunday of March 01:00 UTC (02:00 CET) to the last Sunday of October 01:00 UTC (03:00 CEST).
    const two_rule_zone paris{"Paris-like", hours{1}, "CET", minutes{60}, "CEST", {March, Sunday, 5, hours{2}}, {October, Sunday, 5, hours{3}}};
    const zone_info summer = paris.info(sys_days{year{2026} / June / 1});
    EXPECT_EQ(summer.begin, sys_days{year{2026} / March / 29} + hours{1});
    EXPECT_EQ(summer.end, sys_days{year{2026} / October / 25} + hours{1});
}

} // namespace
