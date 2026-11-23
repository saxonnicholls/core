//
//  day_count_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/day_count_conformance.hpp>: the kit passes every convention, through its type
//  and through the interface, and catches each broken rule.
//

#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/interfaces/day_count.hpp>
#include <snicholls/core/interfaces/day_count_conformance.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string_view>

namespace {

using namespace std::chrono;
using snicholls::check_day_count;
using fraction = snicholls::finance::year_fraction;

constexpr sys_days from = sys_days{2024y / February / 20d};

struct unnamed {
    static constexpr std::string_view name() noexcept { return ""; }
    static constexpr std::int64_t day_count(sys_days a, sys_days b) noexcept { return (b - a).count(); }
    static constexpr fraction year_fraction(sys_days a, sys_days b) noexcept { return {day_count(a, b), 365}; }
};
struct nonzero {
    static constexpr std::string_view name() noexcept { return "nonzero"; }
    static constexpr std::int64_t day_count(sys_days a, sys_days b) noexcept { return (b - a).count() + 1; }
    static constexpr fraction year_fraction(sys_days a, sys_days b) noexcept { return {day_count(a, b), 365}; }
};
struct falling {
    static constexpr std::string_view name() noexcept { return "falling"; }
    static constexpr std::int64_t day_count(sys_days a, sys_days b) noexcept { return (b - a).count() % 5; }
    static constexpr fraction year_fraction(sys_days a, sys_days b) noexcept { return {day_count(a, b), 365}; }
};
struct per_month {   // a "year" of 12 days
    static constexpr std::string_view name() noexcept { return "per month"; }
    static constexpr std::int64_t day_count(sys_days a, sys_days b) noexcept { return (b - a).count(); }
    static constexpr fraction year_fraction(sys_days a, sys_days b) noexcept { return {day_count(a, b), 12}; }
};

static_assert(!check_day_count(snicholls::finance::act_act_isda{}, from, 20));
static_assert(check_day_count(nonzero{}, from, 5).has_value());

TEST(DayCountConformance, PassesEveryConventionByTypeAndByInterface)
{
    EXPECT_FALSE(check_day_count(snicholls::finance::thirty_360{}, from, 60));
    const snicholls::day_count_model<snicholls::finance::thirty_e_360> model;
    const snicholls::day_count_interface& chosen = model;
    EXPECT_FALSE(check_day_count(chosen, from, 60));
}

TEST(DayCountConformance, NamesTheFailedCheck)
{
    EXPECT_EQ(*check_day_count(unnamed{}, from, 10), "day_count: name() is empty");
    EXPECT_EQ(*check_day_count(nonzero{}, from, 10), "day_count: a period of no days is not zero");
    EXPECT_EQ(*check_day_count(falling{}, from, 10), "day_count: the count or the fraction falls as the end moves later");
    EXPECT_EQ(*check_day_count(per_month{}, from, 10), "day_count: the fraction is not a year's worth of days per counted day");
}

} // namespace
