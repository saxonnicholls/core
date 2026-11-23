//
//  day_count_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/day_count.hpp>: what models day_count_convention and what does not.
//

#include <snicholls/core/concepts/day_count.hpp>
#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/interfaces/day_count.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string_view>

namespace {

using std::chrono::sys_days;
using snicholls::day_count_convention;
using fraction = snicholls::finance::year_fraction;

// a convention of the caller's own: 1/252 per calendar day, for the sake of the test
struct per_252 {
    std::string_view name() const noexcept { return "ACT/252"; }
    std::int64_t day_count(sys_days a, sys_days b) const noexcept { return (b - a).count(); }
    fraction year_fraction(sys_days a, sys_days b) const noexcept { return {day_count(a, b), 252}; }
};
struct no_name {
    std::int64_t day_count(sys_days, sys_days) const noexcept { return 0; }
    fraction year_fraction(sys_days, sys_days) const noexcept { return {}; }
};
struct double_fraction {
    std::string_view name() const noexcept { return "double"; }
    std::int64_t day_count(sys_days, sys_days) const noexcept { return 0; }
    double year_fraction(sys_days, sys_days) const noexcept { return 0; }
};
struct int_count {
    std::string_view name() const noexcept { return "int"; }
    int day_count(sys_days, sys_days) const noexcept { return 0; }
    fraction year_fraction(sys_days, sys_days) const noexcept { return {}; }
};

static_assert(day_count_convention<snicholls::finance::act_360>);
static_assert(day_count_convention<snicholls::finance::act_act_isda>);
static_assert(day_count_convention<snicholls::day_count_interface>);
static_assert(day_count_convention<per_252>);
static_assert(!day_count_convention<no_name>);
static_assert(!day_count_convention<double_fraction>);   // the fraction is exact, never a double
static_assert(!day_count_convention<int_count>);
static_assert(!day_count_convention<int>);

TEST(DayCountConcept, AConventionOfTheCallersOwnModelsIt)
{
    const per_252 c;
    const sys_days a{std::chrono::days{0}};
    EXPECT_EQ(c.year_fraction(a, a + std::chrono::days{126}), fraction(1, 2));
    EXPECT_TRUE(snicholls::has_name<per_252>);
}

} // namespace
