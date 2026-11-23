//
//  holiday_calendar_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/holiday_calendar.hpp>: what models holiday_calendar and what does not.
//

#include <snicholls/core/concepts/holiday_calendar.hpp>
#include <snicholls/core/finance/business_day.hpp>
#include <snicholls/core/interfaces/holiday_calendar.hpp>

#include <gtest/gtest.h>

#include <chrono>

namespace {

using std::chrono::sys_days;
using snicholls::holiday_calendar;

struct closed_on_first {   // a calendar of the caller's own
    bool is_business_day(sys_days d) const { return std::chrono::year_month_day(d).day() != std::chrono::day{1}; }
};
struct mutable_only {
    bool is_business_day(sys_days) { return true; }   // not callable on a const calendar
};
struct wrong_argument {
    bool is_business_day(int) const { return true; }
};
struct no_answer {
    void is_business_day(sys_days) const {}
};

constexpr auto predicate = [](sys_days) noexcept { return false; };

static_assert(holiday_calendar<snicholls::finance::weekend_calendar>);
static_assert(holiday_calendar<snicholls::finance::weekends_and_holidays<decltype(predicate)>>);
static_assert(holiday_calendar<snicholls::holiday_calendar_interface>);
static_assert(holiday_calendar<closed_on_first>);
static_assert(!holiday_calendar<mutable_only>);
static_assert(!holiday_calendar<wrong_argument>);
static_assert(!holiday_calendar<no_answer>);
static_assert(!holiday_calendar<decltype(predicate)>);   // a bare predicate is wrapped in weekends_and_holidays

TEST(HolidayCalendarConcept, ACalendarOfTheCallersOwnWorksWithTheRules)
{
    using namespace std::chrono;
    const sys_days first = sys_days{2026y / March / 1d};   // a Sunday, and the 1st
    const auto r = snicholls::finance::adjust(first, snicholls::finance::business_day_convention::following, closed_on_first{});
    ASSERT_TRUE(r.is_ok());
    EXPECT_EQ(*r, first + days(1));
}

} // namespace
