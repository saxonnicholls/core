//
//  holiday_calendar_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/holiday_calendar_conformance.hpp>: the kit passes the reference calendars,
//  through their types and through the interface, and catches each broken rule.
//

#include <snicholls/core/finance/business_day.hpp>
#include <snicholls/core/interfaces/holiday_calendar.hpp>
#include <snicholls/core/interfaces/holiday_calendar_conformance.hpp>

#include <gtest/gtest.h>

#include <chrono>

namespace {

using namespace std::chrono;
using snicholls::check_holiday_calendar;

constexpr sys_days from = sys_days{2026y / January / 1d};

struct flickering {
    mutable int calls = 0;
    bool is_business_day(sys_days) const { return ++calls % 2 == 0; }
};
struct shut_for_a_year {   // closed from 2026-03-01 for 400 days
    static constexpr bool is_business_day(sys_days d) noexcept
    {
        return snicholls::is_business_day(d) && !(d >= sys_days{2026y / March / 1d} && d < sys_days{2026y / March / 1d} + days(400));
    }
};

static_assert(!check_holiday_calendar(snicholls::finance::weekend_calendar{}, from, 40));

TEST(HolidayCalendarConformance, PassesTheReferenceCalendarsByTypeAndByInterface)
{
    EXPECT_FALSE(check_holiday_calendar(snicholls::finance::weekend_calendar{}, from, 366));
    const auto christmas = [](sys_days d) { return d == sys_days{2026y / December / 25d}; };
    EXPECT_FALSE(check_holiday_calendar(snicholls::finance::weekends_and_holidays{christmas}, from, 366));
    const snicholls::holiday_calendar_model<snicholls::finance::weekend_calendar> model;
    const snicholls::holiday_calendar_interface& chosen = model;
    EXPECT_FALSE(check_holiday_calendar(chosen, from, 366));
}

TEST(HolidayCalendarConformance, NamesTheFailedCheck)
{
    EXPECT_EQ(*check_holiday_calendar(flickering{}, from, 10), "holiday_calendar: is_business_day() changes between calls");
    EXPECT_EQ(*check_holiday_calendar(shut_for_a_year{}, from, 366),
              "holiday_calendar: no business day within finance::max_business_day_search days");
}

} // namespace
