//
//  holiday_calendar_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/holiday_calendar.hpp>: the adapter forwards, the interface models the
//  concept, and the adjustment rules accept a calendar chosen at run time.
//

#include <snicholls/core/finance/business_day.hpp>
#include <snicholls/core/interfaces/holiday_calendar.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace {

using namespace std::chrono;
using namespace snicholls;

constexpr sys_days saturday = sys_days{2026y / November / 28d};

static_assert(holiday_calendar_model<finance::weekend_calendar>{}.is_business_day(saturday) == false);
static_assert(holiday_calendar_model<finance::weekend_calendar>{}.is_business_day(saturday + days(2)));
static_assert(noexcept(std::declval<const holiday_calendar_interface&>().is_business_day(saturday)));

TEST(HolidayCalendarInterface, CalendarsChosenByMarketAtRunTime)
{
    const std::set<sys_days> sydney_holidays{sys_days{2026y / December / 25d}, sys_days{2026y / December / 28d}};
    auto sydney = [&](sys_days d) { return sydney_holidays.contains(d); };
    std::map<std::string, std::unique_ptr<holiday_calendar_interface>> by_market;
    by_market["weekends"] = std::make_unique<holiday_calendar_model<finance::weekend_calendar>>();
    by_market["XASX"] = std::make_unique<holiday_calendar_model<finance::weekends_and_holidays<decltype(sydney)>>>(
        finance::weekends_and_holidays{sydney});

    const sys_days christmas = sys_days{2026y / December / 25d};
    const auto following = finance::business_day_convention::following;
    EXPECT_EQ(*finance::adjust(christmas, following, *by_market.at("weekends")), christmas);
    EXPECT_EQ(*finance::adjust(christmas, following, *by_market.at("XASX")), christmas + days(4));   // Tuesday 29th
    EXPECT_FALSE(by_market.at("XASX")->is_business_day(sys_days{2026y / December / 28d}));
}

} // namespace
