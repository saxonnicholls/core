//
//  business_day_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/business_day.hpp>: every rule against QuantLib 1.44's Calendar::adjust on the
//  fixture tests/fixtures/finance/business_day.txt (scripts/generate_finance_fixtures.py), over its WeekendsOnly
//  calendar and over TARGET, built here from the holidays the fixture lists; the rules at compile time; the error.
//

#include <snicholls/core/finance/business_day.hpp>
#include <snicholls/core/interfaces/holiday_calendar.hpp>

#include "../support/fixture_file.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono;
using namespace snicholls::finance;

constexpr sys_days day(int y, unsigned m, unsigned d) { return sys_days{year{y} / month{m} / ::std::chrono::day{d}}; }

constexpr sys_days adjusted(sys_days d, business_day_convention c)
{
    return *adjust(d, c, weekend_calendar{});
}

// Saturday 2024-08-31: following leaves August, so modified following goes back to Friday the 30th
static_assert(adjusted(day(2024, 8, 31), business_day_convention::following) == day(2024, 9, 2));
static_assert(adjusted(day(2024, 8, 31), business_day_convention::modified_following) == day(2024, 8, 30));
static_assert(adjusted(day(2024, 8, 31), business_day_convention::preceding) == day(2024, 8, 30));
static_assert(adjusted(day(2024, 8, 31), business_day_convention::unadjusted) == day(2024, 8, 31));
// Sunday 2024-09-01: preceding leaves September, so modified preceding goes forward to Monday the 2nd
static_assert(adjusted(day(2024, 9, 1), business_day_convention::preceding) == day(2024, 8, 30));
static_assert(adjusted(day(2024, 9, 1), business_day_convention::modified_preceding) == day(2024, 9, 2));
// a business day is never moved
static_assert(adjusted(day(2024, 9, 3), business_day_convention::modified_following) == day(2024, 9, 3));
static_assert(noexcept(adjust(std::declval<sys_days>(), business_day_convention::following, weekend_calendar{})));

struct closed {
    static constexpr bool is_business_day(sys_days) noexcept { return false; }
};
static_assert(adjust(sys_days{}, business_day_convention::following, closed{}).unwrap_err_unchecked() ==
              business_day_errc::no_business_day);
static_assert(adjust(sys_days{}, business_day_convention::unadjusted, closed{}).is_ok());

constexpr std::array<business_day_convention, 5> fixture_order{
    business_day_convention::unadjusted, business_day_convention::following, business_day_convention::modified_following,
    business_day_convention::preceding, business_day_convention::modified_preceding};

TEST(FinanceBusinessDay, AgreesWithQuantLib)
{
    const auto rows = snicholls::test::read_fixture("finance/business_day.txt");
    std::set<sys_days> holidays;
    for (const auto& row : rows)
        if (row.fields[0] == "holiday")
            holidays.insert(snicholls::test::fixture_date(row.fields[1]));
    ASSERT_GT(holidays.size(), 20U);
    const weekends_and_holidays target{[&](sys_days d) { return holidays.contains(d); }};
    // the same calendar chosen at run time, through the interface
    const snicholls::holiday_calendar_model<weekends_and_holidays<std::function<bool(sys_days)>>> target_virtual{
        {[&](sys_days d) { return holidays.contains(d); }}};
    const snicholls::holiday_calendar_interface& target_interface = target_virtual;

    std::size_t checked = 0;
    for (const auto& row : rows) {
        if (row.fields[0] == "holiday")
            continue;
        ASSERT_EQ(row.fields.size(), 7U) << "line " << row.line;
        const sys_days d = snicholls::test::fixture_date(row.fields[1]);
        for (std::size_t i = 0; i < fixture_order.size(); ++i) {
            SCOPED_TRACE("tests/fixtures/finance/business_day.txt:" + std::to_string(row.line) + " " + row.fields[0] +
                         " " + row.fields[1] + " rule " + std::to_string(i));
            const sys_days expected = snicholls::test::fixture_date(row.fields[2 + i]);
            if (row.fields[0] == "weekends") {
                EXPECT_EQ(*adjust(d, fixture_order[i], weekend_calendar{}), expected);
            } else {
                EXPECT_EQ(*adjust(d, fixture_order[i], target), expected);
                EXPECT_EQ(*adjust(d, fixture_order[i], target_interface), expected);
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 4000U);
}

TEST(FinanceBusinessDay, WeekendCalendarIsMondayToFriday)
{
    EXPECT_EQ(weekend_calendar::name(), "weekends");
    for (int i = 0; i < 14; ++i) {
        const sys_days d = day(2026, 11, 16) + days(i);   // a Monday
        EXPECT_EQ(weekend_calendar::is_business_day(d), i % 7 < 5) << i;
    }
}

TEST(FinanceBusinessDay, AHolidayPredicateClosesWeekdaysOnly)
{
    const sys_days christmas = day(2026, 12, 25);   // a Friday
    const weekends_and_holidays cal{[=](sys_days d) { return d == christmas; }};
    EXPECT_FALSE(cal.is_business_day(christmas));
    EXPECT_TRUE(cal.is_business_day(christmas - days(1)));
    EXPECT_FALSE(cal.is_business_day(christmas + days(1)));   // Saturday
    EXPECT_EQ(*adjust(christmas, business_day_convention::following, cal), day(2026, 12, 28));
    EXPECT_EQ(*adjust(christmas, business_day_convention::preceding, cal), day(2026, 12, 24));
}

TEST(FinanceBusinessDay, AClosedCalendarFailsInsteadOfLooping)
{
    for (const auto c : fixture_order) {
        const auto r = adjust(day(2026, 1, 1), c, closed{});
        if (c == business_day_convention::unadjusted) {
            EXPECT_TRUE(r.is_ok());
        } else {
            ASSERT_FALSE(r.is_ok());
            EXPECT_EQ(r.unwrap_err_unchecked(), business_day_errc::no_business_day);
        }
    }
    // a run of closed days just inside the limit still adjusts
    const sys_days from = day(2026, 1, 1);
    const auto long_closure = [=](sys_days d) { return d >= from && d < from + days(max_business_day_search - 6); };
    const weekends_and_holidays cal{long_closure};
    EXPECT_TRUE(adjust(from, business_day_convention::following, cal).is_ok());
}

// The operations from crtp's calendar (0.27.0), on dates computed by hand over the weekend calendar.
constexpr std::chrono::sys_days on(int y, unsigned m, unsigned d)
{
    return std::chrono::sys_days{std::chrono::year{y} / std::chrono::month{m} / std::chrono::day{d}};
}

static_assert(*adjust(on(2026, 2, 14), business_day_convention::half_month_modified_following, weekend_calendar{}) == on(2026, 2, 13));
static_assert(*adjust(on(2026, 2, 14), business_day_convention::nearest, weekend_calendar{}) == on(2026, 2, 13));
static_assert(*adjust(on(2026, 2, 15), business_day_convention::nearest, weekend_calendar{}) == on(2026, 2, 16));
static_assert(*end_of_month(on(2026, 1, 10), weekend_calendar{}) == on(2026, 1, 30));
static_assert(is_end_of_month(on(2026, 1, 30), weekend_calendar{}) && !is_end_of_month(on(2026, 1, 29), weekend_calendar{}));
static_assert(*advance_business_days(on(2026, 1, 30), 1, weekend_calendar{}) == on(2026, 2, 2));
static_assert(*advance_business_days(on(2026, 2, 2), -1, weekend_calendar{}) == on(2026, 1, 30));
static_assert(*advance_business_days(on(2026, 2, 1), 0, weekend_calendar{}) == on(2026, 2, 1));
static_assert(count_business_days(on(2026, 2, 2), on(2026, 2, 9), weekend_calendar{}) == 5);
static_assert(count_business_days(on(2026, 2, 2), on(2026, 2, 9), weekend_calendar{}, true, true) == 6);
static_assert(count_business_days(on(2026, 2, 9), on(2026, 2, 2), weekend_calendar{}) == -5);
static_assert(count_business_days(on(2026, 2, 2), on(2026, 2, 2), weekend_calendar{}, true, true) == 1);

TEST(FinanceBusinessDay, HalfMonthModifiedFollowingStaysInItsHalf)
{
    // 2026-08-15 is a Saturday: following is Monday the 17th, in the second half, so preceding: Friday the 14th.
    EXPECT_EQ(*adjust(on(2026, 8, 15), business_day_convention::half_month_modified_following, weekend_calendar{}), on(2026, 8, 14));
    // 2026-05-30 is a Saturday at the month's end: like modified following, back to Friday the 29th.
    EXPECT_EQ(*adjust(on(2026, 5, 30), business_day_convention::half_month_modified_following, weekend_calendar{}), on(2026, 5, 29));
    // 2026-08-08 is a Saturday in the first half: following, Monday the 10th.
    EXPECT_EQ(*adjust(on(2026, 8, 8), business_day_convention::half_month_modified_following, weekend_calendar{}), on(2026, 8, 10));
}

TEST(FinanceBusinessDay, AdvanceMonthsWithAndWithoutTheEndOfMonthRule)
{
    EXPECT_EQ(*advance_months(on(2026, 1, 30), 1, business_day_convention::following, true, weekend_calendar{}), on(2026, 2, 27));
    EXPECT_EQ(*advance_months(on(2026, 1, 30), 1, business_day_convention::following, false, weekend_calendar{}), on(2026, 3, 2));
    EXPECT_EQ(*advance_months(on(2026, 3, 31), -1, business_day_convention::modified_following, false, weekend_calendar{}), on(2026, 2, 27));
    EXPECT_EQ(*advance_months(on(2026, 1, 15), 12, business_day_convention::unadjusted, true, weekend_calendar{}), on(2027, 1, 15));
}

TEST(FinanceBusinessDay, AdjustedAndJointCalendars)
{
    const adjusted_calendar holiday{weekend_calendar{}, {on(2026, 2, 4)}, {on(2026, 2, 7)}};   // close a Wednesday, open a Saturday
    EXPECT_FALSE(holiday.is_business_day(on(2026, 2, 4)));
    EXPECT_TRUE(holiday.is_business_day(on(2026, 2, 7)));
    EXPECT_TRUE(holiday.is_business_day(on(2026, 2, 5)));
    EXPECT_EQ(holiday.base().name(), "weekends");
    EXPECT_EQ(holidays_between(on(2026, 2, 2), on(2026, 2, 8), holiday), (std::vector<std::chrono::sys_days>{on(2026, 2, 4)}));
    EXPECT_EQ(holidays_between(on(2026, 2, 2), on(2026, 2, 8), holiday, true),
              (std::vector<std::chrono::sys_days>{on(2026, 2, 4), on(2026, 2, 8)}));
    const joint_calendar both{weekend_calendar{}, holiday};
    const joint_calendar either{weekend_calendar{}, holiday, joint_rule::join_business_days};
    EXPECT_FALSE(both.is_business_day(on(2026, 2, 4)));
    EXPECT_TRUE(either.is_business_day(on(2026, 2, 4)));
    EXPECT_FALSE(both.is_business_day(on(2026, 2, 7)));
    EXPECT_TRUE(either.is_business_day(on(2026, 2, 7)));
    static_assert(snicholls::holiday_calendar<decltype(both)>);
    EXPECT_EQ(count_business_days(on(2026, 2, 2), on(2026, 2, 9), both), 4);
}

TEST(FinanceBusinessDay, NearestAndEndOfMonthFailOnAClosedYear)
{
    const weekends_and_holidays closed{[](std::chrono::sys_days) { return true; }};
    EXPECT_EQ(adjust(on(2026, 2, 4), business_day_convention::nearest, closed).unwrap_err(), business_day_errc::no_business_day);
    EXPECT_TRUE(end_of_month(on(2026, 2, 4), closed).is_err());
    EXPECT_FALSE(is_end_of_month(on(2026, 2, 4), closed));
    EXPECT_TRUE(advance_business_days(on(2026, 2, 4), 1, closed).is_err());
}

} // namespace
