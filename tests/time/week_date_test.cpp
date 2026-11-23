//
//  week_date_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/week_date.hpp>: both directions against Python's datetime.isocalendar() and
//  fromisocalendar() on the fixture tests/fixtures/time/week_date.txt (scripts/generate_week_date_fixtures.py),
//  every day from 1600 to 2400 round-tripped, and the rules at compile time.
//

#include <snicholls/core/time/week_date.hpp>

#include "../support/fixture_file.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <utility>

namespace {

using namespace std::chrono;
using snicholls::iso_week_date;
using snicholls::week_date_errc;

static_assert(snicholls::to_iso_week_date(sys_days{2024y / December / 30d}) == iso_week_date{2025, 1, 1});
static_assert(snicholls::to_iso_week_date(sys_days{2021y / January / 3d}) == iso_week_date{2020, 53, 7});
static_assert(snicholls::to_iso_week_date(sys_days{2026y / November / 23d}) == iso_week_date{2026, 48, 1});
static_assert(snicholls::to_iso_week_date(sys_days{}) == iso_week_date{});   // 1970-01-01, a Thursday
static_assert(snicholls::to_sys_days(iso_week_date{2020, 53, 7}) == sys_days{2021y / January / 3d});
static_assert(snicholls::iso_weeks_in_year(2020) == 53 && snicholls::iso_weeks_in_year(2015) == 53);   // leap + Wednesday, Thursday
static_assert(snicholls::iso_weeks_in_year(2021) == 52 && snicholls::iso_weeks_in_year(2026) == 53);
static_assert(snicholls::make_iso_week_date(2021, 53, 1).unwrap_err_unchecked() == week_date_errc::invalid_week);
static_assert(snicholls::make_iso_week_date(2021, 0, 1).unwrap_err_unchecked() == week_date_errc::invalid_week);
static_assert(snicholls::make_iso_week_date(2021, 1, 8).unwrap_err_unchecked() == week_date_errc::invalid_weekday);
static_assert(snicholls::make_iso_week_date(2021, 1, 0).unwrap_err_unchecked() == week_date_errc::invalid_weekday);
static_assert(*snicholls::make_iso_week_date(2026, 53, 7) == iso_week_date{2026, 53, 7});
static_assert(iso_week_date{2020, 53, 7} < iso_week_date{2021, 1, 1});
static_assert(noexcept(snicholls::to_iso_week_date(std::declval<sys_days>())) && noexcept(snicholls::to_sys_days(iso_week_date{})));

TEST(TimeWeekDate, AgreesWithPythonsIsocalendar)
{
    const auto rows = snicholls::test::read_fixture("time/week_date.txt");
    std::size_t days_checked = 0, invalid_checked = 0;
    for (const auto& row : rows) {
        SCOPED_TRACE("tests/fixtures/time/week_date.txt:" + std::to_string(row.line));
        if (row.fields[0] == "day") {
            ASSERT_EQ(row.fields.size(), 5U);
            const sys_days d = snicholls::test::fixture_date(row.fields[1]);
            const iso_week_date expected{std::stoi(row.fields[2]), static_cast<unsigned>(std::stoi(row.fields[3])),
                                         static_cast<unsigned>(std::stoi(row.fields[4]))};
            EXPECT_EQ(snicholls::to_iso_week_date(d), expected);
            const auto made = snicholls::make_iso_week_date(expected.year, expected.week, expected.weekday);
            ASSERT_TRUE(made.is_ok());
            EXPECT_EQ(snicholls::to_sys_days(*made), d);
            ++days_checked;
        } else {
            ASSERT_EQ(row.fields[0], "invalid");
            EXPECT_FALSE(snicholls::make_iso_week_date(std::stoi(row.fields[1]), static_cast<unsigned>(std::stoi(row.fields[2])),
                                                       static_cast<unsigned>(std::stoi(row.fields[3])))
                             .is_ok());
            ++invalid_checked;
        }
    }
    EXPECT_GT(days_checked, 1000U);
    EXPECT_GT(invalid_checked, 100U);
}

TEST(TimeWeekDate, EveryDayFrom1600To2400RoundTripsInOrder)
{
    const sys_days first = sys_days{1600y / January / 1d}, last = sys_days{2400y / December / 31d};
    iso_week_date previous = snicholls::to_iso_week_date(first - days(1));
    for (sys_days d = first; d <= last; d += days(1)) {
        const iso_week_date w = snicholls::to_iso_week_date(d);
        if (!(w > previous) || snicholls::to_sys_days(w) != d || w.week > snicholls::iso_weeks_in_year(w.year) ||
            w.weekday != weekday(d).iso_encoding()) {
            FAIL() << "day " << d.time_since_epoch().count();
        }
        previous = w;
    }
}

TEST(TimeWeekDate, AFourHundredYearCycleHas71LongYears)
{
    int long_years = 0;
    for (int y = 2000; y < 2400; ++y)
        long_years += snicholls::iso_weeks_in_year(y) == 53;
    EXPECT_EQ(long_years, 71);
}

} // namespace
