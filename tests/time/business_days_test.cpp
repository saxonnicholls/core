//
//  business_days_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/business_days.hpp>.
//

// interfaces/utc/business_days.hpp: O(1) weekday arithmetic, checked against brute-force loops.

#include <chrono>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include <snicholls/core/time/business_days.hpp>

#include "../support/time_fixtures.hpp"

namespace {

using namespace std::chrono;

sys_days brute_add(sys_days d, std::int64_t n)
{
    const int step = n > 0 ? 1 : -1;
    for (std::int64_t left = n > 0 ? n : -n; left > 0;) {
        d += days(step);
        if (snicholls::is_business_day(d)) --left;
    }
    return d;
}

std::int64_t brute_between(sys_days a, sys_days b)
{
    std::int64_t n = 0;
    for (sys_days d = a < b ? a : b; d < (a < b ? b : a); d += days(1)) n += snicholls::is_business_day(d);
    return a <= b ? n : -n;
}

static_assert(snicholls::is_weekend(Saturday) && snicholls::is_weekend(Sunday) && !snicholls::is_weekend(Monday) && !snicholls::is_weekend(Friday));
static_assert(snicholls::add_business_days(sys_days{2024y / May / 3}, 1) == sys_days{2024y / May / 6});   // Fri + 1 = Mon
static_assert(snicholls::add_business_days(sys_days{2024y / May / 3}, 3) == sys_days{2024y / May / 8});   // Fri + 3 = Wed
static_assert(snicholls::add_business_days(sys_days{2024y / May / 4}, 0) == sys_days{2024y / May / 4});   // n == 0: unchanged
static_assert(snicholls::previous_weekday(sys_days{2024y / May / 3}, Friday) == sys_days{2024y / April / 26});
static_assert(snicholls::next_weekday(sys_days{2024y / May / 3}, Monday) == sys_days{2024y / May / 6});
static_assert(noexcept(snicholls::add_business_days(std::declval<sys_days>(), 1)) &&
              noexcept(snicholls::business_days_between(std::declval<sys_days>(), std::declval<sys_days>())));

TEST(UtcBusinessDays, AddMatchesBruteForceIncludingWeekendStarts)
{
    for (sys_days d = sys_days{1969y / December / 1}; d < sys_days{1970y / March / 1}; d += days(1))   // across the epoch
        for (std::int64_t n = -40; n <= 40; ++n)
            ASSERT_EQ(snicholls::add_business_days(d, n), n == 0 ? d : brute_add(d, n)) << d.time_since_epoch().count() << ' ' << n;
    for (sys_days d = sys_days{2024y / January / 1}; d < sys_days{2024y / March / 1}; d += days(1))
        ASSERT_EQ(snicholls::add_business_days(d, 2611), brute_add(d, 2611));   // ten years
    EXPECT_EQ(snicholls::add_business_days(sys_days{1677y / September / 21}, 1), brute_add(sys_days{1677y / September / 21}, 1));
}

TEST(UtcBusinessDays, BetweenMatchesBruteForceAndInvertsAdd)
{
    const sys_days base = sys_days{1969y / December / 20};
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j)
            ASSERT_EQ(snicholls::business_days_between(base + days(i), base + days(j)), brute_between(base + days(i), base + days(j)));
    for (sys_days d = sys_days{2024y / January / 1}; d < sys_days{2024y / February / 1}; d += days(1)) {
        if (!snicholls::is_business_day(d)) continue;
        for (std::int64_t n = -30; n <= 30; ++n) ASSERT_EQ(snicholls::business_days_between(d, snicholls::add_business_days(d, n)), n);
    }
}

TEST(UtcBusinessDays, PreviousAndNextWeekdayAreStrict)
{
    for (sys_days d = sys_days{2024y / April / 1}; d < sys_days{2024y / May / 1}; d += days(1))
        for (unsigned w = 0; w < 7; ++w) {
            const sys_days p = snicholls::previous_weekday(d, weekday(w));
            const sys_days n = snicholls::next_weekday(d, weekday(w));
            ASSERT_EQ(weekday(p), weekday(w));
            ASSERT_EQ(weekday(n), weekday(w));
            ASSERT_TRUE(p < d && d - p <= days(7));
            ASSERT_TRUE(n > d && n - d <= days(7));
        }
}

TEST(UtcBusinessDays, AddMonthsClampsToTheEndOfTheMonth)
{
    static_assert(snicholls::add_months(2024y / January / 31, 1) == 2024y / February / 29);
    static_assert(snicholls::add_months(2023y / January / 31, 1) == 2023y / February / 28);
    static_assert(snicholls::add_months(2024y / March / 31, -1) == 2024y / February / 29);   // the original made 2024-03-02
    static_assert(snicholls::add_months(2024y / May / 31, -3) == 2024y / February / 29);
    static_assert(snicholls::add_months(2024y / February / 29, 12) == 2025y / February / 28);
    static_assert(snicholls::add_months(2024y / December / 15, 1) == 2025y / January / 15);
    static_assert(snicholls::add_months(2024y / January / 15, -13) == 2022y / December / 15);
    for (int y = 1999; y <= 2001; ++y)
        for (unsigned m = 1; m <= 12; ++m)
            for (unsigned d = 1; d <= 31; ++d) {
                const year_month_day from{year(y), month(m), day(d)};
                if (!from.ok()) continue;
                for (int n = -25; n <= 25; ++n) {
                    const year_month_day to = snicholls::add_months(from, n);
                    ASSERT_TRUE(to.ok());
                    const year_month expect = year_month(from.year(), from.month()) + months(n);
                    ASSERT_EQ(year_month(to.year(), to.month()), expect);
                    ASSERT_EQ(to.day(), std::min(from.day(), (expect / last).day()));
                }
            }
}

// Portability fixture (tests/fixtures/time/business_days.tsv, Python stepping one day at a time).
TEST(UtcBusinessDaysFixtures, AddBetweenPreviousAndNextMatchPython)
{
    const auto rows = snicholls::test::read_time_fixture("business_days.tsv");
    for (const auto& row : rows) {
        const auto& f = row.fields;
        ASSERT_EQ(f.size(), 4U) << "line " << row.line;
        const sys_days a{days(snicholls::test::to_i64(f[1]))};
        const std::int64_t b = snicholls::test::to_i64(f[2]);
        const std::int64_t expected = snicholls::test::to_i64(f[3]);
        if (f[0] == "add") {
            EXPECT_EQ(snicholls::add_business_days(a, b).time_since_epoch().count(), expected) << "line " << row.line;
        } else if (f[0] == "between") {
            EXPECT_EQ(snicholls::business_days_between(a, sys_days{days(b)}), expected) << "line " << row.line;
        } else if (f[0] == "prev") {
            EXPECT_EQ(snicholls::previous_weekday(a, weekday(static_cast<unsigned>(b))).time_since_epoch().count(), expected)
                << "line " << row.line;
        } else if (f[0] == "next") {
            EXPECT_EQ(snicholls::next_weekday(a, weekday(static_cast<unsigned>(b))).time_since_epoch().count(), expected)
                << "line " << row.line;
        } else {
            ADD_FAILURE() << "unknown op " << f[0] << " at line " << row.line;
        }
    }
    EXPECT_GT(rows.size(), 1500U);
}

} // namespace
