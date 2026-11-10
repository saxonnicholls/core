//
//  civil_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/civil.hpp>.
//

// interfaces/utc/civil.hpp: Hinnant's calendar algorithms, checked exhaustively against std::chrono.

#include <chrono>
#include <cstdint>

#include <gtest/gtest.h>

#include <snicholls/core/time/civil.hpp>

#include "../support/time_fixtures.hpp"

namespace {

using namespace std::chrono;

// ---- compile time ---------------------------------------------------------------------------------------

static_assert(snicholls::days_from_civil(1970, 1, 1) == 0);
static_assert(snicholls::days_from_civil(2000, 3, 1) == 11017);
static_assert(snicholls::days_from_civil(2024, 2, 29) == 19782);
static_assert(snicholls::days_from_civil(0, 1, 1) == -719528);
static_assert(snicholls::days_from_civil(-1, 12, 31) == -719529);
static_assert(snicholls::civil_from_days(19782) == snicholls::civil_date{2024, 2, 29});
static_assert(snicholls::civil_from_days(-1) == snicholls::civil_date{1969, 12, 31});
static_assert(snicholls::civil_from_days(-719529) == snicholls::civil_date{-1, 12, 31});
static_assert(snicholls::weekday_from_days(0) == 4);    // 1970-01-01 was a Thursday
static_assert(snicholls::weekday_from_days(-1) == 3);
static_assert(snicholls::weekday_from_days(-4) == 0);   // Sunday
static_assert(snicholls::weekday_from_days(-5) == 6);
static_assert(snicholls::is_leap_year(2000) && snicholls::is_leap_year(2024) && !snicholls::is_leap_year(1900) && !snicholls::is_leap_year(2023));
static_assert(snicholls::is_leap_year(0) && snicholls::is_leap_year(-4) && !snicholls::is_leap_year(-100) && snicholls::is_leap_year(-400));
static_assert(snicholls::last_day_of_month(2024, 2) == 29 && snicholls::last_day_of_month(2023, 2) == 28);
static_assert(snicholls::last_day_of_month(2023, 4) == 30 && snicholls::last_day_of_month(2023, 12) == 31);
static_assert(noexcept(snicholls::days_from_civil(0, 1, 1)) && noexcept(snicholls::civil_from_days(0)));
static_assert(snicholls::civil_date{} == snicholls::civil_date{1970, 1, 1});

// Every day std::chrono::year_month_day can represent: years -32767..32767 (about 24 million days).
TEST(UtcCivilExhaustive, EveryDayStdChronoCanRepresentAgrees)
{
    const auto first = sys_days(year::min() / January / 1).time_since_epoch().count();
    const auto last = sys_days(year::max() / December / 31).time_since_epoch().count();
    std::int64_t failures = 0;
    for (std::int64_t z = first; z <= last; ++z) {
        const year_month_day ref{sys_days(days(z))};
        const snicholls::civil_date c = snicholls::civil_from_days(z);
        failures += c.year != static_cast<int>(ref.year()) || c.month != static_cast<unsigned>(ref.month()) ||
                    c.day != static_cast<unsigned>(ref.day());
        failures += snicholls::days_from_civil(c.year, c.month, c.day) != z;
        failures += snicholls::weekday_from_days(z) != weekday(sys_days(days(z))).c_encoding();
        if (failures != 0) {
            ADD_FAILURE() << "first mismatch at day " << z;
            break;
        }
    }
    EXPECT_EQ(failures, 0);
}

// Every (year, month, day) triple of the RFC 3339 range 0000..9999, validity included.
TEST(UtcCivilExhaustive, EveryValidDateOfYears0000To9999RoundTrips)
{
    std::int64_t expected = snicholls::days_from_civil(0, 1, 1);
    for (std::int64_t y = 0; y <= 9999; ++y)
        for (unsigned m = 1; m <= 12; ++m) {
            ASSERT_EQ(snicholls::last_day_of_month(y, m), static_cast<unsigned>((year(static_cast<int>(y)) / month(m) / last).day()));
            for (unsigned d = 1; d <= snicholls::last_day_of_month(y, m); ++d, ++expected) {
                ASSERT_EQ(snicholls::days_from_civil(y, m, d), expected) << y << '-' << m << '-' << d;
                ASSERT_EQ(snicholls::civil_from_days(expected), (snicholls::civil_date{y, m, d}));
                ASSERT_EQ(snicholls::civil_from_days_fast(expected), (snicholls::civil_date{y, m, d}));   // formatter path
            }
        }
    EXPECT_EQ(expected, snicholls::days_from_civil(10000, 1, 1));
}

TEST(UtcCivilRange, HugeDayCountsRoundTrip)
{
    for (std::int64_t z : {std::int64_t{1} << 40, -(std::int64_t{1} << 40), std::int64_t{1} << 50, -(std::int64_t{1} << 50),
                           (std::int64_t{1} << 59) + 12345, -(std::int64_t{1} << 59) - 12345}) {
        const snicholls::civil_date c = snicholls::civil_from_days(z);
        EXPECT_EQ(snicholls::days_from_civil(c.year, c.month, c.day), z);
        EXPECT_GE(c.month, 1U);
        EXPECT_LE(c.day, snicholls::last_day_of_month(c.year, c.month));
        EXPECT_EQ(snicholls::weekday_from_days(z), snicholls::weekday_from_days(z - 7));
    }
}

TEST(UtcCivilValidation, DaysFromCivilDoesNotValidate)
{
    // Documented: no validation, an overflowing day rolls into the next month.
    EXPECT_EQ(snicholls::days_from_civil(2025, 2, 30), snicholls::days_from_civil(2025, 3, 2));
}

// Portability fixture (tests/fixtures/time/civil.tsv, from Python's datetime.date): the same answers everywhere.
TEST(UtcCivilFixtures, DaysDatesWeekdaysAndMonthLengthsMatchPython)
{
    const auto rows = snicholls::test::read_time_fixture("civil.tsv");
    for (const auto& row : rows) {
        const auto& f = row.fields;
        ASSERT_EQ(f.size(), 6U) << "line " << row.line;
        const std::int64_t z = snicholls::test::to_i64(f[0]);
        const snicholls::civil_date c{snicholls::test::to_i64(f[1]), static_cast<unsigned>(std::stoul(f[2])),
                                      static_cast<unsigned>(std::stoul(f[3]))};
        EXPECT_EQ(snicholls::civil_from_days(z), c) << "line " << row.line;
        EXPECT_EQ(snicholls::civil_from_days_fast(z), c) << "line " << row.line;
        EXPECT_EQ(snicholls::days_from_civil(c.year, c.month, c.day), z) << "line " << row.line;
        EXPECT_EQ(snicholls::weekday_from_days(z), std::stoul(f[4])) << "line " << row.line;
        EXPECT_EQ(snicholls::last_day_of_month(c.year, c.month), std::stoul(f[5])) << "line " << row.line;
    }
    EXPECT_EQ(rows.size(), 400U);
}

} // namespace
