//
//  day_count_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/day_count.hpp>: every convention against QuantLib 1.44 on the fixture
//  tests/fixtures/finance/day_count.txt (scripts/generate_finance_fixtures.py, never core's own output), the ISDA
//  rules at compile time, and the properties the conventions share.
//

#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/interfaces/day_count_conformance.hpp>

#include "../support/fixture_file.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include <type_traits>

namespace {

using namespace std::chrono;
using namespace snicholls::finance;

constexpr sys_days day(int y, unsigned m, unsigned d) { return sys_days{year{y} / month{m} / ::std::chrono::day{d}}; }

// ISDA 2006 4.16 at compile time
static_assert(act_360::year_fraction(day(2024, 1, 1), day(2024, 4, 1)) == year_fraction{91, 360});
static_assert(act_365_fixed::year_fraction(day(2024, 1, 1), day(2025, 1, 1)) == year_fraction{366, 365});
static_assert(thirty_360::day_count(day(2024, 1, 31), day(2024, 2, 29)) == 29);    // D1 31 -> 30
static_assert(thirty_360::day_count(day(2024, 1, 30), day(2024, 3, 31)) == 60);    // D2 31 -> 30 after D1 30
static_assert(thirty_360::day_count(day(2024, 1, 15), day(2024, 3, 31)) == 76);    // D2 31 kept after D1 15
static_assert(thirty_e_360::day_count(day(2024, 1, 15), day(2024, 3, 31)) == 75);  // D2 31 -> 30 always
static_assert(thirty_e_360::day_count(day(2024, 2, 29), day(2024, 3, 31)) == 31);  // no end-of-February rule
static_assert(act_act_isda::year_fraction(day(2023, 12, 31), day(2024, 1, 2)) == year_fraction{366 + 365, 133590});
static_assert(act_act_isda::year_fraction(day(2024, 1, 1), day(2025, 1, 1)) == year_fraction{1, 1});
static_assert(act_act_isda::year_fraction(day(2020, 1, 1), day(2030, 1, 1)) == year_fraction{10, 1});
static_assert(act_act_isda::year_fraction(day(2025, 1, 2), day(2023, 12, 31)) == year_fraction{-(133590 + 366 + 366), 133590});
static_assert(act_360::day_count(day(2024, 3, 1), day(2024, 2, 1)) == -29);

// every convention is an empty, trivially copyable type whose operations do not throw
template <typename C>
constexpr bool plain = std::is_empty_v<C> && std::is_trivially_copyable_v<C> &&
                       noexcept(C::day_count(std::declval<sys_days>(), std::declval<sys_days>())) &&
                       noexcept(C::year_fraction(std::declval<sys_days>(), std::declval<sys_days>()));
static_assert(plain<act_360> && plain<act_365_fixed> && plain<thirty_360> && plain<thirty_e_360> && plain<act_act_isda>);

TEST(FinanceDayCount, NamesAreTheMarketNames)
{
    EXPECT_EQ(act_360::name(), "ACT/360");
    EXPECT_EQ(act_365_fixed::name(), "ACT/365F");
    EXPECT_EQ(thirty_360::name(), "30/360");
    EXPECT_EQ(thirty_e_360::name(), "30E/360");
    EXPECT_EQ(act_act_isda::name(), "ACT/ACT ISDA");
}

template <typename C>
void expect_row(const snicholls::test::fixture_line& row, std::size_t at, const char* name)
{
    const sys_days a = snicholls::test::fixture_date(row.fields[0]), b = snicholls::test::fixture_date(row.fields[1]);
    SCOPED_TRACE(std::string("tests/fixtures/finance/day_count.txt:") + std::to_string(row.line) + " " + name + " " +
                 row.fields[0] + " -> " + row.fields[1]);
    EXPECT_EQ(C::day_count(a, b), std::stoll(row.fields[at]));
    // QuantLib's double (ACT/ACT summed in doubles) against the exact fraction rounded once: a few ulps apart at most
    const double oracle = std::stod(row.fields[at + 1]);
    EXPECT_NEAR(C::year_fraction(a, b).to_double(), oracle, 1e-15 * std::max(1.0, std::abs(oracle)));
}

TEST(FinanceDayCount, AgreesWithQuantLib)
{
    const auto rows = snicholls::test::read_fixture("finance/day_count.txt");
    ASSERT_GT(rows.size(), 500U);
    for (const auto& row : rows) {
        ASSERT_EQ(row.fields.size(), 12U) << "line " << row.line;
        expect_row<act_360>(row, 2, "ACT/360");
        expect_row<act_365_fixed>(row, 4, "ACT/365F");
        expect_row<thirty_360>(row, 6, "30/360");
        expect_row<thirty_e_360>(row, 8, "30E/360");
        expect_row<act_act_isda>(row, 10, "ACT/ACT ISDA");
    }
}

TEST(FinanceDayCount, ActualConventionsAreAntisymmetric)
{
    for (int i = 0; i < 800; i += 7)
        for (int j = 0; j < 800; j += 11) {
            const sys_days a = day(2023, 11, 3) + days(i), b = day(2023, 11, 3) + days(j);
            EXPECT_EQ(act_act_isda::year_fraction(a, b), year_fraction(-act_act_isda::year_fraction(b, a).numerator(),
                                                                       act_act_isda::year_fraction(b, a).denominator()));
            EXPECT_EQ(act_360::day_count(a, b), -act_360::day_count(b, a));
        }
}

TEST(FinanceDayCount, ActActIsdaSumsWholeYearsExactly)
{
    for (int y = 1601; y < 2400; y += 13) {
        const sys_days a = day(y, 1, 1), b = day(y + 7, 1, 1);
        EXPECT_EQ(act_act_isda::year_fraction(a, b), year_fraction(7, 1)) << y;
    }
}

// a basis names a convention: the same fraction as the type, and stable values
static_assert(accrual_fraction(day_count_basis::act_360, day(2024, 1, 1), day(2024, 4, 1)) == year_fraction{91, 360});
static_assert(noexcept(accrual_fraction(day_count_basis::act_360, std::declval<sys_days>(), std::declval<sys_days>())));
static_assert(static_cast<int>(day_count_basis::act_360) == 1 && static_cast<int>(day_count_basis::act_act_isda) == 5);

TEST(FinanceDayCount, ABasisGivesItsConventionsFraction)
{
    const sys_days a = day(2023, 12, 15), b = day(2024, 3, 31);
    EXPECT_EQ(accrual_fraction(day_count_basis::act_360, a, b), act_360::year_fraction(a, b));
    EXPECT_EQ(accrual_fraction(day_count_basis::act_365_fixed, a, b), act_365_fixed::year_fraction(a, b));
    EXPECT_EQ(accrual_fraction(day_count_basis::thirty_360, a, b), thirty_360::year_fraction(a, b));
    EXPECT_EQ(accrual_fraction(day_count_basis::thirty_e_360, a, b), thirty_e_360::year_fraction(a, b));
    EXPECT_EQ(accrual_fraction(day_count_basis::act_act_isda, a, b), act_act_isda::year_fraction(a, b));
    EXPECT_EQ(accrual_fraction(static_cast<day_count_basis>(0), a, b), year_fraction{}) << "not an enumerator";
}

TEST(FinanceDayCount, EveryConventionPassesTheConformanceKit)
{
    // a window over a leap February and a year end
    const sys_days from = day(2023, 12, 20);
    EXPECT_FALSE(snicholls::check_day_count(act_360{}, from, 120));
    EXPECT_FALSE(snicholls::check_day_count(act_365_fixed{}, from, 120));
    EXPECT_FALSE(snicholls::check_day_count(thirty_360{}, from, 120));
    EXPECT_FALSE(snicholls::check_day_count(thirty_e_360{}, from, 120));
    EXPECT_FALSE(snicholls::check_day_count(act_act_isda{}, from, 120));
}

} // namespace
