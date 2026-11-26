//
//  returns_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/returns.hpp>: every measure against the fixture tests/fixtures/finance/returns.txt
//  (scripts/generate_returns_fixtures.py: numpy-financial 1.0.0's irr and numpy.roots, Python's decimal and fractions;
//  never core's own output), the exact measures at compile time, and every error.
//

#include <snicholls/core/finance/returns.hpp>
#include <snicholls/core/money/asset.hpp>

#include "../support/fixture_file.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace std::chrono;
using namespace snicholls::finance;
using snicholls::money;

constexpr snicholls::asset aud = snicholls::checked_asset("AUD", 2), usd = snicholls::checked_asset("USD", 2);
constexpr money cents(std::int64_t v) { return money::from_units(aud, static_cast<std::uint64_t>(v < 0 ? -v : v), v < 0); }
constexpr sys_days on(int y, unsigned m, unsigned d) { return sys_days{year{y} / month{m} / day{d}}; }

// the exact measures at compile time
static_assert(*period_return({cents(100000), cents(0), cents(105000)}) == ratio{1, 20});
static_assert(*period_return({cents(100000), cents(50000), cents(150000)}) == ratio{});
static_assert([] {
    const valuation months[] = {{cents(100000), cents(0), cents(80000)}, {cents(80000), cents(0), cents(100000)}};
    return *time_weighted_return(months) == ratio{};   // -20% then +25%
}());
static_assert([] {
    const money_flow in[] = {{on(2026, 2, 15), cents(500000)}};
    return *modified_dietz(cents(1000000), cents(1550000), in, on(2026, 1, 1), on(2026, 3, 31)) == ratio{89, 2220};
}());
static_assert([] {
    const double flows[] = {-100, 50, 50};
    return *irr(flows) == 0.0;
}());
static_assert(noexcept(irr(std::declval<std::span<const double>>())) && noexcept(period_return(std::declval<valuation>())));
static_assert(time_weighted_return(std::span<const double>{}) == 0.0);

std::vector<double> numbers(const std::vector<std::string>& f, std::size_t from, std::size_t n)
{
    std::vector<double> out;
    for (std::size_t i = 0; i < n; ++i)
        out.push_back(std::stod(f.at(from + i)));
    return out;
}

money minor(const std::string& s) { return cents(std::stoll(s)); }

std::vector<money_flow> money_flows(const std::vector<std::string>& f, std::size_t from)
{
    std::vector<money_flow> out;
    const auto n = std::stoul(f.at(from));
    for (std::size_t i = 0; i < n; ++i)
        out.push_back({snicholls::test::fixture_date(f.at(from + 1 + 2 * i)), minor(f.at(from + 2 + 2 * i))});
    return out;
}

void expect_near(double actual, double expected, const std::string& where)
{
    EXPECT_NEAR(actual, expected, 1e-9 * std::max(1.0, std::fabs(expected))) << where;
}

TEST(FinanceReturns, AgreesWithTheOracles)
{
    const auto rows = snicholls::test::read_fixture("finance/returns.txt");
    std::size_t seen[5]{};
    for (const auto& row : rows) {
        const auto& f = row.fields;
        const std::string where = "returns.txt:" + std::to_string(row.line) + " " + f.at(0) + " " + f.at(1);
        if (f[0] == "irr") {
            ++seen[0];
            const std::size_t k = std::stoul(f.at(3));
            const auto expected = numbers(f, 4, k);
            const auto flows = numbers(f, 5 + k, std::stoul(f.at(4 + k)));
            const auto r = irr(flows);
            double found[4]{};
            const std::size_t n = irr_roots(flows, found);
            if (f[2] == "one") {
                ASSERT_TRUE(r.is_ok()) << where;
                expect_near(*r, expected[0], where);
            } else if (f[2] == "none") {
                ASSERT_TRUE(r.is_err()) << where;
                EXPECT_EQ(r.unwrap_err(), returns_errc::no_root) << where;
                EXPECT_EQ(n, 0u) << where;
            } else {
                ASSERT_TRUE(r.is_err()) << where;
                EXPECT_EQ(r.unwrap_err(), returns_errc::multiple_roots) << where;
                ASSERT_EQ(n, k) << where;
                for (std::size_t i = 0; i < k; ++i)
                    EXPECT_NEAR(found[i], expected[i], 1e-7) << where << " rate " << i;
            }
        } else if (f[0] == "xirr") {
            ++seen[1];
            std::vector<cash_flow> flows;
            for (std::size_t i = 0, n = std::stoul(f.at(3)); i < n; ++i)
                flows.push_back({snicholls::test::fixture_date(f.at(4 + 2 * i)), std::stod(f.at(5 + 2 * i))});
            const auto r = xirr(flows);
            ASSERT_TRUE(r.is_ok()) << where;
            expect_near(*r, std::stod(f[2]), where);
        } else if (f[0] == "mwr") {
            ++seen[2];
            const auto flows = money_flows(f, 7);
            const auto r = money_weighted_return(minor(f.at(5)), minor(f.at(6)), flows, snicholls::test::fixture_date(f.at(3)),
                                                 snicholls::test::fixture_date(f.at(4)));
            ASSERT_TRUE(r.is_ok()) << where;
            expect_near(*r, std::stod(f[2]), where);
        } else if (f[0] == "dietz") {
            ++seen[3];
            const auto flows = money_flows(f, 8);
            const auto r = modified_dietz(minor(f.at(6)), minor(f.at(7)), flows, snicholls::test::fixture_date(f.at(4)),
                                          snicholls::test::fixture_date(f.at(5)));
            ASSERT_TRUE(r.is_ok()) << where;
            EXPECT_EQ(*r, (ratio{std::stoll(f[2]), std::stoll(f[3])})) << where;
        } else if (f[0] == "twr") {
            ++seen[4];
            std::vector<valuation> periods;
            for (std::size_t i = 0, n = std::stoul(f.at(4)); i < n; ++i)
                periods.push_back({minor(f.at(5 + 3 * i)), minor(f.at(6 + 3 * i)), minor(f.at(7 + 3 * i))});
            const auto r = time_weighted_return(periods);
            ASSERT_TRUE(r.is_ok()) << where;
            EXPECT_EQ(*r, (ratio{std::stoll(f[2]), std::stoll(f[3])})) << where;
            std::vector<double> each;
            for (const auto& p : periods)
                each.push_back(period_return(p)->to_double());
            expect_near(time_weighted_return(each), r->to_double(), where + " in double");
        } else {
            ADD_FAILURE() << "unknown record " << where;
        }
    }
    for (const std::size_t n : seen)
        EXPECT_GE(n, 3u) << "every measure has cases";
}

TEST(FinanceReturns, PeriodErrors)
{
    EXPECT_EQ(period_return({cents(100), money::zero(usd), cents(100)}).unwrap_err(), returns_errc::asset_mismatch);
    EXPECT_EQ(period_return({cents(100), cents(-100), cents(5)}).unwrap_err(), returns_errc::zero_capital);
    EXPECT_EQ(period_return({cents(-100), cents(0), cents(5)}).unwrap_err(), returns_errc::zero_capital);
    const money huge = money::from_units(aud, snicholls::uint256::from_limbs(0, 1));
    EXPECT_EQ(period_return({huge, cents(0), huge}).unwrap_err(), returns_errc::out_of_range);
    EXPECT_EQ(time_weighted_return(std::span<const valuation>{}).unwrap_err(), returns_errc::empty);
    const valuation mixed[] = {{cents(100), cents(0), cents(110)}, {money::zero(usd), money::zero(usd), money::zero(usd)}};
    EXPECT_EQ(time_weighted_return(mixed).unwrap_err(), returns_errc::asset_mismatch);
    const valuation broke[] = {{cents(100), cents(0), cents(0)}, {cents(0), cents(0), cents(10)}};
    EXPECT_EQ(time_weighted_return(broke).unwrap_err(), returns_errc::zero_capital);
}

TEST(FinanceReturns, AnExactChainThatLeaves64BitsSaysSo)
{
    std::vector<valuation> periods;
    std::int64_t v = 1'000'003;
    for (int i = 0; i < 12; ++i) {
        const std::int64_t e = v + 999'983 + 7 * i;   // unrelated primes-ish: nothing cancels
        periods.push_back({cents(v), cents(0), cents(e)});
        v = e + 11;
    }
    EXPECT_EQ(time_weighted_return(periods).unwrap_err(), returns_errc::out_of_range);
    std::vector<double> each;
    for (const auto& p : periods)
        each.push_back(period_return(p)->to_double());
    EXPECT_GT(time_weighted_return(each), 0.0) << "the double chain still answers";
}

TEST(FinanceReturns, DietzErrors)
{
    const money_flow late[] = {{on(2026, 4, 1), cents(10)}};
    EXPECT_EQ(modified_dietz(cents(100), cents(110), late, on(2026, 1, 1), on(2026, 3, 31)).unwrap_err(), returns_errc::invalid_dates);
    const money_flow unordered[] = {{on(2026, 2, 1), cents(10)}, {on(2026, 1, 15), cents(10)}};
    EXPECT_EQ(modified_dietz(cents(100), cents(110), unordered, on(2026, 1, 1), on(2026, 3, 31)).unwrap_err(),
              returns_errc::invalid_dates);
    EXPECT_EQ(modified_dietz(cents(100), cents(110), {}, on(2026, 1, 1), on(2026, 1, 1)).unwrap_err(), returns_errc::invalid_dates);
    EXPECT_EQ(modified_dietz(cents(100), money::zero(usd), {}, on(2026, 1, 1), on(2026, 2, 1)).unwrap_err(), returns_errc::asset_mismatch);
    const money_flow dollars[] = {{on(2026, 1, 2), money::zero(usd)}};
    EXPECT_EQ(modified_dietz(cents(100), cents(100), dollars, on(2026, 1, 1), on(2026, 2, 1)).unwrap_err(), returns_errc::asset_mismatch);
    EXPECT_EQ(modified_dietz(cents(0), cents(100), {}, on(2026, 1, 1), on(2026, 2, 1)).unwrap_err(), returns_errc::zero_capital);
    const money huge = money::from_units(aud, snicholls::uint256::from_limbs(0, 1));
    EXPECT_EQ(modified_dietz(huge, huge, {}, on(2026, 1, 1), on(2026, 2, 1)).unwrap_err(), returns_errc::out_of_range);
}

TEST(FinanceReturns, RateErrors)
{
    EXPECT_EQ(irr(std::span<const double>{}).unwrap_err(), returns_errc::empty);
    EXPECT_EQ(xirr(std::span<const cash_flow>{}).unwrap_err(), returns_errc::empty);
    const cash_flow backwards[] = {{on(2026, 2, 1), -100}, {on(2026, 1, 1), 110}};
    EXPECT_EQ(xirr(backwards).unwrap_err(), returns_errc::invalid_dates);
    const double beyond[] = {-1, 1e7};   // a rate of 9,999,999: beyond the range
    EXPECT_EQ(irr(beyond).unwrap_err(), returns_errc::no_root);
    const double zeros[] = {0, 0, 0};
    EXPECT_EQ(irr(zeros).unwrap_err(), returns_errc::no_root);
    double one[1]{};
    const double two_rates[] = {-100, 230, -132};
    EXPECT_EQ(irr_roots(two_rates, one), 2u) << "the count, though only one was written";
    EXPECT_NEAR(one[0], 0.1, 1e-9);
    EXPECT_EQ(irr_roots(std::span<const double>{}, one), 0u);
    EXPECT_EQ(money_weighted_return(cents(100), cents(110), {}, on(2026, 1, 1), on(2025, 1, 1)).unwrap_err(),
              returns_errc::invalid_dates);
    EXPECT_EQ(money_weighted_return(cents(0), cents(0), {}, on(2026, 1, 1), on(2027, 1, 1)).unwrap_err(), returns_errc::no_root);
}

TEST(FinanceReturns, TheEdgesOfTheRange)
{
    const double near_floor[] = {-100'000, 1};   // r = -0.99999, just inside the range
    const auto r = irr(near_floor);
    ASSERT_TRUE(r.is_ok());
    EXPECT_NEAR(*r, -0.99999, 1e-12);
    const double huge[] = {-1, 100'000};          // r = 99,999
    EXPECT_NEAR(*irr(huge), 99'999.0, 1e-6);
    const double double_root[] = {-1, 2, -1};   // (1 - x)^2: touches zero at r = 0, both sides negative
    double found[2]{};
    EXPECT_EQ(irr_roots(double_root, found), 1u) << "r = 0 is on the grid, so the touching root is found";
}

} // namespace
