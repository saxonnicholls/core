//
//  cash_ladder_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/cash_ladder.hpp>.
//

#include <snicholls/core/finance/cash_ladder.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace snicholls;
using namespace snicholls::finance;
using namespace std::chrono;
using snicholls::literals::operator""_iso4217;

constexpr asset aud = "AUD"_iso4217;
constexpr asset usd = "USD"_iso4217;
constexpr sys_days today = sys_days{2026y / November / 26};

constexpr money m(std::string_view text) { return *parse_money(aud, text); }
constexpr cash_flow flow(int days_from_today, std::string_view text) { return {today + days{days_from_today}, m(text)}; }

constexpr cash_ladder_terms terms(std::string_view opening, std::string_view threshold, overdue_policy p = {})
{
    return {today, m(opening), m(threshold), p};
}

// a ladder built entirely in a constant expression
constexpr bool built_at_compile_time()
{
    const std::array flows{flow(2, "-30"), flow(0, "10"), flow(2, "5"), flow(-1, "100"), flow(-1, "-7")};
    std::array<cash_rung, 6> rungs{};
    const auto l = make_cash_ladder(flows, terms("20", "0"), rungs);
    return l && l->rungs.size() == 2 && l->rungs[0].on == today && l->rungs[0].inflow == m("10") &&
           l->rungs[0].outflow == m("7") && l->rungs[0].cumulative == m("23") && l->rungs[1].net == m("-25") &&
           l->closing == m("-2") && l->first_below == today + days{2} && l->overdue.inflow == m("100");
}
static_assert(built_at_compile_time());
static_assert(noexcept(make_cash_ladder({}, std::declval<const cash_ladder_terms&>(), {})));
static_assert(overdue_policy{} == overdue_policy::prudent());
static_assert(overdue_policy::prudent().inflows == overdue_treatment::report_only);
static_assert(overdue_policy::prudent().outflows == overdue_treatment::due_today);

TEST(CashLadder, NoFlowsIsTheOpeningAlone)
{
    std::array<cash_rung, 1> rungs{};
    const auto l = make_cash_ladder({}, terms("100", "50"), rungs);
    ASSERT_TRUE(l);
    EXPECT_TRUE(l->rungs.empty());
    EXPECT_EQ(l->start, m("100"));
    EXPECT_EQ(l->closing, m("100"));
    EXPECT_EQ(l->lowest, m("100"));
    EXPECT_TRUE(l->stays_above());
    EXPECT_TRUE(l->overdue.empty());
    EXPECT_EQ(l->overdue.inflow, money::zero(aud));

    const auto below = make_cash_ladder({}, terms("10", "50"), std::span<cash_rung>{});
    ASSERT_TRUE(below);
    EXPECT_EQ(below->first_below, today);   // already below at the start
}

TEST(CashLadder, PerDateInflowOutflowNetAndCumulativeInDateOrder)
{
    // given out of order; two flows share a date
    const std::vector flows{flow(3, "-40"), flow(1, "25.50"), flow(3, "15"), flow(0, "-5.25"), flow(1, "-0.50")};
    std::vector<cash_rung> rungs(flows.size() + 1);
    const auto l = make_cash_ladder(flows, terms("100", "0"), rungs);
    ASSERT_TRUE(l);
    ASSERT_EQ(l->rungs.size(), 3u);
    EXPECT_EQ(l->rungs[0], (cash_rung{today, m("0"), m("5.25"), m("-5.25"), m("94.75")}));
    EXPECT_EQ(l->rungs[1], (cash_rung{today + days{1}, m("25.50"), m("0.50"), m("25"), m("119.75")}));
    EXPECT_EQ(l->rungs[2], (cash_rung{today + days{3}, m("15"), m("40"), m("-25"), m("94.75")}));
    EXPECT_EQ(l->closing, m("94.75"));
    EXPECT_EQ(l->lowest, m("94.75"));
    EXPECT_TRUE(l->stays_above());
    EXPECT_EQ(l->rungs.data(), rungs.data());   // written into the caller's span
}

TEST(CashLadder, FindsTheFirstDateBelowTheThresholdAndTheLowestPosition)
{
    const std::vector flows{flow(1, "-60"), flow(2, "-50"), flow(3, "200"), flow(4, "-300")};
    std::vector<cash_rung> rungs(5);
    const auto l = make_cash_ladder(flows, terms("100", "0"), rungs);
    ASSERT_TRUE(l);
    EXPECT_EQ(l->first_below, today + days{2});   // 40 on day 1, -10 on day 2
    EXPECT_EQ(l->lowest, m("-110"));                // the later, deeper breach
    EXPECT_EQ(l->closing, m("-110"));
    EXPECT_FALSE(l->stays_above());

    const auto at = make_cash_ladder(flows, terms("100", "40"), rungs);   // exactly at the floor is not a breach
    ASSERT_TRUE(at);
    EXPECT_EQ(at->first_below, today + days{2});
    const std::vector touch{flow(1, "-60")};
    EXPECT_TRUE(make_cash_ladder(touch, terms("100", "40"), rungs)->stays_above());
    EXPECT_EQ(make_cash_ladder(touch, terms("100", "40.01"), rungs)->first_below, today + days{1});
}

TEST(CashLadder, EveryFlowOverdueUnderEachTreatment)
{
    const std::vector flows{flow(-3, "80"), flow(-2, "-30"), flow(-1, "-20")};
    std::vector<cash_rung> rungs(4);

    const auto prudent = make_cash_ladder(flows, terms("10", "0"), rungs);
    ASSERT_TRUE(prudent);
    EXPECT_EQ(prudent->overdue, (overdue_bucket{m("80"), m("50"), 1, 2}));
    EXPECT_EQ(prudent->start, m("10"));   // the late inflow is not trusted
    ASSERT_EQ(prudent->rungs.size(), 1u);
    EXPECT_EQ(prudent->rungs[0], (cash_rung{today, m("0"), m("50"), m("-50"), m("-40")}));   // the late outflows, today
    EXPECT_EQ(prudent->first_below, today);

    const auto counted = make_cash_ladder(flows, terms("10", "0", overdue_policy::both(overdue_treatment::count)), rungs);
    ASSERT_TRUE(counted);
    EXPECT_TRUE(counted->rungs.empty());
    EXPECT_EQ(counted->start, m("40"));   // 10 + 80 - 30 - 20
    EXPECT_EQ(counted->closing, m("40"));
    EXPECT_TRUE(counted->stays_above());
    EXPECT_EQ(counted->overdue, prudent->overdue);   // reported whatever the treatment

    const auto reported = make_cash_ladder(flows, terms("10", "0", overdue_policy::both(overdue_treatment::report_only)), rungs);
    ASSERT_TRUE(reported);
    EXPECT_TRUE(reported->rungs.empty());
    EXPECT_EQ(reported->closing, m("10"));

    const auto today_all = make_cash_ladder(flows, terms("10", "0", overdue_policy::both(overdue_treatment::due_today)), rungs);
    ASSERT_TRUE(today_all);
    ASSERT_EQ(today_all->rungs.size(), 1u);
    EXPECT_EQ(today_all->rungs[0], (cash_rung{today, m("80"), m("50"), m("30"), m("40")}));

    const overdue_policy mixed{overdue_treatment::count, overdue_treatment::report_only};
    const auto l = make_cash_ladder(flows, terms("10", "0", mixed), rungs);
    ASSERT_TRUE(l);
    EXPECT_EQ(l->start, m("90"));
    EXPECT_TRUE(l->rungs.empty());
}

TEST(CashLadder, OverdueDueTodayJoinsTodaysFlows)
{
    const std::vector flows{flow(0, "5"), flow(-4, "-7"), flow(2, "1")};
    std::vector<cash_rung> rungs(4);
    const auto l = make_cash_ladder(flows, terms("0", "-100"), rungs);
    ASSERT_TRUE(l);
    ASSERT_EQ(l->rungs.size(), 2u);
    EXPECT_EQ(l->rungs[0], (cash_rung{today, m("5"), m("7"), m("-2"), m("-2")}));
    EXPECT_EQ(l->overdue, (overdue_bucket{m("0"), m("7"), 0, 1}));
}

TEST(CashLadder, ArithmeticIsExactPastSixtyFourBits)
{
    // 18-decimal amounts well past 2^64 minor units, and a one-unit difference that decides the breach
    const asset eth = *make_asset("ETH", 18);
    const auto e = [&](std::string_view t) { return *parse_money(eth, t); };
    const std::vector<cash_flow> flows{{today + days{1}, e("-123456789.000000000000000001")},
                                       {today + days{1}, e("0.000000000000000001")},
                                       {today + days{2}, e("-0.000000000000000001")}};
    std::vector<cash_rung> rungs(4);
    const cash_ladder_terms t{today, e("123456789"), e("0"), {}};
    const auto l = make_cash_ladder(flows, t, rungs);
    ASSERT_TRUE(l);
    EXPECT_EQ(l->rungs[0].cumulative, e("0"));
    EXPECT_EQ(l->rungs[0].inflow, e("0.000000000000000001"));
    EXPECT_EQ(l->closing, e("-0.000000000000000001"));
    EXPECT_EQ(l->first_below, today + days{2});
}

TEST(CashLadder, RefusesAnotherAssetAndOverflow)
{
    std::vector<cash_rung> rungs(3);
    const std::vector<cash_flow> mixed{flow(1, "5"), {today, *parse_money(usd, "1")}};
    EXPECT_EQ(make_cash_ladder(mixed, terms("0", "0"), rungs).err(), money_errc::asset_mismatch);
    const cash_ladder_terms other_floor{today, m("0"), *parse_money(usd, "0"), {}};
    EXPECT_EQ(make_cash_ladder({}, other_floor, rungs).err(), money_errc::asset_mismatch);

    const money huge = money::from_units(aud, uint256::max());
    const std::vector<cash_flow> too_much{{today, huge}, {today, m("0.01")}};
    EXPECT_EQ(make_cash_ladder(too_much, terms("0", "0"), rungs).err(), money_errc::overflow);
    const std::vector<cash_flow> late_too_much{{today - days{1}, huge}, {today - days{2}, m("0.01")}};
    EXPECT_EQ(make_cash_ladder(late_too_much, terms("0", "0"), rungs).err(), money_errc::overflow);
    const std::vector<cash_flow> position_too_much{{today, huge}};
    EXPECT_EQ(make_cash_ladder(position_too_much, terms("1", "0"), rungs).err(), money_errc::overflow);
}

TEST(CashLadder, RefusesASpanWithTooFewRungs)
{
    const std::vector flows{flow(1, "1"), flow(2, "1"), flow(3, "1")};
    std::vector<cash_rung> rungs(2);
    EXPECT_EQ(make_cash_ladder(flows, terms("0", "0"), rungs).err(), money_errc::buffer_too_small);
    const std::vector same_day{flow(1, "1"), flow(1, "2"), flow(1, "3")};
    EXPECT_TRUE(make_cash_ladder(same_day, terms("0", "0"), std::span{rungs}.first(1)));   // one date, one rung
}

} // namespace
