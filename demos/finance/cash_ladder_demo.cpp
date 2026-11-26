//
//  cash_ladder_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/cash_ladder.hpp>: a week of settlements against a trust account under the prudent
//  overdue policy and under counting everything; the first breach of the floor; another asset and a short span
//  refused. Exits non-zero if a figure is not the one documented.
//

#include <snicholls/core/finance/cash_ladder.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
void print(const snicholls::finance::cash_rung& r)
{
    const std::chrono::year_month_day ymd{r.on};
    std::printf("  %d-%02u-%02u  in %10s  out %10s  net %10s  cumulative %10s\n", static_cast<int>(ymd.year()),
                static_cast<unsigned>(ymd.month()), static_cast<unsigned>(ymd.day()), to_string(r.inflow).c_str(),
                to_string(r.outflow).c_str(), to_string(r.net).c_str(), to_string(r.cumulative).c_str());
}
} // namespace

int main()
{
    std::printf("snicholls-core cash_ladder demo\n");
    using namespace std::chrono;
    using namespace snicholls;
    using namespace snicholls::finance;
    using snicholls::literals::operator""_iso4217;
    constexpr asset aud = "AUD"_iso4217;
    const auto a = [&](const char* t) { return *parse_money(aud, t); };

    //! [ladder]
    const sys_days today = 2026y / November / 26;
    const std::array<cash_flow, 6> flows{{
        {today - days{2}, a("1200.00")},   // a provider payout that has not arrived: overdue
        {today - days{1}, a("-300.00")},   // a share purchase that has not settled yet: overdue
        {today, a("-450.00")},             // card payouts today
        {today + days{1}, a("-900.00")},   // a stock-market settlement
        {today + days{4}, a("2500.00")},   // Australia Post's twice-weekly settlement
        {today + days{1}, a("150.00")},    // a refund received
    }};
    std::array<cash_rung, flows.size() + 1> rungs{};
    const cash_ladder_terms terms{today, a("1000.00"), a("0.00"), overdue_policy::prudent()};
    const auto ladder = make_cash_ladder(flows, terms, rungs);
    // ladder->rungs: per date inflow, outflow, net, cumulative; ladder->first_below: the first breach
    //! [ladder]

    std::puts("The prudent policy (overdue inflows reported only, overdue outflows due today)");
    if (!ladder) {
        std::puts("  FAIL: no ladder");
        return 1;
    }
    for (const auto& r : ladder->rungs)
        print(r);
    check(ladder->rungs.size() == 3, "three dates: today, tomorrow and Monday");
    check(ladder->rungs[0].outflow == a("750.00"), "today pays the card payouts and the overdue purchase");
    check(ladder->rungs[0].cumulative == a("250.00"), "250 left tonight");
    check(ladder->first_below == today + days{1}, "below zero tomorrow, before Monday's settlement");
    check(ladder->lowest == a("-500.00"), "500 short at the lowest");
    check(ladder->closing == a("2000.00"), "2000 at the close");
    check(ladder->overdue.inflow == a("1200.00") && ladder->overdue.inflows == 1, "the late payout is reported");
    check(ladder->overdue.outflow == a("300.00") && ladder->overdue.outflows == 1, "the late purchase is reported");

    std::puts("Counting every overdue flow as settled");
    const cash_ladder_terms trusting{today, a("1000.00"), a("0.00"), overdue_policy::both(overdue_treatment::count)};
    const auto counted = make_cash_ladder(flows, trusting, rungs);
    check(counted && counted->start == a("1900.00"), "the start includes the late payout and purchase");
    check(counted && counted->stays_above(), "and the account never goes below zero: the optimism the default avoids");
    const overdue_policy only_out{overdue_treatment::report_only, overdue_treatment::report_only};
    check(make_cash_ladder(flows, {today, a("1000.00"), a("0.00"), only_out}, rungs)->start == a("1000.00"),
          "reporting both leaves the opening alone");
    check(make_cash_ladder(flows, {today, a("1000.00"), a("0.00"), overdue_policy::both(overdue_treatment::due_today)}, rungs)
                  ->rungs[0].inflow == a("1200.00"),
          "due today moves the late payout into today's inflow");

    std::puts("Refusals");
    const std::array<cash_flow, 1> usd_flow{{{today, *parse_money("USD"_iso4217, "1")}}};
    check(make_cash_ladder(usd_flow, terms, rungs).err() == money_errc::asset_mismatch, "one asset per ladder");
    std::array<cash_rung, 2> two{};
    check(make_cash_ladder(flows, terms, two).err() == money_errc::buffer_too_small, "a span with too few rungs");
    check(make_cash_ladder({}, terms, two)->closing == a("1000.00"), "no flows: the opening is the close");

    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
