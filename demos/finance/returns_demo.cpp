//
//  returns_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/returns.hpp>: a quarter of a fund with a contribution and a withdrawal, measured
//  four ways (exact time-weighted, exact modified Dietz, the money-weighted return, and the time-weighted return in
//  double), then the IRR of a bond's flows, flows with two rates and flows with none. Exits non-zero if a result is
//  not the one documented.
//

#include <snicholls/core/finance/returns.hpp>
#include <snicholls/core/money/asset.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
constexpr snicholls::asset aud = snicholls::checked_asset("AUD", 2);
snicholls::money cents(std::int64_t v) { return snicholls::money::from_units(aud, static_cast<std::uint64_t>(std::llabs(v)), v < 0); }
} // namespace

int main()
{
    std::printf("snicholls-core finance returns demo\n");
    using namespace std::chrono;
    using namespace snicholls::finance;

    //! [twr]
    // A quarter: $1m on 1 January; $500k in on 1 February; $200k out on 1 March; $1.32m at the end.
    const valuation months[] = {
        {cents(100'000'000), cents(0), cents(101'000'000)},             // January: +1%
        {cents(101'000'000), cents(50'000'000), cents(153'000'000)},    // February, after the contribution
        {cents(153'000'000), cents(-20'000'000), cents(132'000'000)},   // March, after the withdrawal
    };
    const auto twr = time_weighted_return(months);                       // exact
    const cash_flow flows[] = {{sys_days{2026y / February / 1d}, cents(50'000'000)},
                                {sys_days{2026y / March / 1d}, cents(-20'000'000)}};
    const sys_days start = sys_days{2026y / January / 1d}, end = sys_days{2026y / March / 31d};
    const auto dietz = modified_dietz(cents(100'000'000), cents(132'000'000), flows, start, end);   // exact
    const auto mwr = money_weighted_return(cents(100'000'000), cents(132'000'000), flows, start, end);   // annual
    //! [twr]

    std::printf("  TWR %lld/%lld = %.6f%%, modified Dietz %lld/%lld = %.6f%%, MWR %.6f%% a year\n",
                static_cast<long long>(twr->numerator()), static_cast<long long>(twr->denominator()), 100 * twr->to_double(),
                static_cast<long long>(dietz->numerator()), static_cast<long long>(dietz->denominator()),
                100 * dietz->to_double(), 100 * *mwr);
    check(*twr == ratio{7874, 502075}, "TWR = 1.01 * 153/151 * 132/133 - 1, exactly");
    check(std::fabs(dietz->to_double() - twr->to_double()) < 0.002, "modified Dietz is close to it");
    std::vector<double> each;
    for (const auto& m : months)
        each.push_back(period_return(m)->to_double());
    check(std::fabs(time_weighted_return(each) - twr->to_double()) < 1e-15, "chained in double, the same to rounding");
    check(mwr.is_ok() && *mwr > 0, "the investor gained");

    //! [irr]
    const double bond[] = {-950, 50, 50, 50, 50, 1050};   // bought at 95, 5% coupons, five years
    const auto yield = irr(bond);                         // about 6.19% a period
    const double two[] = {-100, 230, -132};               // worth zero at 10% and at 20%
    double rates[4]{};
    const std::size_t n = irr_roots(two, rates);
    const xirr_flow dated[] = {{sys_days{2026y / January / 1d}, -10'000}, {sys_days{2026y / March / 1d}, 2'750},
                               {sys_days{2026y / October / 30d}, 4'250}, {sys_days{2027y / February / 15d}, 3'250},
                               {sys_days{2027y / April / 1d}, 2'750}};
    const auto annual = xirr(dated);
    //! [irr]

    std::printf("  bond IRR %.6f%%, XIRR %.6f%%, two rates %.6f and %.6f\n", 100 * *yield, 100 * *annual, rates[0], rates[1]);
    check(std::fabs(*yield - 0.061932282681517403) < 1e-12, "the bond's IRR is numpy-financial's");
    check(irr(two).unwrap_err() == returns_errc::multiple_roots && n == 2 && std::fabs(rates[0] - 0.1) < 1e-9 &&
              std::fabs(rates[1] - 0.2) < 1e-9,
          "two rates: the IRR is not defined, and irr_roots lists both");
    check(std::fabs(*annual - 0.37485859768737118) < 1e-9, "XIRR agrees with the decimal reference");
    const double all_in[] = {100, 50};
    check(irr(all_in).unwrap_err() == returns_errc::no_root, "flows of one sign have no rate");
    check(irr(std::span<const double>{}).unwrap_err() == returns_errc::empty, "no flows");
    check(time_weighted_return(std::span<const valuation>{}).unwrap_err() == returns_errc::empty, "no periods");
    check(period_return({cents(100), cents(-100), cents(0)}).unwrap_err() == returns_errc::zero_capital,
          "nothing invested has no return");
    return failures == 0 ? 0 : 1;
}
