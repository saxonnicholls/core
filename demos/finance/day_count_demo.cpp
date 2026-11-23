//
//  day_count_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/day_count.hpp>: one period under each of the five conventions, with the days
//  each counts and its exact year fraction. Exits non-zero if a value is not the one ISDA defines.
//

#include <snicholls/core/finance/day_count.hpp>

#include <chrono>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

template <snicholls::day_count_convention C>
void show(C c, std::chrono::sys_days a, std::chrono::sys_days b)
{
    const auto f = c.year_fraction(a, b);
    std::printf("  %-13.*s %4lld days  %lld/%lld = %.12f\n", static_cast<int>(c.name().size()), c.name().data(),
                static_cast<long long>(c.day_count(a, b)), static_cast<long long>(f.numerator()),
                static_cast<long long>(f.denominator()), f.to_double());
}
} // namespace

int main()
{
    std::printf("snicholls-core day_count demo\n");
    using namespace std::chrono;
    using namespace snicholls::finance;

    //! [day_count]
    const sys_days start = sys_days{2023y / December / 31d}, end = sys_days{2024y / March / 31d};
    constexpr year_fraction money_market = act_360::year_fraction(sys_days{2024y / January / 2d}, sys_days{2024y / April / 2d});
    static_assert(money_market == year_fraction{91, 360});

    const auto bond = thirty_360::day_count(start, end);        // 31st -> 30th at both ends: 90 days
    const auto euro = thirty_e_360::day_count(start, end);      // 90 days
    const auto isda = act_act_isda::year_fraction(start, end);  // 1/365 + 90/366, exactly
    //! [day_count]

    show(act_360{}, start, end);
    show(act_365_fixed{}, start, end);
    show(thirty_360{}, start, end);
    show(thirty_e_360{}, start, end);
    show(act_act_isda{}, start, end);
    const sys_days mid = sys_days{2024y / January / 15d};
    show(thirty_360{}, mid, end);
    show(thirty_e_360{}, mid, end);

    check(bond == 90 && euro == 90, "30/360 and 30E/360 count 90 days from 31 December to 31 March");
    check(thirty_360::day_count(mid, end) == 76 && thirty_e_360::day_count(mid, end) == 75,
          "from the 15th, 30/360 keeps an end on the 31st and 30E/360 does not");
    check(isda == year_fraction{366 + 90 * 365, 133590}, "ACT/ACT ISDA splits the period at the year end");
    check(act_365_fixed::year_fraction(start, end) == year_fraction{91, 365}, "ACT/365F: 91 actual days over 365");
    check(act_360::day_count(end, start) == -91, "a reversed period counts negative");
    return failures == 0 ? 0 : 1;
}
