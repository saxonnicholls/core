//
//  day_count_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/day_count.hpp>: an accrual written once over the concept, run with core's
//  conventions and with one of the caller's own. Exits non-zero if an accrual is not the one documented.
//

#include <snicholls/core/concepts/day_count.hpp>
#include <snicholls/core/finance/day_count.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [day_count]
// interest in minor units on a notional at a rate in basis points, for any convention, rounded down once
template <snicholls::day_count_convention C>
constexpr std::int64_t accrue(const C& convention, std::int64_t notional, std::int64_t bp, std::chrono::sys_days a, std::chrono::sys_days b)
{
    const auto f = convention.year_fraction(a, b);
    return notional * bp * f.numerator() / (10'000 * f.denominator());
}

// a convention of the caller's own models the concept by having the three members
struct act_364 {
    static constexpr std::string_view name() noexcept { return "ACT/364"; }
    static constexpr std::int64_t day_count(std::chrono::sys_days a, std::chrono::sys_days b) noexcept { return (b - a).count(); }
    static constexpr snicholls::finance::year_fraction year_fraction(std::chrono::sys_days a, std::chrono::sys_days b) noexcept
    {
        return {day_count(a, b), 364};
    }
};
static_assert(snicholls::day_count_convention<act_364>);
//! [day_count]
} // namespace

int main()
{
    std::printf("snicholls-core day_count concept demo\n");
    using namespace std::chrono;
    const sys_days a = sys_days{2026y / January / 1d}, b = sys_days{2026y / July / 2d};   // 182 days
    const std::int64_t on_360 = accrue(snicholls::finance::act_360{}, 1'000'000'00, 500, a, b);
    const std::int64_t on_364 = accrue(act_364{}, 1'000'000'00, 500, a, b);
    std::printf("  5%% on 1,000,000.00 for 182 days: ACT/360 %lld, ACT/364 %lld cents\n", static_cast<long long>(on_360),
                static_cast<long long>(on_364));
    check(on_360 == 2'527'777, "ACT/360: 100,000,000 * 0.05 * 182 / 360, rounded down");
    check(on_364 == 2'500'000, "ACT/364: 182 days is exactly half a year");
    return failures == 0 ? 0 : 1;
}
