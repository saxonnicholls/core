//
//  day_count_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/day_count_conformance.hpp>: the kit run over core's conventions and over a
//  broken one, which it names. Exits non-zero if the kit's verdict is not the one documented.
//

#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/interfaces/day_count_conformance.hpp>

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

struct per_week {   // broken: a "year" of seven days
    static constexpr std::string_view name() noexcept { return "ACT/7"; }
    static constexpr std::int64_t day_count(std::chrono::sys_days a, std::chrono::sys_days b) noexcept { return (b - a).count(); }
    static constexpr snicholls::finance::year_fraction year_fraction(std::chrono::sys_days a, std::chrono::sys_days b) noexcept
    {
        return {day_count(a, b), 7};
    }
};
} // namespace

int main()
{
    std::printf("snicholls-core day_count conformance demo\n");
    using namespace std::chrono;
    //! [conformance]
    constexpr sys_days from = sys_days{2024y / February / 1d};
    static_assert(!snicholls::check_day_count(snicholls::finance::thirty_360{}, from, 45));   // conforms
    const auto broken = snicholls::check_day_count(per_week{}, from, 45);                     // names the failed check
    //! [conformance]
    check(!snicholls::check_day_count(snicholls::finance::act_act_isda{}, from, 90), "ACT/ACT ISDA conforms");
    check(broken && *broken == "day_count: the fraction is not a year's worth of days per counted day", "ACT/7 is caught");
    if (broken)
        std::printf("  %.*s\n", static_cast<int>(broken->size()), broken->data());
    return failures == 0 ? 0 : 1;
}
