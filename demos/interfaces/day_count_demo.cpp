//
//  day_count_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/day_count.hpp>: the day count of a trade chosen at run time from its terms,
//  through the interface, and passed to code written for the concept. Exits non-zero if a fraction is wrong.
//

#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/interfaces/day_count.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [interface]
// one model per convention, as constants; a trade's terms name one
constexpr snicholls::day_count_model<snicholls::finance::act_360> act_360;
constexpr snicholls::day_count_model<snicholls::finance::act_365_fixed> act_365f;
constexpr snicholls::day_count_model<snicholls::finance::act_act_isda> act_act;

const snicholls::day_count_interface* by_name(std::string_view terms)
{
    const std::array<const snicholls::day_count_interface*, 3> offered{&act_360, &act_365f, &act_act};
    for (const snicholls::day_count_interface* c : offered)
        if (c->name() == terms)
            return c;
    return nullptr;
}

// code written for the concept takes the interface too
template <snicholls::day_count_convention C>
double in_years(const C& c, std::chrono::sys_days a, std::chrono::sys_days b) { return c.year_fraction(a, b).to_double(); }
//! [interface]
} // namespace

int main()
{
    std::printf("snicholls-core day_count interface demo\n");
    using namespace std::chrono;
    const sys_days a = sys_days{2026y / January / 1d}, b = sys_days{2027y / January / 1d};
    const snicholls::day_count_interface* chosen = by_name("ACT/360");
    check(chosen != nullptr && chosen->day_count(a, b) == 365, "ACT/360 found by name counts 365 days in 2026");
    check(chosen != nullptr && chosen->year_fraction(a, b) == snicholls::finance::year_fraction{73, 72}, "365/360 = 73/72");
    check(by_name("ACT/ACT ISDA") != nullptr && in_years(*by_name("ACT/ACT ISDA"), a, b) == 1.0, "ACT/ACT ISDA: one year");
    check(by_name("BUS/252") == nullptr, "a convention not offered is not found");
    return failures == 0 ? 0 : 1;
}
