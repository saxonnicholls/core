//
//  holiday_calendar_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/holiday_calendar.hpp>: a market calendar of the caller's own, a table of
//  closures loaded at run time, used by core's adjustment rules. Exits non-zero if a date is not the one documented.
//

#include <snicholls/core/concepts/holiday_calendar.hpp>
#include <snicholls/core/finance/business_day.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [holiday_calendar]
// the caller's calendar: sorted closures, as a server loads them; anything with is_business_day models the concept
struct exchange_calendar {
    std::vector<std::chrono::sys_days> closures;   // sorted
    bool is_business_day(std::chrono::sys_days d) const
    {
        return snicholls::is_business_day(d) && !std::binary_search(closures.begin(), closures.end(), d);
    }
};
static_assert(snicholls::holiday_calendar<exchange_calendar>);
//! [holiday_calendar]
} // namespace

int main()
{
    std::printf("snicholls-core holiday_calendar concept demo\n");
    using namespace std::chrono;
    const exchange_calendar xnys{{sys_days{2026y / November / 26d}, sys_days{2026y / December / 25d}}};
    const auto r = snicholls::finance::adjust(sys_days{2026y / November / 26d}, snicholls::finance::business_day_convention::preceding, xnys);
    check(!xnys.is_business_day(sys_days{2026y / November / 26d}), "Thanksgiving is closed");
    check(xnys.is_business_day(sys_days{2026y / November / 27d}), "the day after is open");
    check(r && *r == sys_days{2026y / November / 25d}, "preceding moves Thanksgiving to the Wednesday");
    return failures == 0 ? 0 : 1;
}
