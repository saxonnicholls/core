//
//  holiday_calendar_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/holiday_calendar_conformance.hpp>: the kit run over a calendar from data and
//  over one that never opens. Exits non-zero if the kit's verdict is not the one documented.
//

#include <snicholls/core/finance/business_day.hpp>
#include <snicholls/core/interfaces/holiday_calendar_conformance.hpp>

#include <chrono>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    std::printf("snicholls-core holiday_calendar conformance demo\n");
    using namespace std::chrono;
    //! [conformance]
    constexpr sys_days from = sys_days{2026y / January / 1d};
    const auto new_year = [](sys_days d) { return d == sys_days{2026y / January / 1d}; };
    const auto good = snicholls::check_holiday_calendar(snicholls::finance::weekends_and_holidays{new_year}, from, 365);
    const auto never = snicholls::check_holiday_calendar(snicholls::finance::weekends_and_holidays{[](sys_days) { return true; }}, from, 7);
    //! [conformance]
    check(!good, "a calendar with a holiday conforms");
    check(never && *never == "holiday_calendar: no business day within finance::max_business_day_search days",
          "a calendar that never opens is caught");
    if (never)
        std::printf("  %.*s\n", static_cast<int>(never->size()), never->data());
    return failures == 0 ? 0 : 1;
}
