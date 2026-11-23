//
//  week_date_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/week_date.hpp>: the ISO week dates around a new year, both directions, and the
//  validation of a week that does not exist. Exits non-zero if a week date is not the one ISO 8601 gives.
//

#include <snicholls/core/time/week_date.hpp>

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
    std::printf("snicholls-core week_date demo\n");
    using namespace std::chrono;

    //! [week_date]
    constexpr snicholls::iso_week_date w = snicholls::to_iso_week_date(sys_days{2024y / December / 30d});
    static_assert(w == snicholls::iso_week_date{2025, 1, 1});                 // a Monday in week 1 of 2025

    constexpr auto settlement = snicholls::make_iso_week_date(2026, 53, 4);   // 2026 has 53 weeks
    static_assert(settlement && snicholls::to_sys_days(*settlement) == sys_days{2026y / December / 31d});
    static_assert(!snicholls::make_iso_week_date(2027, 53, 1));               // 2027 has 52
    //! [week_date]

    for (sys_days d = sys_days{2026y / December / 26d}; d <= sys_days{2027y / January / 5d}; d += days(1)) {
        const auto x = snicholls::to_iso_week_date(d);
        const year_month_day ymd{d};
        std::printf("  %d-%02u-%02u  %d-W%02u-%u\n", static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()),
                    static_cast<unsigned>(ymd.day()), x.year, x.week, x.weekday);
    }
    check(snicholls::iso_weeks_in_year(2026) == 53 && snicholls::iso_weeks_in_year(2027) == 52, "2026 has 53 weeks, 2027 52");
    check(snicholls::to_iso_week_date(sys_days{2027y / January / 3d}) == snicholls::iso_week_date{2026, 53, 7},
          "3 January 2027 is still in 2026's last week");
    check(snicholls::make_iso_week_date(2026, 1, 8).unwrap_err_unchecked() == snicholls::week_date_errc::invalid_weekday,
          "weekday 8 is refused");
    check(snicholls::make_iso_week_date(2027, 53, 1).unwrap_err_unchecked() == snicholls::week_date_errc::invalid_week,
          "week 53 of 2027 is refused");
    return failures == 0 ? 0 : 1;
}
