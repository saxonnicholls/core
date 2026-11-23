//
//  holiday_calendar_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/holiday_calendar.hpp>: calendars held by market behind the interface,
//  chosen per trade at run time, and used by the adjustment rules. Exits non-zero if a date is wrong.
//

#include <snicholls/core/finance/business_day.hpp>
#include <snicholls/core/interfaces/holiday_calendar.hpp>

#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>

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
    std::printf("snicholls-core holiday_calendar interface demo\n");
    using namespace std::chrono;
    using namespace snicholls;

    //! [interface]
    struct set_calendar {   // a market's holidays, loaded at start-up
        std::set<sys_days> holidays;
        bool is_business_day(sys_days d) const noexcept { return snicholls::is_business_day(d) && !holidays.contains(d); }
    };
    std::map<std::string, std::unique_ptr<holiday_calendar_interface>> calendars;
    calendars["XLON"] = std::make_unique<holiday_calendar_model<set_calendar>>(set_calendar{{sys_days{2026y / December / 25d}, sys_days{2026y / December / 28d}}});
    calendars["weekends"] = std::make_unique<holiday_calendar_model<finance::weekend_calendar>>();

    const holiday_calendar_interface& london = *calendars.at("XLON");   // from the trade's terms
    const auto pay = finance::adjust(sys_days{2026y / December / 25d}, finance::business_day_convention::modified_following, london);
    //! [interface]

    check(pay && *pay == sys_days{2026y / December / 29d}, "Christmas and the Boxing Day substitute roll to the 29th");
    check(calendars.at("weekends")->is_business_day(sys_days{2026y / December / 25d}), "the weekend calendar has no holidays");
    check(!london.is_business_day(sys_days{2026y / December / 26d}), "Saturday is closed");
    return failures == 0 ? 0 : 1;
}
