//
//  business_day_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/business_day.hpp>: a month-end payment date adjusted by every rule, over the
//  weekend calendar and over a calendar built from the caller's own holidays; a broken calendar refused. Exits
//  non-zero if a date is not the one the rule gives.
//

#include <snicholls/core/finance/business_day.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <set>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
void print(const char* rule, std::chrono::sys_days d)
{
    const std::chrono::year_month_day ymd{d};
    std::printf("  %-20s %d-%02u-%02u\n", rule, static_cast<int>(ymd.year()), static_cast<unsigned>(ymd.month()),
                static_cast<unsigned>(ymd.day()));
}
} // namespace

int main()
{
    std::printf("snicholls-core business_day demo\n");
    using namespace std::chrono;
    using namespace snicholls::finance;

    //! [business_day]
    const sys_days saturday = sys_days{2026y / October / 31d};
    const auto following = adjust(saturday, business_day_convention::following, weekend_calendar{});            // 2 November
    const auto modified = adjust(saturday, business_day_convention::modified_following, weekend_calendar{});    // 30 October

    // holidays are the caller's data: here a set, loaded at start-up
    const std::set<sys_days> holidays{sys_days{2026y / December / 25d}, sys_days{2026y / December / 28d}};
    const weekends_and_holidays market{[&](sys_days d) { return holidays.contains(d); }};
    const auto boxing_week = adjust(sys_days{2026y / December / 25d}, business_day_convention::following, market);   // 29 December
    //! [business_day]

    constexpr std::array rules{business_day_convention::unadjusted, business_day_convention::following,
                               business_day_convention::modified_following, business_day_convention::preceding,
                               business_day_convention::modified_preceding};
    constexpr std::array names{"unadjusted", "following", "modified following", "preceding", "modified preceding"};
    std::printf("  Saturday 2026-10-31:\n");
    for (std::size_t i = 0; i < rules.size(); ++i)
        print(names[i], *adjust(saturday, rules[i], weekend_calendar{}));
    const sys_days sunday = sys_days{2026y / November / 1d};
    std::printf("  Sunday 2026-11-01:\n");
    for (std::size_t i = 0; i < rules.size(); ++i)
        print(names[i], *adjust(sunday, rules[i], market));

    check(following && *following == sys_days{2026y / November / 2d}, "following rolls into November");
    check(modified && *modified == sys_days{2026y / October / 30d}, "modified following stays in October");
    check(*adjust(sunday, business_day_convention::modified_preceding, market) == sys_days{2026y / November / 2d},
          "modified preceding stays in November");
    check(boxing_week && *boxing_week == sys_days{2026y / December / 29d}, "the caller's holidays are skipped");
    check(weekend_calendar::name() == "weekends" && !weekend_calendar::is_business_day(saturday), "the reference calendar");
    const weekends_and_holidays never_open{[](sys_days) { return true; }};
    const auto broken = adjust(saturday, business_day_convention::following, never_open);
    check(!broken && broken.unwrap_err_unchecked() == business_day_errc::no_business_day,
          "a calendar with no business day is refused after max_business_day_search days");
    return failures == 0 ? 0 : 1;
}
