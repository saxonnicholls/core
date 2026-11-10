//
//  business_days_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/business_days.hpp>.
//

// Demo: interfaces/utc/business_days.hpp - O(1) Monday-to-Friday arithmetic and month arithmetic.

#include <chrono>
#include <iostream>

#include <snicholls/core/time/business_days.hpp>
#include <snicholls/core/time/rfc3339.hpp>

//! [demo]
int main()
{
    using namespace std::chrono;
    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << '\n';
        failures += !ok;
    };
    const auto ymd = [](sys_days d) { return snicholls::to_string(year_month_day(d)); };

    const sys_days friday = sys_days{2024y / May / 3};
    const sys_days saturday = friday + days(1);
    const sys_days monday = friday + days(3);
    std::cout << "Friday " << ymd(friday) << " + 3 business days = " << ymd(snicholls::add_business_days(friday, 3)) << '\n';
    check("weekend / business day", snicholls::is_weekend(Saturday) && snicholls::is_weekend(Sunday) && !snicholls::is_weekend(Friday) &&
                                        snicholls::is_business_day(friday) && !snicholls::is_business_day(saturday));
    check("Friday + 1 = Monday, + 3 = Wednesday", snicholls::add_business_days(friday, 1) == monday &&
                                                     snicholls::add_business_days(friday, 3) == monday + days(2));
    check("Saturday + 1 = Monday, Sunday + 1 = Monday", snicholls::add_business_days(saturday, 1) == monday &&
                                                           snicholls::add_business_days(saturday + days(1), 1) == monday);
    check("Monday - 1 = Friday, Saturday - 1 = Friday, Sunday - 2 = Thursday",
          snicholls::add_business_days(monday, -1) == friday && snicholls::add_business_days(saturday, -1) == friday &&
              snicholls::add_business_days(saturday + days(1), -2) == friday - days(1));
    check("+10 and -10 are two weeks", snicholls::add_business_days(monday, 10) == monday + days(14) &&
                                           snicholls::add_business_days(monday, -10) == monday - days(14));
    check("n == 0 leaves the day unchanged", snicholls::add_business_days(saturday, 0) == saturday);
    check("business days in [Friday, next Friday) = 5, reversed = -5, across 1970 too",
          snicholls::business_days_between(friday, friday + days(7)) == 5 && snicholls::business_days_between(friday + days(7), friday) == -5 &&
              snicholls::business_days_between(sys_days{1969y / December / 26}, sys_days{1970y / January / 2}) == 5);
    std::cout << "last Friday before " << ymd(friday) << " = " << ymd(snicholls::previous_weekday(friday, Friday)) << '\n';
    check("previous / next weekday are strict", snicholls::previous_weekday(friday, Friday) == friday - days(7) &&
                                                   snicholls::previous_weekday(monday, Friday) == friday &&
                                                   snicholls::next_weekday(friday, Monday) == monday &&
                                                   snicholls::next_weekday(monday, Monday) == monday + days(7));
    std::cout << "2024-01-31 + 1 month = " << snicholls::to_string(snicholls::add_months(2024y / January / 31, 1)) << '\n';
    check("add_months clamps to the end of the month", snicholls::add_months(2024y / January / 31, 1) == 2024y / February / 29 &&
                                                          snicholls::add_months(2024y / March / 31, -1) == 2024y / February / 29 &&
                                                          snicholls::add_months(2024y / March / 15, -3) == 2023y / December / 15);

    std::cout << (failures == 0 ? "OK\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
//! [demo]
