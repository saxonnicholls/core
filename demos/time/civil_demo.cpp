//
//  civil_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/civil.hpp>.
//

// Demo: interfaces/utc/civil.hpp - Hinnant's calendar algorithms on plain integers.

#include <cstdint>
#include <iostream>

#include <snicholls/core/time/civil.hpp>

//! [demo]
int main()
{
    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << '\n';
        failures += !ok;
    };

    const std::int64_t z = snicholls::days_from_civil(2024, 2, 29);
    const snicholls::civil_date c = snicholls::civil_from_days(z);
    std::cout << "2024-02-29 is day " << z << " since 1970-01-01; back: " << c.year << '-' << c.month << '-' << c.day
              << ", weekday " << snicholls::weekday_from_days(z) << " (0 = Sunday)\n";
    check("days_from_civil(2024, 2, 29) == 19782", z == 19782);
    check("civil_from_days round-trips", c == snicholls::civil_date{2024, 2, 29});
    check("2024-02-29 is a Thursday", snicholls::weekday_from_days(z) == 4);
    check("before the epoch: day -1 is 1969-12-31, a Wednesday",
          snicholls::civil_from_days(-1) == snicholls::civil_date{1969, 12, 31} && snicholls::weekday_from_days(-1) == 3);
    check("day -5 is a Saturday (the negative branch)", snicholls::weekday_from_days(-5) == 6);
    check("year 0 (1 BC) is a leap year; 1900 is not; 2000 is",
          snicholls::is_leap_year(0) && !snicholls::is_leap_year(1900) && snicholls::is_leap_year(2000));
    check("last days: Feb 2024 = 29, Feb 2023 = 28, Apr = 30, Dec = 31",
          snicholls::last_day_of_month(2024, 2) == 29 && snicholls::last_day_of_month(2023, 2) == 28 &&
              snicholls::last_day_of_month(2023, 4) == 30 && snicholls::last_day_of_month(2023, 12) == 31);
    check("no validation: 2025-02-30 is 2025-03-02",
          snicholls::days_from_civil(2025, 2, 30) == snicholls::days_from_civil(2025, 3, 2));
    check("default civil_date is the epoch", snicholls::civil_date{} == snicholls::civil_from_days(0));

    std::cout << (failures == 0 ? "OK\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
//! [demo]
