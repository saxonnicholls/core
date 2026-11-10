//
//  solar_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/solar.hpp>: the sun's position at an instant, daylight at a place, and the
//  day/night terminator. Exits non-zero if any result is not the one documented.
//

#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/solar.hpp>

#include <cmath>
#include <iostream>

int main()
{
    using namespace snicholls::literals;
    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << '\n';
        failures += !ok;
    };

    //! [solar]
    const snicholls::solar_position sun = snicholls::sun_position("2026-06-21T12:00:00Z"_utc);   // June solstice
    const bool london = snicholls::is_daylight(sun, 51.5, -0.1);
    const double elevation = snicholls::solar_elevation_deg(sun, 51.5, -0.1);
    const auto line = snicholls::terminator(sun, 1.0);   // 361 (longitude, latitude) points
    //! [solar]
    std::cout << "declination " << sun.declination_deg << " deg, equation of time " << sun.equation_of_time_min
              << " min, subsolar longitude " << sun.subsolar_longitude_deg << " deg\n";
    check("solstice declination is the obliquity, 23.436 deg", std::abs(sun.declination_deg - 23.436) < 0.01);
    check("subsolar latitude equals the declination", sun.subsolar_latitude_deg == sun.declination_deg);
    check("London is in daylight at noon UTC", london && elevation > 60.0);
    check("Sydney is in the dark at 22:00 local", !snicholls::is_daylight(sun, -33.9, 151.2));
    check("the terminator has 361 points at elevation 0", line.size() == 361 &&
          std::abs(snicholls::solar_elevation_deg(sun, line[90].second, line[90].first)) < 1e-6);
    const auto equinox = snicholls::terminator(snicholls::sun_position("2026-03-20T14:46:00Z"_utc), 30.0);
    check("an equinox terminator is finite everywhere", !std::isnan(equinox.front().second) && equinox.size() == 13);

    std::cout << (failures == 0 ? "OK\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
