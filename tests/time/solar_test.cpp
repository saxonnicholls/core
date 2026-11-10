//
//  solar_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/solar.hpp>.
//

// interfaces/utc/solar.hpp against published astronomy: the 2026 equinoxes and solstices (USNO, "Earth's Seasons":
// 2026-03-20 14:46, 2026-06-21 08:24, 2026-09-23 00:05, 2026-12-21 20:50 UTC), equation-of-time extremes, solar noon.

#include <cmath>

#include <gtest/gtest.h>

#include <snicholls/core/time/solar.hpp>
#include <snicholls/core/time/rfc3339.hpp>

namespace {

using namespace std::chrono;
using namespace snicholls::literals;

TEST(UtcSolar, EquinoxesAndSolstices2026)
{
    // Declination changes ~0.4 degree/day at the equinoxes: within a minute of the published instant it is ~0.
    EXPECT_NEAR(snicholls::sun_position("2026-03-20T14:46:00Z"_utc).declination_deg, 0.0, 0.01);
    EXPECT_LT(snicholls::sun_position("2026-03-20T12:00:00Z"_utc).declination_deg, 0.0);
    EXPECT_GT(snicholls::sun_position("2026-03-20T18:00:00Z"_utc).declination_deg, 0.0);
    EXPECT_NEAR(snicholls::sun_position("2026-09-23T00:05:00Z"_utc).declination_deg, 0.0, 0.01);
    EXPECT_GT(snicholls::sun_position("2026-09-22T20:00:00Z"_utc).declination_deg, 0.0);
    // Solstices: the obliquity of the ecliptic, 23.436 degrees in 2026.
    EXPECT_NEAR(snicholls::sun_position("2026-06-21T08:24:00Z"_utc).declination_deg, 23.436, 0.01);
    EXPECT_NEAR(snicholls::sun_position("2026-12-21T20:50:00Z"_utc).declination_deg, -23.436, 0.01);
}

TEST(UtcSolar, EquationOfTimeExtremes)
{
    // Published extremes: about -14.2 min near Feb 11 and +16.4 min near Nov 3.
    EXPECT_NEAR(snicholls::sun_position("2026-02-11T12:00:00Z"_utc).equation_of_time_min, -14.2, 0.2);
    EXPECT_NEAR(snicholls::sun_position("2026-11-03T12:00:00Z"_utc).equation_of_time_min, 16.4, 0.2);
}

TEST(UtcSolar, SubsolarPointElevationAndTerminator)
{
    // At 12:00 UTC on the June solstice the sun is overhead near longitude 0 (shifted by the equation of time).
    const auto sun = snicholls::sun_position("2026-06-21T12:00:00Z"_utc);
    EXPECT_NEAR(sun.subsolar_longitude_deg, -sun.equation_of_time_min / 4.0, 1e-9);
    EXPECT_NEAR(snicholls::solar_elevation_deg(sun, sun.subsolar_latitude_deg, sun.subsolar_longitude_deg), 90.0, 1e-6);
    EXPECT_TRUE(snicholls::is_daylight(sun, 51.5, 0.0));     // London at noon
    EXPECT_FALSE(snicholls::is_daylight(sun, -33.9, 151.2)); // Sydney at 22:00
    EXPECT_TRUE(snicholls::is_daylight(sun, 78.2, 15.6));    // Svalbard: midnight sun at midsummer
    const auto t = snicholls::terminator(sun, 1.0);
    ASSERT_EQ(t.size(), 361U);
    for (const auto& [lon, lat] : t) EXPECT_NEAR(snicholls::solar_elevation_deg(sun, lat, lon), 0.0, 1e-6);
    // An equinox: the terminator is (nearly) a meridian pair, never NaN.
    for (const auto& [lon, lat] : snicholls::terminator(snicholls::sun_position("2026-03-20T14:46:00Z"_utc), 30.0)) EXPECT_FALSE(std::isnan(lat));
    EXPECT_GE(sun.subsolar_longitude_deg, -180.0);
    EXPECT_LT(sun.subsolar_longitude_deg, 180.0);
}

} // namespace
