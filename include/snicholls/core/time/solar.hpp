//
//  solar.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The sun's position from a snicholls::utc instant (NOAA algorithm), daylight and the terminator.
/// \ingroup core

#ifndef snicholls_core_time_solar_hpp
#define snicholls_core_time_solar_hpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include <snicholls/core/time/utc.hpp>

namespace snicholls {

/// \brief Where the sun is at an instant (NOAA "General Solar Position Calculations").
/// \ingroup core
struct solar_position
{
    double declination_deg = 0;         ///< Sun's declination: the subsolar latitude, -23.44..+23.44.
    double equation_of_time_min = 0;    ///< Apparent minus mean solar time, minutes (-14.2..+16.4).
    double subsolar_latitude_deg = 0;   ///< Latitude where the sun is overhead (= declination).
    double subsolar_longitude_deg = 0;  ///< Longitude where it is solar noon, -180..180.
};

/// \brief The sun's position at instant `t`.
/// \ingroup core
///
/// \overview
/// NOAA's solar position algorithm (the NOAA Solar Calculator spreadsheets, after Meeus, "Astronomical Algorithms"):
/// accurate to about 0.01 degree in declination and a few seconds in the equation of time for 1800..2100, far better
/// than the one-line approximations used in UIs. Drives the world clock's day/night map. UTC here is snicholls::utc (Unix
/// time); the ~1 minute between UT1 and UTC-with-leap-seconds is below this use's resolution.
///
/// \relations
/// Used by world-clock displays (day and night per city). Group: \ref core.
///
/// \impls
/// None (functions).
///
/// \contracts
/// - **Thread safety:** pure functions.
/// - **Exception safety:** noexcept (the terminator may throw `std::bad_alloc`).
/// - **Complexity:** O(1); the terminator O(points).
///
/// \errors
/// None.
///
/// \examples
/// \code
/// const snicholls::solar_position sun = snicholls::sun_position(snicholls::utc<>::now());
/// \endcode
///
/// \perf
/// Not benchmarked (a few dozen floating-point operations).
///
/// \testing
/// `tests/time/solar_test.cpp`: published 2026 equinox and solstice instants (USNO / IERS), equation-of-time
/// extremes, the subsolar longitude at solar noon, and terminator geometry.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// - Atmospheric refraction is not applied to the position; snicholls::is_daylight uses the standard -0.833 degree horizon.
///
/// \status
/// Experimental, pre-1.0.
/// \param t the instant.
/// \return the position.
template <utc_duration D>
[[nodiscard]] solar_position sun_position(const utc<D>& t) noexcept
{
    constexpr double rad = std::numbers::pi / 180.0, deg = 180.0 / std::numbers::pi;
    const double unix_days = std::chrono::duration<double, std::ratio<86400>>(t.time_since_epoch()).count();
    const double jd = unix_days + 2440587.5;
    const double jc = (jd - 2451545.0) / 36525.0;
    const double l0 = std::fmod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0);
    const double m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
    const double e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);
    const double c = std::sin(m * rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + std::sin(2 * m * rad) * (0.019993 - 0.000101 * jc) +
                     std::sin(3 * m * rad) * 0.000289;
    const double omega = 125.04 - 1934.136 * jc;
    const double app_long = l0 + c - 0.00569 - 0.00478 * std::sin(omega * rad);
    const double mean_obliq = 23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0;
    const double obliq = mean_obliq + 0.00256 * std::cos(omega * rad);
    const double decl = std::asin(std::sin(obliq * rad) * std::sin(app_long * rad)) * deg;
    const double y = std::tan(obliq * rad / 2) * std::tan(obliq * rad / 2);
    const double eot = 4 * deg *
                       (y * std::sin(2 * l0 * rad) - 2 * e * std::sin(m * rad) + 4 * e * y * std::sin(m * rad) * std::cos(2 * l0 * rad) -
                        0.5 * y * y * std::sin(4 * l0 * rad) - 1.25 * e * e * std::sin(2 * m * rad));
    const double minutes_utc = (unix_days - std::floor(unix_days)) * 1440.0;
    double lon = -(minutes_utc + eot - 720.0) / 4.0;   // where apparent solar time is 12:00
    lon = std::fmod(lon + 540.0, 360.0) - 180.0;
    return {decl, eot, decl, lon};
}

/// \brief Solar elevation in degrees at a place, for a position from snicholls::sun_position.
/// \param sun the sun's position.
/// \param latitude_deg,longitude_deg the place.
/// \return the elevation above the horizon (negative below), without refraction.
[[nodiscard]] inline double solar_elevation_deg(const solar_position& sun, double latitude_deg, double longitude_deg) noexcept
{
    constexpr double rad = std::numbers::pi / 180.0;
    const double h = (longitude_deg - sun.subsolar_longitude_deg) * rad;
    const double s = std::sin(latitude_deg * rad) * std::sin(sun.declination_deg * rad) +
                     std::cos(latitude_deg * rad) * std::cos(sun.declination_deg * rad) * std::cos(h);
    return std::asin(std::clamp(s, -1.0, 1.0)) / rad;
}

/// \brief Daylight: the sun's upper limb above the horizon with refraction (elevation > -0.833 degree).
/// \param sun the sun's position.
/// \param latitude_deg,longitude_deg the place.
/// \return true in daylight.
[[nodiscard]] inline bool is_daylight(const solar_position& sun, double latitude_deg, double longitude_deg) noexcept
{
    return solar_elevation_deg(sun, latitude_deg, longitude_deg) > -0.833;
}

/// \brief The day/night terminator: for each longitude step, the latitude where the sun's elevation is zero.
/// \param sun the sun's position.
/// \param step_deg the longitude step (e.g. 1 degree: 361 points from -180 to 180).
/// \return (longitude, latitude) pairs, degrees.
[[nodiscard]] inline std::vector<std::pair<double, double>> terminator(const solar_position& sun, double step_deg = 1.0)
{
    constexpr double rad = std::numbers::pi / 180.0;
    // At an equinox tan(decl) -> 0: nudge it so the terminator is the (correct) near-vertical line.
    const double tan_decl = std::abs(sun.declination_deg) < 1e-6 ? 1e-6 : std::tan(sun.declination_deg * rad);
    std::vector<std::pair<double, double>> points;
    for (double lon = -180.0; lon <= 180.0 + 1e-9; lon += step_deg)
        points.emplace_back(lon, std::atan(-std::cos((lon - sun.subsolar_longitude_deg) * rad) / tan_decl) / rad);
    return points;
}

} // namespace snicholls

#endif /* snicholls_core_time_solar_hpp */
