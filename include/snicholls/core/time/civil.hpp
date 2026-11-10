//
//  civil.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Proleptic Gregorian calendar arithmetic on plain integers.
/// \ingroup core
///
/// Proleptic Gregorian calendar arithmetic on plain integers (Howard Hinnant, "chrono-Compatible Low-Level Date
/// Algorithms", http://howardhinnant.github.io/date_algorithms.html):
///
/// \code
///   snicholls::days_from_civil(2024, 2, 29)   == 19782        // days since 1970-01-01
///   snicholls::civil_from_days(19782)         == {2024, 2, 29}
///   snicholls::weekday_from_days(19782)       == 4             // 0 = Sunday ... 6 = Saturday (C encoding)
/// \endcode
///
/// The same algorithms std::chrono::year_month_day uses, without its range (year -32767..32767) or validity
/// checks: constexpr, noexcept, branch-light, exact for every day count |z| < 2^60 (any snicholls::utc is within
/// +-2^37 days). No validation: days_from_civil(2025, 2, 30) is 2025-03-02 - check with
/// snicholls::last_day_of_month first when the input is untrusted. Year 0 is 1 BC (ISO 8601 / astronomical).

#ifndef snicholls_core_time_civil_hpp
#define snicholls_core_time_civil_hpp

#include <cstdint>

namespace snicholls {

/// \brief A proleptic Gregorian date as plain integers (year, month, day).
/// \ingroup core
///
/// \overview
/// See the file description above.
///
/// \relations
/// - Conversions `snicholls::days_from_civil`, `snicholls::civil_from_days`; used by `snicholls::utc` text.
/// - Group: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - `constexpr`, `noexcept`, O(1).
///
/// \errors
/// None.
///
/// \examples
/// \snippet civil_demo.cpp demo
///
/// \perf
/// `snicholls::civil_from_days` 8.8 ns (macOS) / 8.9 ns (Ubuntu) vs `std::year_month_day` 9.3 / 5.3 ns.
///
/// \testing
/// - Tests: tests/time/civil_test.cpp.
/// - Demo: demos/time/civil_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only .
///
/// \gotchas
/// None known.
///
/// \status
/// Stable, pre-1.0.
struct civil_date
{
    /// \brief The year.
    std::int64_t year{1970};
    /// \brief The month.
    unsigned month{1};
    /// \brief The day.
    unsigned day{1};

    /// \brief Equality.
    /// \return the result.
    friend constexpr bool operator==(const civil_date&, const civil_date&) noexcept = default;
};

/// \brief Whether the value is leap year.
/// \param y the y.
/// \return `true` if the value is leap year.
constexpr bool is_leap_year(std::int64_t y) noexcept
{
    return y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
}

/// \brief Last day of month m (1..12) of year y: 28..31.
/// \ingroup core
/// \param y the y.
/// \param m the m.
/// \return the result.
constexpr unsigned last_day_of_month(std::int64_t y, unsigned m) noexcept
{
    return m == 2 ? (is_leap_year(y) ? 29U : 28U) : (m == 4 || m == 6 || m == 9 || m == 11) ? 30U : 31U;
}

/// \brief Days since 1970-01-01 of y-m-d (m 1..12, d 1..31; not validated).
/// \ingroup core
/// \param y the y.
/// \param m the m.
/// \param d the d.
/// \return the result.
constexpr std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) noexcept
{
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const auto yoe = static_cast<unsigned>(y - era * 400);                     // [0, 399]
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;       // [0, 365]
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;                 // [0, 146096]
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

/// \brief The civil date of z days since 1970-01-01.
/// \ingroup core
/// \param z the z.
/// \return the civil date of z days since 1970-01-01.
constexpr civil_date civil_from_days(std::int64_t z) noexcept
{
    z += 719468;
    const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const auto doe = static_cast<unsigned>(z - era * 146097);                       // [0, 146096]
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;      // [0, 399]
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                    // [0, 365]
    const unsigned mp = (5 * doy + 2) / 153;                                         // [0, 11]
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;                                 // [1, 31]
    const unsigned m = mp < 10 ? mp + 3 : mp - 9;                                    // [1, 12]
    return {static_cast<std::int64_t>(yoe) + era * 400 + (m <= 2), m, d};
}

/// \brief civil_from_days for the years 0000..9999 only, in 32-bit unsigned arithmetic.
///
/// The same algorithm shifted by one 400-year era so nothing is negative (no era branch, no 64-bit division).
///
/// Used by the RFC 3339 formatter. Precondition: the year is in [0, 9999] (checked exhaustively in the tests). Neri-Schneider's algorithm (the one libstdc++ uses) was measured too: faster on Ubuntu/libstdc++, but 3x slower inside to_chars with AppleClang/arm64, so it is not used.
/// \ingroup core
/// \param z the z.
/// \return the result.
constexpr civil_date civil_from_days_fast(std::int64_t z) noexcept
{
    const auto u = static_cast<std::uint32_t>(z + 719468 + 146097);
    const std::uint32_t era = u / 146097;
    const std::uint32_t doe = u - era * 146097;
    const std::uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const std::uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const std::uint32_t mp = (5 * doy + 2) / 153;
    const std::uint32_t d = doy - (153 * mp + 2) / 5 + 1;
    const std::uint32_t m = mp < 10 ? mp + 3 : mp - 9;
    return {static_cast<std::int64_t>(yoe + era * 400 + (m <= 2)) - 400, m, d};
}

/// \brief Day of the week of z days since 1970-01-01 (a Thursday): 0 = Sunday ...
///
/// 6 = Saturday.
/// \ingroup core
/// \param z the z.
/// \return the result.
constexpr unsigned weekday_from_days(std::int64_t z) noexcept
{
    return static_cast<unsigned>(z >= -4 ? (z + 4) % 7 : (z + 5) % 7 + 6);
}

} // namespace snicholls

#endif /* snicholls_core_time_civil_hpp */
