//
//  utc.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::utc<Duration>`: a UTC instant.
/// \ingroup core
///
/// snicholls::utc\<Duration> - a UTC instant: exactly one std::chrono::sys_time\<Duration>, nothing else.
///
/// \code
///   auto t = snicholls::utc<>::now();                                   // nanoseconds on every platform
///   auto q = snicholls::utc<>::from_unix_millis(1714486985123);         // market data (Massive: Unix ms)
///   snicholls::utc<> d = std::chrono::sys_days{2025y / April / 30} + 14h;   // lossless std::chrono converts implicitly
///   q.unix_millis(); q.date(); q.time_of_day(); q.weekday(); q.floor<std::chrono::days>();
///   q + 5ms;  q - t;  q < t;  std::hash<snicholls::utc<>>{}(q);
/// \endcode
///
/// Text (RFC 3339) is in time/rfc3339.hpp, std::format in time/utc_format.hpp.
///
/// Semantics:
/// - POSIX / Unix time, i.e. std::chrono::system_clock's epoch and std::chrono::sys_time: every day has 86 400 s.
///   Leap seconds are not representable (as in sys_time, time_t and every Unix-ms feed); a difference across a
///   leap second is one SI second short. This is NOT std::chrono::utc_time, which counts leap seconds.
/// - The default Duration is nanoseconds on every platform (never system_clock::duration, which is us on macOS
/// \code
///    and ns on Linux). utc<> spans 1677-09-21T00:12:43.145224192Z .. 2262-04-11T23:47:16.854775807Z;
///    utc_us spans +-292 277 years.
/// \endcode
/// - Default construction is the epoch (1970-01-01T00:00:00Z), like std::chrono::time_point. Never "now".
/// - Conversions to whole units floor (round toward the past), so instants before 1970 are correct.
/// - sizeof == sizeof(Duration::rep), trivially copyable, standard layout. Everything is constexpr and noexcept
///   except now().
///
/// **Portability.** Identical on macOS (libc++), Linux (libstdc++, GCC and Clang, x86-64 and ARM64) and Windows
/// (MSVC), and checked by the same tests on each against committed fixtures generated from Python's `datetime` and
/// exact integer arithmetic (`tests/fixtures/time/`, `scripts/generate_time_fixtures.py`), never from core's own
/// output:
/// - the ticks of every instant, at every resolution; every `from_unix_*` / `unix_*` conversion, flooring negative
///   values toward the past;
/// - the RFC 3339 text written by `to_chars` / `to_text` / `to_string` / `operator<<` / `std::format("{}")`, at the
///   resolution's own digits and at every precision 0..9; and what `parse_utc` / `parse_unix` / `parse_date`
///   accept, reject (with which `std::errc`) and return, strict and lenient. No text form reads the locale (tested
///   with an odd global and stream locale) or goes through `double`;
/// - the calendar: `civil.hpp` and `business_days.hpp` for every day of years 0001..9999;
/// - `to_timespec` / `from_timespec` (a 64-bit `time_t` is required and asserted; `tv_nsec`, a `long` that is 32-bit
///   on Windows, only holds [0, 1e9)).
///
/// Not identical, by nature:
/// - `now()` reads `std::chrono::system_clock`, whose tick is 1 us on libc++, 1 ns on libstdc++ and 100 ns on MSVC.
///   `utc<>::now()` is always nanoseconds, so on macOS its last three digits are zero; coarser resolutions floor.
///   The value is the clock's own tick, never interpolated (tested).
/// - `std::hash<utc<D>>` is the standard library's hash of the tick count: never persist or compare it across
///   builds.
/// - `std::format` with a chrono spec (`{:%F %T}`) is forwarded to the standard library's formatter. Without the
///   `L` flag it uses the C locale and is identical (tested); with `L` it follows the locale, as the standard says.
/// - `solar.hpp` uses the platform's `<cmath>`, so its results agree to the published accuracy (0.01 degree), not
///   bit for bit.

#ifndef snicholls_core_time_utc_hpp
#define snicholls_core_time_utc_hpp

#include <cassert>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <limits>
#include <ratio>
#include <type_traits>

#include <snicholls/core/time/civil.hpp>

namespace snicholls {

namespace detail {

template <typename T>
struct is_chrono_duration : std::false_type {};

template <typename Rep, typename Period>
struct is_chrono_duration<std::chrono::duration<Rep, Period>> : std::true_type {};

// k if den == 10^k (k = 0..9), otherwise -1.
constexpr int decimal_exponent(std::intmax_t den) noexcept
{
    int k = 0;
    while (den > 1 && den % 10 == 0 && k < 10) {
        den /= 10;
        ++k;
    }
    return den == 1 && k <= 9 ? k : -1;
}

// Decimal digits of a second a Period resolves: s = 0, ms = 3, us = 6, ns = 9; whole-second multiples
// (minutes, hours, days) = 0; anything else (1/3 s, 7/10 s) = -1.
template <typename Period>
inline constexpr int utc_fraction_digits =
    Period::den == 1 ? 0 : (Period::num == 1 ? decimal_exponent(Period::den) : -1);

} // namespace detail

// A Duration snicholls::utc can hold: a signed integral std::chrono::duration whose period is 1/10^k s (k = 0..9) or
// a whole number of seconds (minutes, hours, days).
template <typename D>
concept utc_duration = detail::is_chrono_duration<D>::value && std::is_integral_v<typename D::rep> &&
                       std::is_signed_v<typename D::rep> && detail::utc_fraction_digits<typename D::period> >= 0;

/// \brief A UTC instant: exactly one `std::chrono::sys_time<Duration>`, nothing else.
/// \ingroup core
///
/// \overview
/// See the file description above.
///
/// \relations
/// - Text: `time/rfc3339.hpp` (`to_chars`, `from_chars`, `parse_utc`); format: `utc_format.hpp`; calendar: `civil.hpp`, `business_days.hpp`; clocks, time zones, JSON and SQL mappings belong in the consuming library.
/// - Group: \ref core.
///
/// \impls
/// Aliases: `utc_ns` (= `utc<>`), `utc_us`, `utc_ms`, `utc_s`.
///
/// \contracts
/// - Same size and triviality as `sys_time<Duration>`; comparison and arithmetic across durations are exact.
///
/// \errors
/// None.
///
/// \examples
/// \snippet utc_demo.cpp demo
///
/// \perf
/// Release, 4096 instants in 2000..2030, median of 10, ns per value (macOS AppleClang 17 / Ubuntu clang 18; date not recorded): `snicholls::to_chars` 29 / 22 (`std::format_to_n` 1310 / 339); `snicholls::to_string` 73-85 / 33-39 (`std::format` 1492 / 361); `snicholls::from_chars` (strict, validated) 29-32 / 27 (`sscanf` + `timegm` 6674 / 461); Unix ms round trip 3.1 vs raw `std::chrono` 3.1 (zero overhead). The benchmark is benchmarks/time/utc_benchmark.cpp (not on Windows: its baselines are POSIX).
///
/// \testing
/// - Tests: tests/time/utc_test.cpp, tests/time/utc_format_test.cpp.
/// - Demo: demos/time/utc_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only .
///
/// \gotchas
/// None known.
///
/// \status
/// Stable, pre-1.0.
///
/// \tparam Duration the tick (default nanoseconds).
template <utc_duration Duration = std::chrono::nanoseconds>
class utc
{
public:
    /// \brief The duration.
    using duration = Duration;
    /// \brief The rep.
    using rep = typename Duration::rep;
    /// \brief The period.
    using period = typename Duration::period;
    /// \brief The clock.
    using clock = std::chrono::system_clock;
    /// \brief The time point.
    using time_point = std::chrono::sys_time<Duration>;

    /// \brief Fraction digits written by snicholls::to_chars: 9 for ns, 6 for us, 3 for ms, 0 for s and coarser.
    static constexpr int fraction_digits = detail::utc_fraction_digits<period>;

    // ---- construction ------------------------------------------------------------------------------------

    /// \brief Constructs it.
    constexpr utc() noexcept = default;   // the epoch

    /// \brief Constructs it.
    /// \param tp the tp.
    constexpr utc(const time_point& tp) noexcept : tp_(tp) {}   // implicit: it is the same thing

    /// \brief From any sys_time / utc that converts losslessly (exactly when std::chrono converts implicitly).
    ///
    /// For a lossy conversion use std::chrono::floor / ceil / round first.
    /// \param tp the tp.
    template <typename D2>
        requires(!std::is_same_v<D2, Duration> && std::is_convertible_v<D2, Duration>)
    constexpr utc(const std::chrono::sys_time<D2>& tp) noexcept : tp_(tp)
    {
    }

    /// \brief Constructs it.
    /// \param other the other value.
    template <utc_duration D2>
        requires(!std::is_same_v<D2, Duration> && std::is_convertible_v<D2, Duration>)
    constexpr utc(const utc<D2>& other) noexcept : tp_(other.sys_time())
    {
    }

    /// \brief The current time, floored to Duration.
    ///
    /// (macOS system_clock ticks in us, so the last 3 ns digits are 0 there).
    /// \return the current time, floored to Duration.
    [[nodiscard]] static utc now() noexcept { return utc(std::chrono::floor<Duration>(clock::now())); }

    /// \brief The epoch.
    /// \return the epoch.
    [[nodiscard]] static constexpr utc epoch() noexcept { return utc{}; }
    /// \brief The min.
    /// \return the min.
    [[nodiscard]] static constexpr utc min() noexcept { return utc(time_point(Duration::min())); }
    /// \brief The max.
    /// \return the max.
    [[nodiscard]] static constexpr utc max() noexcept { return utc(time_point(Duration::max())); }

    /// \brief True if d since the epoch, floored to Duration, is representable.
    /// \param d the d.
    /// \return `true` if d since the epoch, floored to Duration, is representable.
    template <typename Rep2, typename Period2>
        requires std::is_integral_v<Rep2> && std::is_signed_v<Rep2>
    [[nodiscard]] static constexpr bool is_representable(const std::chrono::duration<Rep2, Period2>& d) noexcept
    {
        using D2 = std::chrono::duration<Rep2, Period2>;
        if constexpr (std::ratio_less_equal_v<Period2, period> && sizeof(Rep2) <= sizeof(rep)) {
            return true;   // finer (or equal) unit, no wider rep: flooring only shrinks the magnitude
        } else if constexpr (std::ratio_less_equal_v<Period2, period>) {
            // Floor in the wider Rep2 (flooring straight into Duration would narrow before we could check).
            const auto f = std::chrono::floor<std::chrono::duration<Rep2, period>>(d);
            return f.count() >= std::numeric_limits<rep>::min() && f.count() <= std::numeric_limits<rep>::max();
        } else {
            constexpr D2 hi = std::chrono::floor<D2>(Duration::max());
            constexpr D2 lo = std::chrono::ceil<D2>(Duration::min());
            return d >= lo && d <= hi;
        }
    }

    /// \brief d since the Unix epoch, floored to Duration.
    ///
    /// Precondition: is_representable(d).
    /// \param d the d.
    /// \return the result.
    template <typename Rep2, typename Period2>
        requires std::is_integral_v<Rep2> && std::is_signed_v<Rep2>
    [[nodiscard]] static constexpr utc from_unix(const std::chrono::duration<Rep2, Period2>& d) noexcept
    {
        assert(is_representable(d) && "snicholls::utc::from_unix: outside the range of Duration");
        return utc(time_point(std::chrono::floor<Duration>(d)));
    }

    /// \brief Converts from unix seconds.
    /// \param s the string.
    /// \return the result.
    [[nodiscard]] static constexpr utc from_unix_seconds(std::int64_t s) noexcept { return from_unix(std::chrono::duration<std::int64_t>(s)); }
    /// \brief Converts from unix millis.
    /// \param ms the ms.
    /// \return the result.
    [[nodiscard]] static constexpr utc from_unix_millis(std::int64_t ms) noexcept { return from_unix(std::chrono::duration<std::int64_t, std::milli>(ms)); }
    /// \brief Converts from unix micros.
    /// \param us the us.
    /// \return the result.
    [[nodiscard]] static constexpr utc from_unix_micros(std::int64_t us) noexcept { return from_unix(std::chrono::duration<std::int64_t, std::micro>(us)); }
    /// \brief Converts from unix nanos.
    /// \param ns the ns.
    /// \return the result.
    [[nodiscard]] static constexpr utc from_unix_nanos(std::int64_t ns) noexcept { return from_unix(std::chrono::duration<std::int64_t, std::nano>(ns)); }

    /// \brief POSIX struct timespec (tv_nsec in [0, 1e9)), floored to Duration.
    ///
    /// Precondition: the instant is representable. Every `to_timespec()` result is, including those of `min()`.
    /// \param ts the ts.
    /// \return the result.
    [[nodiscard]] static constexpr utc from_timespec(const std::timespec& ts) noexcept
    {
        const auto s = static_cast<std::int64_t>(ts.tv_sec);
        const auto ns = static_cast<std::int64_t>(ts.tv_nsec);
        if constexpr (std::ratio_less_equal_v<period, std::ratio<1>>) {
            // A negative second with a fraction is (s + 1) s minus the rest, so an instant in the last second
            // above min() never passes through the whole second s, which Duration cannot hold (found by the
            // portability tests: utc<>::min().to_timespec() did not round-trip).
            if (s < 0 && ns > 0)
                return from_unix(std::chrono::duration<std::int64_t>(s + 1)) -
                       std::chrono::ceil<Duration>(std::chrono::nanoseconds(1'000'000'000 - ns));
        }
        return from_unix(std::chrono::duration<std::int64_t>(s)) + std::chrono::floor<Duration>(std::chrono::nanoseconds(ns));
    }

    // ---- std::chrono interop ------------------------------------------------------------------------------

    /// \brief The sys time.
    /// \return the sys time.
    [[nodiscard]] constexpr time_point sys_time() const noexcept { return tp_; }
    /// \brief Converts to `time_point`.
    /// \return the result.
    constexpr operator time_point() const noexcept { return tp_; }   // implicit, like the constructor
    /// \brief The time since epoch.
    /// \return the time since epoch.
    [[nodiscard]] constexpr Duration time_since_epoch() const noexcept { return tp_.time_since_epoch(); }

    // ---- Unix numbers (floored; precondition: the result fits in int64 - always true for seconds..us) -------

    /// \brief Converts to unix.
    /// \return the result.
    template <typename D2>
    [[nodiscard]] constexpr D2 to_unix() const noexcept
    {
        return std::chrono::floor<D2>(tp_.time_since_epoch());
    }
    /// \brief The unix seconds.
    /// \return the unix seconds.
    [[nodiscard]] constexpr std::int64_t unix_seconds() const noexcept { return to_unix<std::chrono::duration<std::int64_t>>().count(); }
    /// \brief The unix millis.
    /// \return the unix millis.
    [[nodiscard]] constexpr std::int64_t unix_millis() const noexcept { return to_unix<std::chrono::duration<std::int64_t, std::milli>>().count(); }
    /// \brief The unix micros.
    /// \return the unix micros.
    [[nodiscard]] constexpr std::int64_t unix_micros() const noexcept { return to_unix<std::chrono::duration<std::int64_t, std::micro>>().count(); }
    /// \brief The unix nanos.
    /// \return the unix nanos.
    [[nodiscard]] constexpr std::int64_t unix_nanos() const noexcept { return to_unix<std::chrono::duration<std::int64_t, std::nano>>().count(); }

    /// \brief Converts to timespec.
    /// \return the result.
    [[nodiscard]] constexpr std::timespec to_timespec() const noexcept
    {
        const auto s = std::chrono::floor<std::chrono::seconds>(tp_);
        std::timespec ts{};
        ts.tv_sec = static_cast<std::time_t>(s.time_since_epoch().count());
        ts.tv_nsec = static_cast<long>(std::chrono::duration_cast<std::chrono::nanoseconds>(tp_ - s).count());
        return ts;
    }

    // ---- calendar (std::chrono types; year_month_day needs |year| <= 32767, always true for utc<>) ----------

    /// \brief The days.
    /// \return the days.
    [[nodiscard]] constexpr std::chrono::sys_days days() const noexcept { return std::chrono::floor<std::chrono::days>(tp_); }
    /// \brief The date.
    /// \return the date.
    [[nodiscard]] constexpr std::chrono::year_month_day date() const noexcept { return std::chrono::year_month_day(days()); }
    /// \brief The weekday.
    /// \return the weekday.
    [[nodiscard]] constexpr std::chrono::weekday weekday() const noexcept { return std::chrono::weekday(days()); }
    /// \brief The time of day.
    /// \return the time of day.
    [[nodiscard]] constexpr std::chrono::hh_mm_ss<Duration> time_of_day() const noexcept
    {
        return std::chrono::hh_mm_ss<Duration>(std::chrono::duration_cast<Duration>(tp_ - days()));
    }

    /// \brief Rounded to a multiple of D2 (e.g.
    ///
    /// std::chrono::days for midnight), keeping this type.
    /// \return the result.
    template <typename D2>
    [[nodiscard]] constexpr utc floor() const noexcept { return from_rounded(std::chrono::floor<D2>(tp_)); }
    /// \brief Rounded to a multiple of D2 (e.g.
    ///
    /// std::chrono::days for midnight), keeping this type.
    /// \return the result.
    template <typename D2>
    [[nodiscard]] constexpr utc ceil() const noexcept { return from_rounded(std::chrono::ceil<D2>(tp_)); }
    /// \brief Rounded to a multiple of D2 (e.g.
    ///
    /// std::chrono::days for midnight), keeping this type.
    /// \return the result.
    template <typename D2>
    [[nodiscard]] constexpr utc round() const noexcept { return from_rounded(std::chrono::round<D2>(tp_)); }

    // ---- arithmetic (like std::chrono::time_point) ---------------------------------------------------------

    /// \brief Adds in place.
    /// \param d the d.
    /// \return the result.
    template <typename Rep2, typename Period2>
        requires std::is_convertible_v<std::chrono::duration<Rep2, Period2>, Duration>
    constexpr utc& operator+=(const std::chrono::duration<Rep2, Period2>& d) noexcept
    {
        tp_ += d;
        return *this;
    }

    /// \brief Subtracts in place.
    /// \param d the d.
    /// \return the result.
    template <typename Rep2, typename Period2>
        requires std::is_convertible_v<std::chrono::duration<Rep2, Period2>, Duration>
    constexpr utc& operator-=(const std::chrono::duration<Rep2, Period2>& d) noexcept
    {
        tp_ -= d;
        return *this;
    }

    // ---- comparison --------------------------------------------------------------------------------------

    /// \brief Equality.
    /// \param a the left operand.
    /// \param b the right operand.
    /// \return the result.
    friend constexpr bool operator==(const utc& a, const utc& b) noexcept
    {
        return a.tp_.time_since_epoch().count() == b.tp_.time_since_epoch().count();
    }
    /// \brief Three-way comparison.
    /// \param a the left operand.
    /// \param b the right operand.
    /// \return the result.
    friend constexpr std::strong_ordering operator<=>(const utc& a, const utc& b) noexcept
    {
        return a.tp_.time_since_epoch().count() <=> b.tp_.time_since_epoch().count();
    }

private:
    template <typename D2>
    static constexpr utc from_rounded(const std::chrono::sys_time<D2>& r) noexcept
    {
        return utc(time_point(std::chrono::duration_cast<Duration>(r.time_since_epoch())));
    }

    time_point tp_{};
};

/// \brief The utc ns.
using utc_ns = utc<std::chrono::nanoseconds>;
/// \brief The utc us.
using utc_us = utc<std::chrono::microseconds>;
/// \brief The utc ms.
using utc_ms = utc<std::chrono::milliseconds>;
/// \brief The utc s.
using utc_s = utc<std::chrono::seconds>;

/// \brief utc + duration -> utc of the common duration.
///
/// (utc\<ms> + 1ns is a utc\<ns>), as for std::chrono::time_point.
/// \ingroup core
/// \param t the t.
/// \param d the d.
/// \return the result.
template <utc_duration D, typename Rep2, typename Period2>
    requires utc_duration<std::common_type_t<D, std::chrono::duration<Rep2, Period2>>>
constexpr auto operator+(const utc<D>& t, const std::chrono::duration<Rep2, Period2>& d) noexcept
{
    return utc<std::common_type_t<D, std::chrono::duration<Rep2, Period2>>>(t.sys_time() + d);
}

/// \brief Addition.
/// \param d the d.
/// \param t the t.
/// \return the result.
template <typename Rep2, typename Period2, utc_duration D>
    requires utc_duration<std::common_type_t<D, std::chrono::duration<Rep2, Period2>>>
constexpr auto operator+(const std::chrono::duration<Rep2, Period2>& d, const utc<D>& t) noexcept
{
    return t + d;
}

/// \brief Subtraction.
/// \param t the t.
/// \param d the d.
/// \return the result.
template <utc_duration D, typename Rep2, typename Period2>
    requires utc_duration<std::common_type_t<D, std::chrono::duration<Rep2, Period2>>>
constexpr auto operator-(const utc<D>& t, const std::chrono::duration<Rep2, Period2>& d) noexcept
{
    return utc<std::common_type_t<D, std::chrono::duration<Rep2, Period2>>>(t.sys_time() - d);
}

/// \brief The signed distance a - b in the common duration.
/// \ingroup core
/// \param a the left operand.
/// \param b the right operand.
/// \return the signed distance a - b in the common duration.
template <utc_duration D1, utc_duration D2>
constexpr auto operator-(const utc<D1>& a, const utc<D2>& b) noexcept
{
    return a.sys_time() - b.sys_time();
}

/// \brief Across resolutions: compared exactly, in the common duration (as std::chrono::time_point).
/// \ingroup core
/// \param a the left operand.
/// \param b the right operand.
/// \return the result.
template <utc_duration D1, utc_duration D2>
    requires(!std::is_same_v<D1, D2>)
constexpr bool operator==(const utc<D1>& a, const utc<D2>& b) noexcept
{
    return a.sys_time() == b.sys_time();
}

/// \brief Three-way comparison.
/// \param a the left operand.
/// \param b the right operand.
/// \return the result.
template <utc_duration D1, utc_duration D2>
    requires(!std::is_same_v<D1, D2>)
constexpr std::strong_ordering operator<=>(const utc<D1>& a, const utc<D2>& b) noexcept
{
    using C = std::common_type_t<D1, D2>;
    return std::chrono::duration_cast<C>(a.time_since_epoch()).count() <=>
           std::chrono::duration_cast<C>(b.time_since_epoch()).count();
}

} // namespace snicholls

// \brief Hashes the tick count.
//
// Like every std::hash, not portable across standard libraries: never persist it.
/// \cond
template <snicholls::utc_duration D>
struct std::hash<snicholls::utc<D>>
{
    /// \brief Invokes it.
    /// \param t the t.
    /// \return the operator.
    std::size_t operator()(const snicholls::utc<D>& t) const noexcept
    {
        return std::hash<typename D::rep>{}(t.time_since_epoch().count());
    }
};
/// \endcond

#endif /* snicholls_core_time_utc_hpp */
