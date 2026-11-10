//
//  rfc3339.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief RFC 3339 text for `snicholls::utc`: `to_chars` / `from_chars` style, exact round trips.
/// \ingroup core
///
/// RFC 3339 text for snicholls::utc - std::to_chars / std::from_chars style: no allocation, no locale, no exceptions,
/// constexpr, exact round trips.
///
/// \code
///   char buf[snicholls::max_chars_v<std::chrono::nanoseconds>];
///   auto [end, ec] = snicholls::to_chars(buf, buf + sizeof buf, t);        // "2024-04-30T14:23:05.123456789Z"
///   auto text = snicholls::to_text(t);  text.view();                      // the same, in a returned buffer
///   snicholls::to_string(t, 3);                                           // "2024-04-30T14:23:05.123Z" (allocates)
/// \endcode
///
/// \code
///   snicholls::utc<> t;
///   auto [ptr, ec] = snicholls::from_chars(first, last, t);               // a prefix, like std::from_chars
///   auto r = snicholls::parse_utc("2024-04-30T10:23:05-04:00");          // the whole string; r.has_value(), *r
///   auto m = snicholls::parse_unix<std::chrono::milliseconds>("1714486985123");   // exact, no double
///   auto d = snicholls::parse_date("2024-04-30");                        // std::chrono::year_month_day
///   using namespace snicholls::literals;  constexpr auto e = "2024-04-30T14:23:05Z"_utc;   // checked at compile time
/// \endcode
///
/// Output: YYYY-MM-DDTHH:MM:SS[.F]Z with exactly Duration's fraction digits (ns 9, us 6, ms 3, s 0) unless a
/// precision 0..9 is given (floored, or zero-padded). Years 0000..9999 only: outside, to_chars fails with
/// errc::value_too_large (never for utc<>, whose whole range is inside).
///
/// Input (strict by default): YYYY-MM-DD (T|t|' ') HH:MM:SS [.1*DIGIT] (Z|z|+HH:MM|-HH:MM).
/// \code
///   errc::invalid_argument      not this syntax, an invalid date/time (2025-02-29, 24:00:00, offset +24:00), or a
///                               second 60 that is not 23:59:60 UTC after applying the offset;
///   errc::result_out_of_range   outside Duration's range, more fraction digits than Duration holds (non-zero ones,
///                               under precision_policy::reject), or a leap second under leap_second_policy::reject.
/// \endcode
/// On failure the output is unchanged and ptr == first, as for std::from_chars.

#ifndef snicholls_core_time_rfc3339_hpp
#define snicholls_core_time_rfc3339_hpp

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include <snicholls/core/time/civil.hpp>
#include <snicholls/core/time/utc.hpp>
#include <snicholls/core/reflect/string_access.hpp>
#include <snicholls/core/reflect/string_traits.hpp>

namespace snicholls {

// ---- options and results ----------------------------------------------------------------------------------

/// \brief Fraction digits finer than the target Duration (e.g.
///
/// ".123456789" into utc_ms).
/// \ingroup core
enum class precision_policy : std::uint8_t {
    reject,   ///< non-zero excess digits are errc::result_out_of_range (zeros are always fine)
    truncate,   ///< floor to Duration
};

/// \brief A leap second, 23:59:60 UTC: not representable in Unix time.
/// \ingroup core
enum class leap_second_policy : std::uint8_t {
    reject,   ///< errc::result_out_of_range
    clamp,   ///< the last representable instant before the insertion (23:59:59.999999999 for ns), which is what
              // std::chrono::utc_clock::to_sys returns
};

/// \brief The parse options.
struct parse_options
{
    /// \brief The precision.
    precision_policy precision = precision_policy::reject;
    /// \brief The leap second.
    leap_second_policy leap_second = leap_second_policy::reject;
    bool require_offset = true;   ///< false: a missing Z / offset means UTC
    bool basic_offset = false;   ///< true: also accept +HHMM (ISO 8601 basic) besides +HH:MM

    /// \brief Everything accepted: truncate, clamp, offset optional, basic offsets.
    /// \return the result.
    static constexpr parse_options lenient() noexcept
    {
        return {precision_policy::truncate, leap_second_policy::clamp, false, true};
    }
};

/// \brief Thrown only by parse_result::value() on an empty result.
/// \ingroup core
struct utc_error : std::runtime_error
{
    /// \brief Constructs it.
    /// \param ec the error code.
    /// \param what the what.
    utc_error(std::errc ec, const char* what) : std::runtime_error(what), code_(ec) {}
    /// \brief The error code.
    /// \return the error code.
    [[nodiscard]] std::errc code() const noexcept { return code_; }

private:
    std::errc code_;
};

namespace detail {

[[noreturn]] inline void throw_utc_error(std::errc ec)
{
    throw utc_error(ec, ec == std::errc::result_out_of_range
                            ? "snicholls::parse_result::value: value out of range (or excess precision / leap second)"
                            : "snicholls::parse_result::value: not a valid RFC 3339 / ISO 8601 value");
}

} // namespace detail

/// \brief A value or a std::errc.
///
/// The exception-free result of snicholls::parse_* (std::expected\<T, std::errc> is not available on both platforms yet).
/// \ingroup core
template <typename T>
class parse_result
{
public:
    /// \brief Constructs it.
    /// \param value the value.
    constexpr parse_result(const T& value) noexcept : value_(value) {}
    /// \brief Constructs it.
    /// \param ec the error code.
    constexpr parse_result(std::errc ec) noexcept : ec_(ec) {}   // precondition: ec != std::errc{}

    /// \brief Whether the value has value.
    /// \return `true` if the value has value.
    [[nodiscard]] constexpr bool has_value() const noexcept { return ec_ == std::errc{}; }
    /// \brief Whether it holds a value.
    /// \return the result.
    constexpr explicit operator bool() const noexcept { return has_value(); }
    /// \brief The error.
    /// \return the error.
    [[nodiscard]] constexpr std::errc error() const noexcept { return ec_; }

    /// \brief Dereference.
    /// \return the result.
    constexpr const T& operator*() const noexcept { return value_; }   // precondition: has_value()
    /// \brief Member access.
    /// \return the result.
    constexpr const T* operator->() const noexcept { return &value_; }
    /// \brief The value.
    /// \return the value.
    [[nodiscard]] constexpr const T& value() const
    {
        if (!has_value()) [[unlikely]]
            detail::throw_utc_error(ec_);
        return value_;
    }
    /// \brief The value or.
    /// \param fallback the fallback.
    /// \return the value or.
    [[nodiscard]] constexpr T value_or(const T& fallback) const noexcept { return has_value() ? value_ : fallback; }

private:
    T value_{};
    std::errc ec_{};
};

// ---- sizes ------------------------------------------------------------------------------------------------

/// \brief Characters written for `digits` fraction digits.
///
/// 20 ("YYYY-MM-DDTHH:MM:SSZ") plus '.' and the digits.
/// \ingroup core
/// \param digits the digits.
/// \return the result.
constexpr std::size_t rfc3339_length(int digits) noexcept
{
    return 20 + (digits > 0 ? static_cast<std::size_t>(digits) + 1 : 0);
}

/// \brief Buffer size that always suffices for snicholls::to_chars of a utc\<D> at its own precision.
/// \ingroup core
/// \param fraction_digits the fraction digits.
/// \return the result.
template <utc_duration D>
inline constexpr std::size_t max_chars_v = rfc3339_length(utc<D>::fraction_digits);

/// \brief ...
///
/// and at any precision (0..9).
/// \ingroup core
inline constexpr std::size_t rfc3339_max_chars = 30;

namespace detail {

constexpr std::array<char, 200> make_digit_pairs() noexcept
{
    std::array<char, 200> a{};
    for (int i = 0; i < 100; ++i) {
        a[static_cast<std::size_t>(2 * i)] = static_cast<char>('0' + i / 10);
        a[static_cast<std::size_t>(2 * i + 1)] = static_cast<char>('0' + i % 10);
    }
    return a;
}

inline constexpr std::array<char, 200> digit_pairs = make_digit_pairs();

constexpr void put2(char* p, unsigned v) noexcept   // v < 100
{
    p[0] = digit_pairs[2 * v];
    p[1] = digit_pairs[2 * v + 1];
}

inline constexpr std::array<std::int64_t, 10> pow10 = {1,       10,       100,       1000,       10000,
                                                         100000,  1000000,  10000000,  100000000,  1000000000};

constexpr std::int64_t floor_div(std::int64_t a, std::int64_t b) noexcept   // b > 0
{
    const std::int64_t q = a / b;
    return q - (a % b < 0);
}

// Seconds since the epoch of 0000-01-01T00:00:00 and 9999-12-31T23:59:59: the RFC 3339 range.
inline constexpr std::int64_t rfc3339_min_seconds = days_from_civil(0, 1, 1) * 86400;
inline constexpr std::int64_t rfc3339_max_seconds = days_from_civil(10000, 1, 1) * 86400 - 1;

// An instant as whole seconds (floored) + nanoseconds in [0, 1e9), with the fraction's excess-precision flag.
struct split_instant
{
    std::int64_t seconds;
    std::int64_t nanos;
    bool excess;   // non-zero digits finer than 1 ns
};

// (seconds, nanos) -> ticks of D. Errors: excess precision under reject, or out of D's range.
template <utc_duration D>
constexpr std::errc to_ticks(split_instant s, precision_policy policy, typename D::rep& out) noexcept
{
    using rep = typename D::rep;
    using period = typename D::period;
    std::int64_t ticks = 0;
    bool excess = s.excess;
    if constexpr (period::den > 1) {   // 1/10^k s
        constexpr std::int64_t den = period::den;
        constexpr std::int64_t ns_per_tick = 1000000000 / den;
        excess |= s.nanos % ns_per_tick != 0;
        if (excess && policy == precision_policy::reject) return std::errc::result_out_of_range;
        constexpr std::int64_t max = std::numeric_limits<std::int64_t>::max();
        constexpr std::int64_t min = std::numeric_limits<std::int64_t>::min();
        constexpr std::int64_t hi = max / den;
        constexpr std::int64_t lo = min / den;   // truncated toward 0: lo - 1 may still hold a valid instant
        if (s.seconds > hi || s.seconds < lo - 1) return std::errc::result_out_of_range;
        const std::int64_t sub = s.nanos / ns_per_tick;   // [0, den)
        if (s.seconds >= 0) {
            ticks = s.seconds * den;
            if (sub > max - ticks) return std::errc::result_out_of_range;
            ticks += sub;
        } else {   // (seconds + 1) * den - (den - sub), never overflowing on the way
            ticks = (s.seconds + 1) * den;
            const std::int64_t back = den - sub;
            if (ticks < min + back) return std::errc::result_out_of_range;
            ticks -= back;
        }
    } else {   // whole seconds (num >= 1)
        constexpr std::int64_t num = period::num;
        const std::int64_t q = floor_div(s.seconds, num);
        excess |= s.nanos != 0 || s.seconds - q * num != 0;
        if (excess && policy == precision_policy::reject) return std::errc::result_out_of_range;
        ticks = q;
    }
    if constexpr (sizeof(rep) < sizeof(std::int64_t)) {
        if (ticks < static_cast<std::int64_t>(std::numeric_limits<rep>::min()) ||
            ticks > static_cast<std::int64_t>(std::numeric_limits<rep>::max()))
            return std::errc::result_out_of_range;
    }
    out = static_cast<rep>(ticks);
    return std::errc{};
}

constexpr bool is_digit(char c) noexcept { return static_cast<unsigned char>(c - '0') <= 9; }

// days_from_civil for y in 0..9999 in 32-bit unsigned arithmetic (no negative-era branch): years are shifted by
// one 400-year era so y - 1 never goes below zero, then the era's 146097 days are taken back off.
constexpr std::int64_t days_from_civil_0000_9999(unsigned y, unsigned m, unsigned d) noexcept
{
    y += 400 - (m <= 2);
    const unsigned era = y / 400;
    const unsigned yoe = y - era * 400;
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<std::int64_t>(era * 146097 + doe) - 719468 - 146097;
}
constexpr unsigned digit(char c) noexcept { return static_cast<unsigned>(c - '0'); }
constexpr unsigned two(const char* p) noexcept { return digit(p[0]) * 10 + digit(p[1]); }

// "YYYY-MM-DD" at p (at least 10 chars available). Syntax and validity.
constexpr bool read_date(const char* p, unsigned& y, unsigned& m, unsigned& d) noexcept
{
    if (!(is_digit(p[0]) && is_digit(p[1]) && is_digit(p[2]) && is_digit(p[3]) && p[4] == '-' &&
          is_digit(p[5]) && is_digit(p[6]) && p[7] == '-' && is_digit(p[8]) && is_digit(p[9])))
        return false;
    y = two(p) * 100 + two(p + 2);
    m = two(p + 5);
    d = two(p + 8);
    if (m - 1 > 11 || d - 1 > 30) return false;   // unsigned: also rejects 0
    if (d <= 28) return true;
    const bool leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);   // 32-bit, y in 0..9999
    return d <= (m == 2 ? 28U + leap : (m == 4 || m == 6 || m == 9 || m == 11) ? 30U : 31U);
}

} // namespace detail

// ---- formatting -------------------------------------------------------------------------------------------

/// \brief Writes `t` as RFC 3339 text into `[first, last)` with `precision` fraction digits.
/// \ingroup core
///
/// \overview
/// RFC 3339 text for `snicholls::utc` in the style of `std::to_chars` / `std::from_chars`: no allocation, no locale, no exceptions, `constexpr`, exact round trips. See the file description above.
///
/// \relations
/// - Reverse: `snicholls::from_chars`, `snicholls::parse_utc`; strings: `snicholls::to_text`, `snicholls::to_string`; options `snicholls::parse_options`.
/// - Group: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - `noexcept`, no allocation; output length `snicholls::rfc3339_length(digits)`.
///
/// \errors
/// Returns `std::to_chars_result` with `errc::value_too_large` if the buffer is too small; parsing reports through `snicholls::parse_result`.
///
/// \examples
/// \snippet rfc3339_demo.cpp demo
///
/// \perf
/// Release, 4096 instants in 2000..2030, median of 10, ns per value (macOS AppleClang 17 / Ubuntu clang 18; date not recorded): `snicholls::to_chars` 29 / 22 (`std::format_to_n` 1310 / 339); `snicholls::to_string` 73-85 / 33-39 (`std::format` 1492 / 361); `snicholls::from_chars` (strict, validated) 29-32 / 27 (`sscanf` + `timegm` 6674 / 461); Unix ms round trip 3.1 vs raw `std::chrono` 3.1 (zero overhead). The benchmark is benchmarks/time/utc_benchmark.cpp (not on Windows: its baselines are POSIX).
///
/// \testing
/// - Tests: tests/time/rfc3339_test.cpp.
/// - Demo: demos/time/rfc3339_demo.cpp.
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
/// \tparam D the duration.
/// \param first the buffer start.
/// \param last the buffer end.
/// \param t the instant.
/// \param precision fraction digits (0-9).
/// \return the end of the text, or an error.
template <utc_duration D>
constexpr std::to_chars_result to_chars(char* first, char* last, const utc<D>& t, int precision) noexcept
{
    using period = typename D::period;
    constexpr int digits = utc<D>::fraction_digits;
    if (precision < 0 || precision > 9) return {last, std::errc::invalid_argument};

    const std::int64_t count = static_cast<std::int64_t>(t.time_since_epoch().count());
    std::int64_t seconds = 0;
    std::int64_t sub = 0;   // ticks of D below one second (0 for whole-second D)
    if constexpr (period::den > 1) {
        seconds = detail::floor_div(count, period::den);
        sub = count - seconds * period::den;
    } else {
        constexpr std::int64_t num = period::num;
        constexpr std::int64_t lo = -(-detail::rfc3339_min_seconds / num);   // ceil(min / num), min < 0
        constexpr std::int64_t hi = detail::rfc3339_max_seconds / num;
        if (count < lo || count > hi) return {last, std::errc::value_too_large};
        seconds = count * num;
    }
    if (seconds < detail::rfc3339_min_seconds || seconds > detail::rfc3339_max_seconds)
        return {last, std::errc::value_too_large};

    const std::size_t n = rfc3339_length(precision);
    if (static_cast<std::size_t>(last - first) < n) return {last, std::errc::value_too_large};

    const std::int64_t days = detail::floor_div(seconds, 86400);
    const auto sod = static_cast<unsigned>(seconds - days * 86400);
    const civil_date c = civil_from_days_fast(days);   // years 0000..9999, checked above
    const auto y = static_cast<unsigned>(c.year);

    char* p = first;
    detail::put2(p, y / 100);
    detail::put2(p + 2, y % 100);
    p[4] = '-';
    detail::put2(p + 5, c.month);
    p[7] = '-';
    detail::put2(p + 8, c.day);
    p[10] = 'T';
    detail::put2(p + 11, sod / 3600);
    p[13] = ':';
    detail::put2(p + 14, sod / 60 % 60);
    p[16] = ':';
    detail::put2(p + 17, sod % 60);
    p += 19;
    if (precision > 0) {
        std::int64_t frac = precision <= digits ? sub / detail::pow10[static_cast<std::size_t>(digits - precision)]
                                                : sub * detail::pow10[static_cast<std::size_t>(precision - digits)];
        *p = '.';
        char* q = p + precision;   // last digit
        int left = precision;
        for (; left >= 2; left -= 2, q -= 2) {
            detail::put2(q - 1, static_cast<unsigned>(frac % 100));
            frac /= 100;
        }
        if (left == 1) *q = static_cast<char>('0' + frac % 10);
        p += precision + 1;
    }
    *p++ = 'Z';
    return {p, std::errc{}};
}

/// \brief At Duration's own precision.
/// \ingroup core
/// \param first the first.
/// \param last the last.
/// \param t the t.
/// \return the result.
template <utc_duration D>
constexpr std::to_chars_result to_chars(char* first, char* last, const utc<D>& t) noexcept
{
    return to_chars(first, last, t, utc<D>::fraction_digits);
}

/// \brief "YYYY-MM-DD".
///
/// Errors: errc::invalid_argument if !d.ok(), errc::value_too_large if the year is outside 0000..9999 or the buffer is shorter than 10.
/// \ingroup core
/// \param first the first.
/// \param last the last.
/// \param d the d.
/// \return the result.
constexpr std::to_chars_result to_chars(char* first, char* last, const std::chrono::year_month_day& d) noexcept
{
    if (!d.ok()) return {last, std::errc::invalid_argument};
    const int y = static_cast<int>(d.year());
    if (y < 0 || y > 9999 || last - first < 10) return {last, std::errc::value_too_large};
    const auto uy = static_cast<unsigned>(y);
    detail::put2(first, uy / 100);
    detail::put2(first + 2, uy % 100);
    first[4] = '-';
    detail::put2(first + 5, static_cast<unsigned>(d.month()));
    first[7] = '-';
    detail::put2(first + 8, static_cast<unsigned>(d.day()));
    return {first + 10, std::errc{}};
}

/// \brief A formatted value in a returned buffer: no allocation.
///
/// Empty (and ec set) if formatting failed.
/// \ingroup core
struct rfc3339_text
{
    /// \brief + "+HH:MM" - "Z": text with an offset fits too
    std::array<char, rfc3339_max_chars + 5> buffer{};
    /// \brief The number of elements.
    std::uint8_t size = 0;
    /// \brief The ec.
    std::errc ec{};

    /// \brief The view.
    /// \return the view.
    [[nodiscard]] constexpr std::string_view view() const noexcept { return {buffer.data(), size}; }
    /// \brief The text as a view.
    /// \return the view.
    constexpr operator std::string_view() const noexcept { return view(); }
    /// \brief Whether it holds a value.
    /// \return the result.
    constexpr explicit operator bool() const noexcept { return ec == std::errc{}; }
};

/// \brief Converts to text.
/// \param t the t.
/// \param precision the precision.
/// \return the result.
template <utc_duration D>
[[nodiscard]] constexpr rfc3339_text to_text(const utc<D>& t, int precision = utc<D>::fraction_digits) noexcept
{
    rfc3339_text text;
    const auto [end, ec] = to_chars(text.buffer.data(), text.buffer.data() + text.buffer.size(), t, precision);
    text.ec = ec;
    if (ec == std::errc{}) text.size = static_cast<std::uint8_t>(end - text.buffer.data());
    return text;
}

/// \brief Converts to text.
/// \param d the d.
/// \return the result.
[[nodiscard]] constexpr rfc3339_text to_text(const std::chrono::year_month_day& d) noexcept
{
    rfc3339_text text;
    const auto [end, ec] = to_chars(text.buffer.data(), text.buffer.data() + text.buffer.size(), d);
    text.ec = ec;
    if (ec == std::errc{}) text.size = static_cast<std::uint8_t>(end - text.buffer.data());
    return text;
}

/// \brief Allocating convenience.
///
/// Empty if formatting failed (only possible outside 0000..9999).
/// \ingroup core
/// \param t the t.
/// \param precision the precision.
/// \return the result.
template <utc_duration D>
[[nodiscard]] std::string to_string(const utc<D>& t, int precision = utc<D>::fraction_digits)
{
    return std::string(to_text(t, precision).view());
}

/// \brief The text form.
/// \param d the d.
/// \return the text form.
[[nodiscard]] inline std::string to_string(const std::chrono::year_month_day& d) { return std::string(to_text(d).view()); }

/// \brief Writes the RFC 3339 text without allocating; sets failbit if it cannot be formatted.
/// \ingroup core
/// \param os the output stream.
/// \param t the t.
/// \return the result.
template <utc_duration D>
std::ostream& operator<<(std::ostream& os, const utc<D>& t)
{
    const rfc3339_text text = to_text(t);
    if (text)
        os.write(text.buffer.data(), text.size);
    else
        os.setstate(std::ios_base::failbit);
    return os;
}

// ---- parsing ----------------------------------------------------------------------------------------------
//
// Two phases, so that errors are reported in a fixed order - syntax and field validity (invalid_argument) before
// representability (result_out_of_range) - whatever follows the timestamp: scan, then evaluate.

namespace detail {

struct rfc3339_scan
{
    split_instant instant{0, 0, false};   // the UTC instant, with second 60 read as 59
    bool leap = false;                    // second 60 (already checked to be 23:59:60 UTC)
    const char* end = nullptr;            // one past the last character consumed
    std::errc ec{};                       // errc{} or invalid_argument
};

constexpr rfc3339_scan scan_rfc3339(const char* first, const char* last, const parse_options& options) noexcept
{
    rfc3339_scan r;
    r.ec = std::errc::invalid_argument;
    if (last - first < 19) return r;

    unsigned y = 0, mo = 0, d = 0;
    if (!read_date(first, y, mo, d)) return r;
    const char* p = first + 10;
    if (!(*p == 'T' || *p == 't' || *p == ' ')) return r;
    if (!(is_digit(p[1]) && is_digit(p[2]) && p[3] == ':' && is_digit(p[4]) && is_digit(p[5]) && p[6] == ':' &&
          is_digit(p[7]) && is_digit(p[8])))
        return r;
    const unsigned hh = two(p + 1), mi = two(p + 4), ss = two(p + 7);
    if (hh > 23 || mi > 59 || ss > 60) return r;
    p += 9;

    if (p != last && *p == '.') {
        ++p;
        if (p == last || !is_digit(*p)) return r;
        const char* const digits = p;
        const char* const nine = last - p > 9 ? p + 9 : last;
        std::uint32_t nanos = 0;
        for (; p != nine && is_digit(*p); ++p) nanos = nanos * 10 + digit(*p);
        nanos *= static_cast<std::uint32_t>(pow10[static_cast<std::size_t>(9 - (p - digits))]);
        r.instant.nanos = nanos;
        for (; p != last && is_digit(*p); ++p) r.instant.excess |= *p != '0';   // beyond 1 ns
    }

    std::int64_t offset = 0;   // seconds east of UTC
    if (p != last && (*p == 'Z' || *p == 'z')) {
        ++p;
    } else if (p != last && (*p == '+' || *p == '-')) {
        const bool negative = *p == '-';
        if (last - p < 3 || !is_digit(p[1]) || !is_digit(p[2])) return r;
        const unsigned oh = two(p + 1);
        const char* q = p + 3;
        if (last - q >= 3 && *q == ':' && is_digit(q[1]) && is_digit(q[2]))
            q += 1;
        else if (!(options.basic_offset && last - q >= 2 && is_digit(q[0]) && is_digit(q[1])))
            return r;
        const unsigned om = two(q);
        if (oh > 23 || om > 59) return r;
        offset = (static_cast<std::int64_t>(oh) * 3600 + om * 60) * (negative ? -1 : 1);
        p = q + 2;
    } else if (options.require_offset) {
        return r;
    }

    r.leap = ss == 60;
    r.instant.seconds =
        days_from_civil_0000_9999(y, mo, d) * 86400 + (hh * 3600 + mi * 60 + (r.leap ? 59 : ss)) - offset;
    if (r.leap && r.instant.seconds - floor_div(r.instant.seconds, 86400) * 86400 != 86399) return r;   // not 23:59 UTC
    r.end = p;
    r.ec = std::errc{};
    return r;
}

template <utc_duration D>
constexpr std::errc evaluate_rfc3339(const rfc3339_scan& s, const parse_options& options, typename D::rep& ticks) noexcept
{
    if (!s.leap) return to_ticks<D>(s.instant, options.precision, ticks);
    if (options.leap_second == leap_second_policy::reject) return std::errc::result_out_of_range;
    // The last representable instant before the insertion: one tick before ceil<D>(next second).
    const split_instant next_second{s.instant.seconds + 1, 0, false};
    typename D::rep next{};
    if (to_ticks<D>(next_second, precision_policy::reject, next) == std::errc{}) {   // D divides the second
        if (next == std::numeric_limits<typename D::rep>::min()) return std::errc::result_out_of_range;
        ticks = static_cast<typename D::rep>(next - 1);
        return std::errc{};
    }
    return to_ticks<D>(next_second, precision_policy::truncate, ticks);   // the floor already lies before it
}

} // namespace detail

/// \brief Converts from chars.
/// \param first the first.
/// \param last the last.
/// \param out receives the result.
/// \param options the options.
/// \return the result.
template <utc_duration D>
constexpr std::from_chars_result from_chars(const char* first, const char* last, utc<D>& out,
                                            parse_options options = {}) noexcept
{
    const detail::rfc3339_scan s = detail::scan_rfc3339(first, last, options);
    if (s.ec != std::errc{}) return {first, s.ec};
    typename D::rep ticks{};
    const std::errc ec = detail::evaluate_rfc3339<D>(s, options, ticks);
    if (ec != std::errc{}) return {first, ec};
    out = utc<D>(typename utc<D>::time_point(D(ticks)));
    return {s.end, std::errc{}};
}

/// \brief "YYYY-MM-DD" (validated, years 0000..9999).
/// \ingroup core
/// \param first the first.
/// \param last the last.
/// \param out receives the result.
/// \return the result.
constexpr std::from_chars_result from_chars(const char* first, const char* last, std::chrono::year_month_day& out) noexcept
{
    unsigned y = 0, m = 0, d = 0;
    if (last - first < 10 || !detail::read_date(first, y, m, d)) return {first, std::errc::invalid_argument};
    out = std::chrono::year_month_day(std::chrono::year(static_cast<int>(y)), std::chrono::month(m), std::chrono::day(d));
    return {first + 10, std::errc{}};
}

namespace detail {

// One past the syntactically valid Unix-number text -?DIGITS(.DIGITS)? at first, or nullptr.
constexpr const char* scan_unix(const char* first, const char* last) noexcept
{
    const char* p = first;
    if (p != last && *p == '-') ++p;
    if (p == last || !is_digit(*p)) return nullptr;
    while (p != last && is_digit(*p)) ++p;
    if (p != last && *p == '.') {
        ++p;
        if (p == last || !is_digit(*p)) return nullptr;
        while (p != last && is_digit(*p)) ++p;
    }
    return p;
}

// [first, end) is syntactically valid (scan_unix).
template <utc_duration Unit, utc_duration D>
constexpr std::errc evaluate_unix(const char* first, const char* end, const parse_options& options,
                                  typename D::rep& ticks) noexcept
{
    constexpr int unit_digits = utc<Unit>::fraction_digits;   // s 0, ms 3, us 6, ns 9
    const char* p = first;
    const bool negative = *p == '-';
    if (negative) ++p;

    // Magnitude in Unit, up to 2^63 (so INT64_MIN ticks parse) or 2^63 - 1.
    constexpr std::uint64_t int64_max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    const std::uint64_t limit = negative ? int64_max + 1 : int64_max;
    std::uint64_t whole = 0;
    for (; p != end && is_digit(*p); ++p) {
        const unsigned dg = digit(*p);
        if (whole > (limit - dg) / 10) return std::errc::result_out_of_range;
        whole = whole * 10 + dg;
    }
    const auto unit_scale = static_cast<std::uint64_t>(pow10[unit_digits]);
    const std::uint64_t whole_seconds = whole / unit_scale;   // <= 2^63
    split_instant s{0, static_cast<std::int64_t>(whole % unit_scale) * pow10[9 - unit_digits], false};
    if (p != end) {   // '.' DIGITS: digit n of the fraction is worth 10^(8 - unit_digits - n) ns
        ++p;
        for (int n = 0; p != end; ++p, ++n) {
            if (n < 9 - unit_digits)
                s.nanos += digit(*p) * pow10[static_cast<std::size_t>(8 - unit_digits - n)];
            else if (*p != '0')
                s.excess = true;
        }
    }
    if (!negative) {
        s.seconds = static_cast<std::int64_t>(whole_seconds);   // <= 2^63 - 1
    } else if (s.nanos == 0 && !s.excess) {
        s.seconds = whole_seconds == 0 ? 0 : -static_cast<std::int64_t>(whole_seconds - 1) - 1;   // 2^63 -> INT64_MIN
    } else {
        // -(W + n + e) with 0 < n + e < 1 s floors to -(W + 1) s + (1e9 - n - [e > 0]) ns: digits finer than
        // 1 ns move a negative value one more ns into the past.
        if (whole_seconds > int64_max) return std::errc::result_out_of_range;   // only 2^63 itself
        s.seconds = -static_cast<std::int64_t>(whole_seconds) - 1;
        s.nanos = 1000000000 - s.nanos - (s.excess ? 1 : 0);
    }
    return to_ticks<D>(s, options.precision, ticks);
}

} // namespace detail

/// \brief A Unix time in Unit as decimal text: -?DIGITS(.DIGITS)?, e.g.
///
/// "1714486985" (s), "1714486985123" (ms), "1714486985.123456789" (s). Exact (never via double); excess fraction digits follow options.precision. The integer part, counted in Unit, must fit in int64 (else errc::result_out_of_range).
/// \ingroup core
/// \param first the first.
/// \param last the last.
/// \param out receives the result.
/// \param options the options.
/// \return a Unix time in Unit as decimal text: -?DIGITS(.DIGITS)?, e.g.
template <utc_duration Unit, utc_duration D>
    requires(Unit::period::num == 1)
constexpr std::from_chars_result from_unix_chars(const char* first, const char* last, utc<D>& out,
                                                 parse_options options = {}) noexcept
{
    const char* const end = detail::scan_unix(first, last);
    if (end == nullptr) return {first, std::errc::invalid_argument};
    typename D::rep ticks{};
    const std::errc ec = detail::evaluate_unix<Unit, D>(first, end, options, ticks);
    if (ec != std::errc{}) return {first, ec};
    out = utc<D>(typename utc<D>::time_point(D(ticks)));
    return {end, std::errc{}};
}

namespace detail {

template <typename S>
concept char_string = string_like<S> && std::same_as<code_unit_t<S>, char>;

} // namespace detail

/// \brief The whole string as an RFC 3339 timestamp.
///
/// Trailing characters are errc::invalid_argument, reported before any range error.
/// \ingroup core
/// \param text the text.
/// \param options the options.
/// \return the whole string as an RFC 3339 timestamp.
template <utc_duration D = std::chrono::nanoseconds, detail::char_string S>
[[nodiscard]] constexpr parse_result<utc<D>> parse_utc(const S& text, parse_options options = {}) noexcept
{
    const auto sv = to_string_view(text);
    const char* const last = sv.data() + sv.size();
    const detail::rfc3339_scan s = detail::scan_rfc3339(sv.data(), last, options);
    if (s.ec != std::errc{}) return s.ec;
    if (s.end != last) return std::errc::invalid_argument;
    typename D::rep ticks{};
    const std::errc ec = detail::evaluate_rfc3339<D>(s, options, ticks);
    if (ec != std::errc{}) return ec;
    return utc<D>(typename utc<D>::time_point(D(ticks)));
}

/// \brief The whole string as a Unix time in Unit (see from_unix_chars).
/// \ingroup core
/// \param text the text.
/// \param options the options.
/// \return the whole string as a Unix time in Unit (see from_unix_chars).
template <utc_duration Unit, utc_duration D = std::chrono::nanoseconds, detail::char_string S>
    requires(Unit::period::num == 1)
[[nodiscard]] constexpr parse_result<utc<D>> parse_unix(const S& text, parse_options options = {}) noexcept
{
    const auto sv = to_string_view(text);
    const char* const last = sv.data() + sv.size();
    if (detail::scan_unix(sv.data(), last) != last || sv.empty()) return std::errc::invalid_argument;
    typename D::rep ticks{};
    const std::errc ec = detail::evaluate_unix<Unit, D>(sv.data(), last, options, ticks);
    if (ec != std::errc{}) return ec;
    return utc<D>(typename utc<D>::time_point(D(ticks)));
}

/// \brief The whole string as "YYYY-MM-DD".
/// \ingroup core
/// \param text the text.
/// \return the whole string as "YYYY-MM-DD".
template <detail::char_string S>
[[nodiscard]] constexpr parse_result<std::chrono::year_month_day> parse_date(const S& text) noexcept
{
    const auto sv = to_string_view(text);
    std::chrono::year_month_day d{};
    const auto [p, ec] = from_chars(sv.data(), sv.data() + sv.size(), d);
    if (ec != std::errc{}) return ec;
    if (p != sv.data() + sv.size()) return std::errc::invalid_argument;
    return d;
}

namespace literals {

/// \brief "2024-04-30T14:23:05.123Z"_utc: a snicholls::utc<> checked at compile time (strict RFC 3339).
/// \ingroup core
/// \param text the text.
/// \param length the length.
/// \return the result.
consteval utc<std::chrono::nanoseconds> operator""_utc(const char* text, std::size_t length)
{
    const auto r = parse_utc(std::string_view(text, length));
    if (!r) throw "snicholls::operator\"\"_utc: not a valid RFC 3339 timestamp in range";
    return *r;
}

} // namespace literals

} // namespace snicholls

#endif /* snicholls_core_time_rfc3339_hpp */
