//
//  rfc3339_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/rfc3339.hpp>.
//

// interfaces/utc/rfc3339.hpp: RFC 3339 formatting and parsing, Unix-number text, dates, parse_result.
//
// Correctness strategy: (1) golden values; (2) exact round trips over every day of 0000..9999 and random instants
// over utc<>'s whole range; (3) formatting compared with std::format's chrono formatter; (4) parsing compared with
// an independent reference parser (std::regex + std::chrono + 128-bit arithmetic) over a mutation corpus that
// covers every malformed-input class; (5) named tests per rejection class.

// GCC 13 with -fsanitize=address,undefined reports a false -Wmaybe-uninitialized inside libstdc++'s <regex>
// (std::function moved inside _NFA::_M_insert_alt), which the reference parsers use. Tests only; Clang and GCC
// without sanitizers are silent.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <array>
#include <chrono>
#include <compare>
#include <concepts>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/time/rfc3339.hpp>

#include "../support/time_fixtures.hpp"

namespace {

using namespace std::chrono;
using namespace std::chrono_literals;
using snicholls::utc;

// A portable signed 128-bit integer for the reference parsers below: two's complement, wrapping. Not
// GCC/Clang's __int128, which MSVC does not have and whose multiplication needs a runtime helper (__muloti4) that
// the Linux ARM64 Clang toolchain does not link. Only what the references use is provided.
class int128
{
public:
    constexpr int128() noexcept = default;
    template <std::integral I>
    constexpr int128(I v) noexcept   // NOLINT: implicit, like the built-in type
        : lo_(static_cast<std::uint64_t>(v)), hi_(std::is_signed_v<I> && v < 0 ? ~std::uint64_t{0} : 0)
    {
    }
    template <std::integral I>
    constexpr explicit operator I() const noexcept { return static_cast<I>(lo_); }

    friend constexpr int128 operator+(int128 a, int128 b) noexcept
    {
        int128 r;
        r.lo_ = a.lo_ + b.lo_;
        r.hi_ = a.hi_ + b.hi_ + (r.lo_ < a.lo_ ? 1 : 0);
        return r;
    }
    friend constexpr int128 operator-(int128 a) noexcept { return int128::from(~a.hi_, ~a.lo_) + 1; }
    friend constexpr int128 operator-(int128 a, int128 b) noexcept { return a + -b; }
    friend constexpr int128 operator*(int128 a, int128 b) noexcept
    {
        const std::uint64_t a0 = a.lo_ & 0xFFFFFFFFU, a1 = a.lo_ >> 32, b0 = b.lo_ & 0xFFFFFFFFU, b1 = b.lo_ >> 32;
        const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
        const std::uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFU) + (p10 & 0xFFFFFFFFU);
        const std::uint64_t lo = (mid << 32) | (p00 & 0xFFFFFFFFU);
        const std::uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32) + a.lo_ * b.hi_ + a.hi_ * b.lo_;
        return from(hi, lo);
    }
    friend constexpr int128 operator/(int128 a, int128 b) noexcept { return divmod(a, b).first; }
    friend constexpr int128 operator%(int128 a, int128 b) noexcept { return divmod(a, b).second; }
    constexpr int128& operator*=(int128 b) noexcept { return *this = *this * b; }
    constexpr int128& operator--() noexcept { return *this = *this - 1; }

    friend constexpr bool operator==(int128 a, int128 b) noexcept = default;
    friend constexpr std::strong_ordering operator<=>(int128 a, int128 b) noexcept
    {
        if (a.hi_ != b.hi_) return static_cast<std::int64_t>(a.hi_) <=> static_cast<std::int64_t>(b.hi_);
        return a.lo_ <=> b.lo_;
    }

private:
    static constexpr int128 from(std::uint64_t hi, std::uint64_t lo) noexcept
    {
        int128 r;
        r.hi_ = hi;
        r.lo_ = lo;
        return r;
    }
    constexpr bool negative() const noexcept { return (hi_ >> 63) != 0; }
    // Truncating division, as for the built-in type (b != 0).
    static constexpr std::pair<int128, int128> divmod(int128 a, int128 b) noexcept
    {
        const bool qneg = a.negative() != b.negative(), rneg = a.negative();
        const int128 n = a.negative() ? -a : a, d = b.negative() ? -b : b;
        int128 q, r;
        for (int i = 127; i >= 0; --i) {
            r = from((r.hi_ << 1) | (r.lo_ >> 63), r.lo_ << 1);
            const std::uint64_t bit = i >= 64 ? (n.hi_ >> (i - 64)) & 1 : (n.lo_ >> i) & 1;
            r.lo_ |= bit;
            if (!(r.hi_ < d.hi_ || (r.hi_ == d.hi_ && r.lo_ < d.lo_))) {
                r = r - d;
                if (i >= 64) q.hi_ |= std::uint64_t{1} << (i - 64);
                else q.lo_ |= std::uint64_t{1} << i;
            }
        }
        return {qneg ? -q : q, rneg ? -r : r};
    }

    std::uint64_t lo_ = 0;
    std::uint64_t hi_ = 0;
};

static_assert(int128(-7) / 2 == -3 && int128(-7) % 2 == -1 && int128(7) / -2 == -3);
static_assert(int128(std::numeric_limits<std::int64_t>::max()) * 10 / 10 == std::numeric_limits<std::int64_t>::max());
static_assert(int128(-1) < 0 && int128(1) > -1 && -int128(5) + 5 == 0);

template <typename D>
std::string fmt(const utc<D>& t, int precision = utc<D>::fraction_digits)
{
    return snicholls::to_string(t, precision);
}

template <typename D = nanoseconds>
std::errc parse_error(std::string_view s, snicholls::parse_options o = {})
{
    return snicholls::parse_utc<D>(s, o).error();
}

// ---- formatting: golden values ----------------------------------------------------------------------------

TEST(UtcRfc3339Format, GoldenValuesAtEveryResolution)
{
    constexpr std::int64_t ns = 1714486985123456789;
    EXPECT_EQ(fmt(utc<>::from_unix_nanos(ns)), "2024-04-30T14:23:05.123456789Z");
    EXPECT_EQ(fmt(utc<microseconds>::from_unix_nanos(ns)), "2024-04-30T14:23:05.123456Z");
    EXPECT_EQ(fmt(utc<milliseconds>::from_unix_nanos(ns)), "2024-04-30T14:23:05.123Z");
    EXPECT_EQ(fmt(utc<seconds>::from_unix_nanos(ns)), "2024-04-30T14:23:05Z");
    EXPECT_EQ(fmt(utc<minutes>::from_unix_nanos(ns)), "2024-04-30T14:23:00Z");
    EXPECT_EQ(fmt(utc<hours>::from_unix_nanos(ns)), "2024-04-30T14:00:00Z");
    EXPECT_EQ(fmt(utc<days>::from_unix_nanos(ns)), "2024-04-30T00:00:00Z");
    EXPECT_EQ(fmt(utc<duration<std::int64_t, std::centi>>::from_unix_nanos(ns)), "2024-04-30T14:23:05.12Z");
    EXPECT_EQ(fmt(utc<duration<std::int64_t, std::ratio<1, 10000000>>>::from_unix_nanos(ns)), "2024-04-30T14:23:05.1234567Z");
    EXPECT_EQ(fmt(utc<>{}), "1970-01-01T00:00:00.000000000Z");
    EXPECT_EQ(fmt(utc<>::from_unix_nanos(-1)), "1969-12-31T23:59:59.999999999Z");
    EXPECT_EQ(fmt(utc<>::min()), "1677-09-21T00:12:43.145224192Z");
    EXPECT_EQ(fmt(utc<>::max()), "2262-04-11T23:47:16.854775807Z");
}

TEST(UtcRfc3339Format, PrecisionFloorsOrPads)
{
    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    const std::array<const char*, 10> expected = {
        "2024-04-30T14:23:05Z",           "2024-04-30T14:23:05.1Z",        "2024-04-30T14:23:05.12Z",
        "2024-04-30T14:23:05.123Z",       "2024-04-30T14:23:05.1234Z",     "2024-04-30T14:23:05.12345Z",
        "2024-04-30T14:23:05.123456Z",    "2024-04-30T14:23:05.1234567Z",  "2024-04-30T14:23:05.12345678Z",
        "2024-04-30T14:23:05.123456789Z"};
    for (int p = 0; p <= 9; ++p) {
        EXPECT_EQ(fmt(t, p), expected[static_cast<std::size_t>(p)]);
        EXPECT_EQ(fmt(t, p).size(), snicholls::rfc3339_length(p));
    }
    EXPECT_EQ(fmt(utc<milliseconds>::from_unix_millis(1714486985123), 9), "2024-04-30T14:23:05.123000000Z");
    EXPECT_EQ(fmt(utc<seconds>::from_unix_seconds(1714486985), 3), "2024-04-30T14:23:05.000Z");
    EXPECT_EQ(fmt(utc<>::from_unix_nanos(-1), 3), "1969-12-31T23:59:59.999Z");   // floored, never rounded up
}

TEST(UtcRfc3339Format, ErrorsLikeStdToChars)
{
    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    std::array<char, 64> buf{};
    char* const b = buf.data();
    for (std::size_t n = 0; n < 30; ++n) {   // one char short of 30 fails, without writing past `last`
        const auto r = snicholls::to_chars(b, b + n, t);
        ASSERT_EQ(r.ec, std::errc::value_too_large) << n;
        ASSERT_EQ(r.ptr, b + n);
    }
    const auto ok = snicholls::to_chars(b, b + 30, t);
    EXPECT_EQ(ok.ec, std::errc{});
    EXPECT_EQ(ok.ptr, b + 30);
    EXPECT_EQ(snicholls::to_chars(b, b + 64, t, -1).ec, std::errc::invalid_argument);
    EXPECT_EQ(snicholls::to_chars(b, b + 64, t, 10).ec, std::errc::invalid_argument);

    // Outside 0000..9999 (only for resolutions coarser than ns).
    const auto y10000 = utc<seconds>(sys_days{year(10000) / January / 1});
    const auto before_year_0 = utc<seconds>(sys_days{year(0) / January / 1}) - 1s;
    EXPECT_EQ(snicholls::to_chars(b, b + 64, y10000).ec, std::errc::value_too_large);
    EXPECT_EQ(snicholls::to_chars(b, b + 64, before_year_0).ec, std::errc::value_too_large);
    EXPECT_EQ(fmt(y10000 - 1s), "9999-12-31T23:59:59Z");
    EXPECT_EQ(fmt(before_year_0 + 1s), "0000-01-01T00:00:00Z");
    EXPECT_EQ(snicholls::to_chars(b, b + 64, utc<seconds>::max()).ec, std::errc::value_too_large);   // no overflow
    EXPECT_EQ(snicholls::to_chars(b, b + 64, utc<seconds>::min()).ec, std::errc::value_too_large);
    EXPECT_EQ(snicholls::to_chars(b, b + 64, utc<days>::max()).ec, std::errc::value_too_large);
    EXPECT_EQ(snicholls::to_chars(b, b + 64, utc<microseconds>::min()).ec, std::errc::value_too_large);
    const snicholls::rfc3339_text failed = snicholls::to_text(y10000);
    EXPECT_FALSE(failed);
    EXPECT_EQ(failed.view(), "");
    EXPECT_EQ(snicholls::to_string(y10000), "");
}

TEST(UtcRfc3339Format, ConstexprNoexceptAndBufferSizes)
{
    static_assert(snicholls::max_chars_v<nanoseconds> == 30 && snicholls::max_chars_v<milliseconds> == 24);
    static_assert(snicholls::max_chars_v<seconds> == 20 && snicholls::rfc3339_max_chars == 30);
    static_assert(snicholls::to_text(utc<>::from_unix_millis(1714486985123)).view() == "2024-04-30T14:23:05.123000000Z");
    static_assert(noexcept(snicholls::to_text(utc<>{})) && noexcept(snicholls::to_chars(nullptr, nullptr, utc<>{})));
    const snicholls::rfc3339_text text = snicholls::to_text(utc<>{}, 0);
    const std::string_view view = text;
    EXPECT_EQ(view, "1970-01-01T00:00:00Z");
    EXPECT_TRUE(text);
}

TEST(UtcRfc3339Format, StreamWritesWithoutFormattingState)
{
    std::ostringstream os;
    os << utc<milliseconds>::from_unix_millis(1714486985123) << '|' << utc<seconds>{};
    EXPECT_EQ(os.str(), "2024-04-30T14:23:05.123Z|1970-01-01T00:00:00Z");
    std::ostringstream bad;
    bad << utc<seconds>::max();
    EXPECT_TRUE(bad.fail());
}

// Independent oracle: the standard library's chrono formatter.
TEST(UtcRfc3339Format, MatchesStdFormatForRandomInstantsAcrossTheWholeRange)
{
    std::mt19937_64 rng(2024);
    for (int i = 0; i < 200000; ++i) {
        const utc<> t = utc<>::from_unix_nanos(static_cast<std::int64_t>(rng()));
        ASSERT_EQ(fmt(t), std::format("{:%FT%TZ}", t.sys_time()));
        const utc<milliseconds> m(floor<milliseconds>(t.sys_time()));
        ASSERT_EQ(fmt(m), std::format("{:%FT%TZ}", m.sys_time()));
    }
    // Every day 0000..9999 in seconds resolution (std::format's own range).
    const auto first = sys_days{year(0) / January / 1}.time_since_epoch().count();
    const auto last = sys_days{year(9999) / December / 31}.time_since_epoch().count();
    for (auto d = first; d <= last; d += 97) {   // a stride co-prime with 7 and 365 visits every weekday/month
        const utc<seconds> s(sys_days(days(d)) + seconds(d % 86400 < 0 ? -(d % 86400) : d % 86400));
        ASSERT_EQ(fmt(s), std::format("{:%FT%TZ}", s.sys_time()));
    }
}

// ---- round trips ------------------------------------------------------------------------------------------

TEST(UtcRfc3339RoundTrip, EveryDayOfYears0000To9999)
{
    const auto first = sys_days{year(0) / January / 1}.time_since_epoch().count();
    const auto last = sys_days{year(9999) / December / 31}.time_since_epoch().count();
    std::mt19937_64 rng(1);
    char buf[snicholls::max_chars_v<seconds>];
    for (auto d = first; d <= last; ++d) {
        const utc<seconds> t(sys_days(days(d)) + seconds(static_cast<std::int64_t>(rng() % 86400)));
        const auto w = snicholls::to_chars(buf, buf + sizeof buf, t);
        ASSERT_EQ(w.ec, std::errc{});
        utc<seconds> back;
        const auto r = snicholls::from_chars(buf, w.ptr, back);
        ASSERT_EQ(r.ec, std::errc{});
        ASSERT_EQ(r.ptr, w.ptr);
        ASSERT_EQ(back, t) << std::string_view(buf, static_cast<std::size_t>(w.ptr - buf));
    }
}

template <typename D>
void round_trip_random(std::uint64_t seed, int count)
{
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i) {
        const utc<D> t(typename utc<D>::time_point(D(static_cast<typename D::rep>(rng() >> 8))));   // in range
        const snicholls::rfc3339_text text = snicholls::to_text(t);
        if (!text) continue;   // outside 0000..9999 for coarse D
        const auto back = snicholls::parse_utc<D>(text.view());
        ASSERT_TRUE(back.has_value()) << text.view();
        ASSERT_EQ(*back, t) << text.view();
    }
}

TEST(UtcRfc3339RoundTrip, RandomInstantsAtEveryResolution)
{
    std::mt19937_64 rng(3);
    for (int i = 0; i < 300000; ++i) {   // the full int64 range for ns
        const utc<> t = utc<>::from_unix_nanos(static_cast<std::int64_t>(rng()));
        const auto back = snicholls::parse_utc(snicholls::to_text(t).view());
        ASSERT_TRUE(back.has_value());
        ASSERT_EQ(*back, t);
    }
    round_trip_random<microseconds>(4, 50000);
    round_trip_random<milliseconds>(5, 50000);
    round_trip_random<duration<std::int64_t, std::ratio<1, 10000000>>>(6, 50000);
    round_trip_random<duration<std::int64_t, std::centi>>(7, 50000);
    for (std::int64_t s = -62167219200; s <= 253402300799; s += 999999937) {   // seconds across 0000..9999
        const auto t = utc<seconds>::from_unix_seconds(s);
        ASSERT_EQ(*snicholls::parse_utc<seconds>(snicholls::to_text(t).view()), t);
    }
}

TEST(UtcRfc3339RoundTrip, EveryPrecisionParsesBackToTheFlooredValue)
{
    std::mt19937_64 rng(8);
    for (int i = 0; i < 20000; ++i) {
        const utc<> t = utc<>::from_unix_nanos(static_cast<std::int64_t>(rng()));
        for (int p = 0; p <= 9; ++p) {
            const auto back = snicholls::parse_utc(snicholls::to_text(t, p).view());
            ASSERT_TRUE(back.has_value());
            const std::int64_t unit = std::array<std::int64_t, 10>{1000000000, 100000000, 10000000, 1000000, 100000,
                                                                   10000, 1000, 100, 10, 1}[static_cast<std::size_t>(p)];
            const std::int64_t ns = t.unix_nanos();
            const std::int64_t floored = (ns / unit - (ns % unit < 0)) * unit;
            ASSERT_EQ(back->unix_nanos(), floored);
        }
    }
}

TEST(UtcRfc3339RoundTrip, BoundariesOfTheNanosecondRange)
{
    EXPECT_EQ(*snicholls::parse_utc("1677-09-21T00:12:43.145224192Z"), utc<>::min());
    EXPECT_EQ(*snicholls::parse_utc("2262-04-11T23:47:16.854775807Z"), utc<>::max());
    EXPECT_EQ(parse_error("1677-09-21T00:12:43.145224191Z"), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error("2262-04-11T23:47:16.854775808Z"), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error("0000-01-01T00:00:00Z"), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error("9999-12-31T23:59:59Z"), std::errc::result_out_of_range);
    EXPECT_EQ(*snicholls::parse_utc<microseconds>("0000-01-01T00:00:00Z"), utc<microseconds>(sys_days{year(0) / January / 1}));
    EXPECT_EQ(*snicholls::parse_utc<microseconds>("9999-12-31T23:59:59.999999Z"),
              utc<microseconds>(sys_days{year(10000) / January / 1}) - 1us);
    // The offset can carry an instant across the boundary.
    EXPECT_EQ(*snicholls::parse_utc("2262-04-12T00:47:16.854775807+01:00"), utc<>::max());
    EXPECT_EQ(parse_error("2262-04-11T23:47:16.854775807-00:01"), std::errc::result_out_of_range);
    EXPECT_EQ(*snicholls::parse_utc<microseconds>("0000-01-01T00:00:00+01:00"),
              utc<microseconds>(sys_days{year(0) / January / 1}) - 1h);   // year -1: fine in microseconds
    EXPECT_EQ(*snicholls::parse_utc<days>("0001-01-01T00:00:00Z"), utc<days>(sys_days{year(1) / January / 1}));
}

// ---- parsing: accepted syntax -----------------------------------------------------------------------------

TEST(UtcRfc3339Parse, AcceptsEverySpelling)
{
    const utc<> t = utc<>::from_unix_seconds(1714486985);
    for (const char* s : {"2024-04-30T14:23:05Z", "2024-04-30t14:23:05z", "2024-04-30 14:23:05Z",
                          "2024-04-30T14:23:05+00:00", "2024-04-30T14:23:05-00:00", "2024-04-30T16:23:05+02:00",
                          "2024-04-30T09:53:05-04:30", "2024-04-30T14:23:05.0Z", "2024-04-30T14:23:05.000000000000000Z",
                          "2024-05-01T13:23:05+23:00", "2024-04-29T15:23:05-23:00"}) {
        const auto r = snicholls::parse_utc(std::string_view(s));
        ASSERT_TRUE(r.has_value()) << s;
        EXPECT_EQ(*r, t) << s;
    }
    EXPECT_EQ(snicholls::parse_utc("2024-04-30T14:23:05.5Z")->unix_millis(), 1714486985500);
    EXPECT_EQ(snicholls::parse_utc("2024-04-30T14:23:05.000000001Z")->unix_nanos(), 1714486985000000001);
}

TEST(UtcRfc3339Parse, AcceptsEveryStringType)
{
    const std::string s = "2024-04-30T14:23:05Z";
    const char* c = s.c_str();
    char array[] = "2024-04-30T14:23:05Z";
    const auto expected = utc<>::from_unix_seconds(1714486985);
    EXPECT_EQ(*snicholls::parse_utc(s), expected);
    EXPECT_EQ(*snicholls::parse_utc(c), expected);
    EXPECT_EQ(*snicholls::parse_utc(array), expected);
    EXPECT_EQ(*snicholls::parse_utc("2024-04-30T14:23:05Z"), expected);
    EXPECT_EQ(*snicholls::parse_utc(std::string_view(s)), expected);
    const char* null = nullptr;
    EXPECT_EQ(snicholls::parse_utc(null).error(), std::errc::invalid_argument);   // a null C string is ""
}

TEST(UtcRfc3339Parse, FromCharsParsesAPrefixLikeStdFromChars)
{
    const std::string_view s = "2024-04-30T14:23:05Z,next";
    utc<> t;
    const auto r = snicholls::from_chars(s.data(), s.data() + s.size(), t);
    EXPECT_EQ(r.ec, std::errc{});
    EXPECT_EQ(r.ptr, s.data() + 20);
    EXPECT_EQ(t.unix_seconds(), 1714486985);
    EXPECT_EQ(parse_error(s), std::errc::invalid_argument);   // parse_utc wants the whole string

    // Failure: ptr == first and the output is untouched.
    utc<> untouched = utc<>::from_unix_nanos(42);
    const std::string_view bad = "2024-04-31T14:23:05Z";
    const auto f = snicholls::from_chars(bad.data(), bad.data() + bad.size(), untouched);
    EXPECT_EQ(f.ec, std::errc::invalid_argument);
    EXPECT_EQ(f.ptr, bad.data());
    EXPECT_EQ(untouched.unix_nanos(), 42);

    // Fraction digits stop at the first non-digit; with the offset optional the rest is left unparsed.
    const std::string_view loose = "2024-04-30T14:23:05.25 rest";
    const auto l = snicholls::from_chars(loose.data(), loose.data() + loose.size(), t, snicholls::parse_options::lenient());
    EXPECT_EQ(l.ec, std::errc{});
    EXPECT_EQ(l.ptr, loose.data() + 22);
    EXPECT_EQ(t.unix_millis(), 1714486985250);
}

// ---- parsing: rejection classes ---------------------------------------------------------------------------

TEST(UtcRfc3339Reject, SyntaxErrors)
{
    for (const char* s : {
             "",                              // empty
             "2024-04-30",                    // date only
             "2024-04-30T14:23Z",             // no seconds (the original's silent seconds-zeroing case)
             "2024-04-30T14:23:05",           // no offset (strict)
             "24-04-30T14:23:05Z",            // 2-digit year
             "+2024-04-30T14:23:05Z",         // signed year
             "20240-04-30T14:23:05Z",         // 5-digit year
             "2024-4-30T14:23:05Z",           // 1-digit month
             "2024-04-3T14:23:05Z",           // 1-digit day
             "2024/04/30T14:23:05Z",          // wrong date separator
             "2024-04-30_14:23:05Z",          // wrong date/time separator
             "2024-04-30T14.23.05Z",          // wrong time separator
             "2024-04-30T1:23:05Z",           // 1-digit hour
             "2024-04-30T14:23:5Z",           // 1-digit second
             "20240430T142305Z",              // ISO 8601 basic format
             "2024-04-30T14:23:05.Z",         // empty fraction
             "2024-04-30T14:23:05,5Z",        // comma fraction
             "2024-04-30T14:23:05ZZ",         // trailing characters
             "2024-04-30T14:23:05Z ",         // trailing space
             " 2024-04-30T14:23:05Z",         // leading space
             "2024-04-30T14:23:05+01",        // offset without minutes
             "2024-04-30T14:23:05+1:00",      // 1-digit offset hour
             "2024-04-30T14:23:05+0100",      // basic offset (strict)
             "2024-04-30T14:23:05+01:0",      // 1-digit offset minute
             "2024-04-30T14:23:05UTC",        // zone name
             "2024-04-30T14:23:05GMT",
             "2024-04-30T14:23:05 +01:00",    // space before offset
             "2024-04-30T14:23:05\xEF\xBC\xBA",   // fullwidth Z
             "١٢٣٤-04-30T14:23:05Z",          // non-ASCII digits
             "1714486985",                    // a Unix number (use parse_unix)
             "Tue, 30 Apr 2024 14:23:05 GMT", // RFC 1123 (not RFC 3339)
             "04/30/2024 14:23:05"}) {        // US order: never guessed
        EXPECT_EQ(parse_error(s), std::errc::invalid_argument) << '"' << s << '"';
    }
    EXPECT_EQ(parse_error(std::string_view("2024-04-30T14:23:05\0Z", 21)), std::errc::invalid_argument);   // NUL
}

TEST(UtcRfc3339Reject, InvalidDatesAndTimes)
{
    for (const char* s : {"2024-00-10T00:00:00Z", "2024-13-10T00:00:00Z", "2024-01-00T00:00:00Z", "2024-01-32T00:00:00Z",
                          "2023-02-29T00:00:00Z", "2024-02-30T00:00:00Z", "1900-02-29T00:00:00Z", "2024-04-31T00:00:00Z",
                          "2024-06-31T00:00:00Z", "2024-09-31T00:00:00Z", "2024-11-31T00:00:00Z", "2024-01-01T24:00:00Z",
                          "2024-01-01T23:60:00Z", "2024-01-01T23:59:61Z", "2024-01-01T99:00:00Z",
                          "2024-01-01T00:00:00+24:00", "2024-01-01T00:00:00+23:60", "2024-01-01T00:00:00-99:00"}) {
        EXPECT_EQ(parse_error(s), std::errc::invalid_argument) << s;
        EXPECT_EQ(parse_error(s, snicholls::parse_options::lenient()), std::errc::invalid_argument) << s;
    }
    EXPECT_TRUE(snicholls::parse_utc("2000-02-29T00:00:00Z"));   // 400-year rule
    EXPECT_TRUE(snicholls::parse_utc("2024-02-29T23:59:59.999999999Z"));
    EXPECT_TRUE(snicholls::parse_utc("2024-01-01T00:00:00+23:59"));
}

TEST(UtcRfc3339Reject, ExcessPrecisionUnlessTruncating)
{
    const char* s = "2024-04-30T14:23:05.1234567891Z";   // 10 digits: finer than ns
    EXPECT_EQ(parse_error(s), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error<milliseconds>("2024-04-30T14:23:05.1234Z"), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error<seconds>("2024-04-30T14:23:05.5Z"), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error<minutes>("2024-04-30T14:23:05Z"), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error<days>("2024-04-30T00:00:00+01:00"), std::errc::result_out_of_range);   // 23:00 the day before
    // Excess zeros are exact, so always accepted.
    EXPECT_EQ(snicholls::parse_utc<milliseconds>("2024-04-30T14:23:05.123000000000Z")->unix_millis(), 1714486985123);
    EXPECT_EQ(snicholls::parse_utc<minutes>("2024-04-30T14:23:00.000Z")->unix_seconds(), 1714486980);

    constexpr snicholls::parse_options truncate{.precision = snicholls::precision_policy::truncate};
    EXPECT_EQ(snicholls::parse_utc(s, truncate)->unix_nanos(), 1714486985123456789);
    EXPECT_EQ(snicholls::parse_utc<milliseconds>("2024-04-30T14:23:05.1239Z", truncate)->unix_millis(), 1714486985123);
    EXPECT_EQ(snicholls::parse_utc<milliseconds>("1969-12-31T23:59:59.9999Z", truncate)->unix_millis(), -1);   // floor
    EXPECT_EQ(snicholls::parse_utc<days>("2024-04-30T00:00:00+01:00", truncate)->days(), sys_days{2024y / April / 29});
}

TEST(UtcRfc3339Reject, LeapSecondsByPolicy)
{
    constexpr snicholls::parse_options clamp{.leap_second = snicholls::leap_second_policy::clamp};
    const utc<> next_day = utc<>(sys_days{2017y / January / 1});
    EXPECT_EQ(parse_error("2016-12-31T23:59:60Z"), std::errc::result_out_of_range);
    EXPECT_EQ(*snicholls::parse_utc("2016-12-31T23:59:60Z", clamp), next_day - 1ns);
    EXPECT_EQ(*snicholls::parse_utc("2016-12-31T23:59:60.999999999Z", clamp), next_day - 1ns);
    EXPECT_EQ(*snicholls::parse_utc("2016-12-31T18:59:60.5-05:00", clamp), next_day - 1ns);   // 23:59:60 UTC
    EXPECT_EQ(*snicholls::parse_utc("2017-01-01T05:29:60+05:30", clamp), next_day - 1ns);
    EXPECT_EQ(*snicholls::parse_utc<milliseconds>("2016-12-31T23:59:60Z", clamp), utc<milliseconds>(sys_days{2017y / January / 1} - 1ms));
    EXPECT_EQ(*snicholls::parse_utc<seconds>("2016-12-31T23:59:60Z", clamp), utc<seconds>(sys_days{2017y / January / 1} - 1s));
    EXPECT_EQ(*snicholls::parse_utc<minutes>("2016-12-31T23:59:60Z", clamp), utc<minutes>(sys_days{2017y / January / 1} - 1min));
    EXPECT_EQ(*snicholls::parse_utc<days>("2016-12-31T23:59:60Z", clamp), utc<days>(sys_days{2016y / December / 31}));
    EXPECT_EQ(*snicholls::parse_utc<hours>("2016-12-31T23:59:60Z", clamp), utc<hours>(sys_days{2017y / January / 1} - 1h));
    // :60 anywhere but 23:59 UTC is invalid under every policy.
    for (const char* s : {"2016-12-31T23:58:60Z", "2016-12-31T22:59:60Z", "2016-12-31T23:59:60+01:00",
                          "2016-12-31T12:00:60Z"}) {
        EXPECT_EQ(parse_error(s), std::errc::invalid_argument) << s;
        EXPECT_EQ(parse_error(s, clamp), std::errc::invalid_argument) << s;
    }
    // Not checked against the IERS table (needs tzdb, absent on macOS): any 23:59:60 UTC is accepted when clamping.
    EXPECT_TRUE(snicholls::parse_utc("2024-04-30T23:59:60Z", clamp));
}

#if defined(__GLIBCXX__) && _GLIBCXX_RELEASE >= 13
// Where std::chrono::utc_clock exists (libstdc++ - not libc++ yet), clamp is exactly utc_clock::to_sys.
TEST(UtcRfc3339Reject, LeapClampEqualsStdUtcClockToSys)
{
    constexpr snicholls::parse_options clamp{.leap_second = snicholls::leap_second_policy::clamp};
    const auto leap = clock_cast<utc_clock>(sys_seconds{sys_days{2017y / January / 1}}) - 1s;   // 23:59:60
    ASSERT_TRUE(get_leap_second_info(leap).is_leap_second);
    for (const auto frac : {0ns, 500000000ns, 999999999ns}) {
        const auto expected = utc_clock::to_sys(leap + frac);
        EXPECT_EQ(snicholls::parse_utc("2016-12-31T23:59:60Z", clamp)->sys_time(), expected);
    }
}
#endif

TEST(UtcRfc3339Reject, LenientOptionsEachRelaxOneRule)
{
    constexpr snicholls::parse_options no_offset{.require_offset = false};
    constexpr snicholls::parse_options basic{.basic_offset = true};
    EXPECT_EQ(snicholls::parse_utc("2024-04-30T14:23:05", no_offset)->unix_seconds(), 1714486985);
    EXPECT_EQ(snicholls::parse_utc("2024-04-30 14:23:05.5", no_offset)->unix_millis(), 1714486985500);
    EXPECT_EQ(parse_error("2024-04-30T14:23:05+0100", no_offset), std::errc::invalid_argument);
    EXPECT_EQ(snicholls::parse_utc("2024-04-30T15:23:05+0100", basic)->unix_seconds(), 1714486985);
    EXPECT_EQ(snicholls::parse_utc("2024-04-30T15:23:05+01:00", basic)->unix_seconds(), 1714486985);
    EXPECT_EQ(parse_error("2024-04-30T14:23:05", basic), std::errc::invalid_argument);
    EXPECT_EQ(parse_error("2024-04-30T14:23:05+01", snicholls::parse_options::lenient()), std::errc::invalid_argument);
    EXPECT_EQ(parse_error("2024-04-30T14:23:05+01:", snicholls::parse_options::lenient()), std::errc::invalid_argument);
    EXPECT_EQ(parse_error("2024-04-30T14:23:05+", snicholls::parse_options::lenient()), std::errc::invalid_argument);
}

// ---- parsing: differential fuzzing against an independent reference ---------------------------------------

// Reference parser: std::regex for the grammar, std::chrono for the calendar, 128-bit integers for the value.
template <typename D>
std::errc reference_parse(const std::string& s, const snicholls::parse_options& o, typename D::rep& out)
{
    static const std::regex strict(R"(^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$)");
    static const std::regex lenient(R"(^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(\.\d+)?([Zz]|[+-]\d{2}:?\d{2})?$)");
    static const std::regex basic_only(R"(^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(\.\d+)?([Zz]|[+-]\d{2}:?\d{2})$)");
    static const std::regex no_offset(R"(^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(\.\d+)?([Zz]|[+-]\d{2}:\d{2})?$)");
    const std::regex& re = o.basic_offset ? (o.require_offset ? basic_only : lenient) : (o.require_offset ? strict : no_offset);
    if (s.find('\0') != std::string::npos) return std::errc::invalid_argument;
    std::smatch m;
    if (!std::regex_match(s, m, re)) return std::errc::invalid_argument;
    const int y = std::stoi(m[1]);
    const unsigned mo = static_cast<unsigned>(std::stoi(m[2])), d = static_cast<unsigned>(std::stoi(m[3]));
    const int h = std::stoi(m[4]), mi = std::stoi(m[5]), sec = std::stoi(m[6]);
    const year_month_day ymd{year(y), month(mo), day(d)};
    if (!ymd.ok() || h > 23 || mi > 59 || sec > 60) return std::errc::invalid_argument;
    std::int64_t offset = 0;
    const std::string z = m[8];
    if (!z.empty() && z != "Z" && z != "z") {
        const int oh = std::stoi(z.substr(1, 2));
        const int om = std::stoi(z.substr(z[3] == ':' ? 4 : 3, 2));
        if (oh > 23 || om > 59) return std::errc::invalid_argument;
        offset = (oh * 3600 + om * 60) * (z[0] == '-' ? -1 : 1);
    }
    const std::int64_t secs = (sys_days(ymd) + hours(h) + minutes(mi) + seconds(sec == 60 ? 59 : sec)).time_since_epoch().count() - offset;
    const std::string frac = m[7].matched ? std::string(m[7]).substr(1) : std::string();
    using period = typename D::period;
    constexpr int k = utc<D>::fraction_digits;
    int128 ticks = 0;
    if (sec == 60) {
        if (((secs % 86400) + 86400) % 86400 != 86399) return std::errc::invalid_argument;
        if (o.leap_second == snicholls::leap_second_policy::reject) return std::errc::result_out_of_range;
        const int128 next = static_cast<int128>(secs + 1) * period::den;   // next second in units of 1/den s
        // ceil(next / num) - 1
        const int128 q = next >= 0 ? (next + period::num - 1) / period::num : -((-next) / period::num);
        ticks = q - 1;
    } else {
        bool excess = false;
        int128 sub = 0;
        if (period::den > 1) {
            for (int i = 0; i < k; ++i) sub = sub * 10 + (i < static_cast<int>(frac.size()) ? frac[static_cast<std::size_t>(i)] - '0' : 0);
            for (std::size_t i = static_cast<std::size_t>(k); i < frac.size(); ++i) excess |= frac[i] != '0';
            ticks = static_cast<int128>(secs) * period::den + sub;
        } else {
            for (char c : frac) excess |= c != '0';
            const int128 n = period::num;
            const int128 q = secs >= 0 ? secs / n : -((-static_cast<int128>(secs) + n - 1) / n);
            excess |= q * n != secs;
            ticks = q;
        }
        if (excess && o.precision == snicholls::precision_policy::reject) return std::errc::result_out_of_range;
    }
    if (ticks < std::numeric_limits<typename D::rep>::min() || ticks > std::numeric_limits<typename D::rep>::max())
        return std::errc::result_out_of_range;
    out = static_cast<typename D::rep>(ticks);
    return std::errc{};
}

std::vector<std::string> mutation_corpus()
{
    const std::vector<std::string> seeds = {
        "2024-04-30T14:23:05Z", "2024-02-29t00:00:00.123456789012z", "1999-12-31 23:59:59.5+05:30",
        "2016-12-31T23:59:60Z", "2016-12-31T18:59:60.25-05:00", "1677-09-21T00:12:43.145224192Z",
        "2262-04-11T23:47:16.854775807Z", "0000-01-01T00:00:00Z", "9999-12-31T23:59:59.999999999999Z",
        "2024-04-30T14:23:05-0030", "2023-02-28T23:59:59.000000001+23:59", "1970-01-01T00:00:00.000Z"};
    const std::string alphabet = std::string("0123456789-:.TtZz +x") + '\0' + '\xff';
    std::vector<std::string> corpus(seeds);
    for (const std::string& s : seeds) {
        for (std::size_t i = 0; i <= s.size(); ++i) {
            corpus.push_back(s.substr(0, i));                        // truncation
            if (i < s.size()) corpus.push_back(s.substr(0, i) + s.substr(i + 1));   // deletion
            for (char c : alphabet) {
                corpus.push_back(s.substr(0, i) + c + s.substr(i));  // insertion
                if (i < s.size()) {
                    std::string r = s;
                    r[i] = c;                                         // substitution
                    corpus.push_back(r);
                }
            }
        }
        // every two-digit field replaced by every value 00..99
        for (std::size_t pos : {5U, 8U, 11U, 14U, 17U}) {
            for (int v = 0; v < 100; ++v) {
                std::string r = s;
                r[pos] = static_cast<char>('0' + v / 10);
                r[pos + 1] = static_cast<char>('0' + v % 10);
                corpus.push_back(r);
            }
        }
    }
    return corpus;
}

template <typename D>
void differential(const std::vector<std::string>& corpus, const snicholls::parse_options& o, const char* label)
{
    std::size_t accepted = 0;
    for (const std::string& s : corpus) {
        typename D::rep expected{};
        const std::errc ref = reference_parse<D>(s, o, expected);
        const auto got = snicholls::parse_utc<D>(s, o);
        ASSERT_EQ(got.error(), ref) << label << " \"" << s << '"';
        if (ref == std::errc{}) {
            ASSERT_EQ(got->time_since_epoch().count(), expected) << label << " \"" << s << '"';
            ++accepted;
        }
    }
    EXPECT_GT(accepted, 100U) << label;   // the corpus exercises the success path too
}

TEST(UtcRfc3339Differential, MutationCorpusAgreesWithReferenceParser)
{
    const std::vector<std::string> corpus = mutation_corpus();
    ASSERT_GT(corpus.size(), 10000U);
    for (const snicholls::parse_options& o :
         {snicholls::parse_options{}, snicholls::parse_options::lenient(), snicholls::parse_options{.require_offset = false},
          snicholls::parse_options{.basic_offset = true}, snicholls::parse_options{.precision = snicholls::precision_policy::truncate},
          snicholls::parse_options{.leap_second = snicholls::leap_second_policy::clamp}}) {
        differential<nanoseconds>(corpus, o, "ns");
        differential<microseconds>(corpus, o, "us");
        differential<milliseconds>(corpus, o, "ms");
        differential<seconds>(corpus, o, "s");
        differential<minutes>(corpus, o, "min");
        differential<days>(corpus, o, "days");
        differential<duration<std::int64_t, std::ratio<7>>>(corpus, o, "7s");   // a period that does not divide a day
    }
}

// ---- Unix numbers as text ---------------------------------------------------------------------------------

TEST(UtcUnixText, ExactDecimalParsingInEveryUnit)
{
    EXPECT_EQ(snicholls::parse_unix<seconds>("1714486985")->unix_seconds(), 1714486985);
    EXPECT_EQ(snicholls::parse_unix<milliseconds>("1714486985123")->unix_millis(), 1714486985123);   // Massive
    EXPECT_EQ(snicholls::parse_unix<microseconds>("1714486985123456")->unix_micros(), 1714486985123456);
    EXPECT_EQ(snicholls::parse_unix<nanoseconds>("1714486985123456789")->unix_nanos(), 1714486985123456789);
    EXPECT_EQ(snicholls::parse_unix<seconds>("1714486985.123456789")->unix_nanos(), 1714486985123456789);   // no double
    EXPECT_EQ(snicholls::parse_unix<milliseconds>("1714486985123.456789")->unix_nanos(), 1714486985123456789);
    EXPECT_EQ(snicholls::parse_unix<seconds>("1714486985.5")->unix_millis(), 1714486985500);   // the original read 1.7 s
    EXPECT_EQ(snicholls::parse_unix<seconds>("-1.5")->unix_millis(), -1500);
    EXPECT_EQ(snicholls::parse_unix<seconds>("-0.000000001")->unix_nanos(), -1);
    EXPECT_EQ(snicholls::parse_unix<seconds>("-0")->unix_nanos(), 0);
    EXPECT_EQ(snicholls::parse_unix<seconds>("0007")->unix_seconds(), 7);
    EXPECT_EQ((snicholls::parse_unix<milliseconds, milliseconds>("1714486985123")->unix_millis()), 1714486985123);
    static_assert(snicholls::parse_unix<milliseconds>("1714486985123")->unix_millis() == 1714486985123);
}

TEST(UtcUnixText, RangeAndPrecision)
{
    EXPECT_EQ(*snicholls::parse_unix<nanoseconds>("9223372036854775807"), utc<>::max());
    EXPECT_EQ(*snicholls::parse_unix<nanoseconds>("-9223372036854775808"), utc<>::min());
    EXPECT_EQ(snicholls::parse_unix<nanoseconds>("9223372036854775808").error(), std::errc::result_out_of_range);
    EXPECT_EQ(snicholls::parse_unix<nanoseconds>("-9223372036854775809").error(), std::errc::result_out_of_range);
    EXPECT_EQ(snicholls::parse_unix<seconds>("9223372037").error(), std::errc::result_out_of_range);   // > 2262 in ns
    EXPECT_EQ(snicholls::parse_unix<seconds>("-9223372036.854775808")->unix_nanos(), utc<>::min().unix_nanos());
    EXPECT_EQ(snicholls::parse_unix<seconds>("-9223372036.854775809").error(), std::errc::result_out_of_range);
    EXPECT_EQ((snicholls::parse_unix<seconds, seconds>("-9223372036854775808")->unix_seconds()), std::numeric_limits<std::int64_t>::min());
    EXPECT_EQ((snicholls::parse_unix<seconds, seconds>("-9223372036854775808.5").error()), std::errc::result_out_of_range);
    EXPECT_EQ(snicholls::parse_unix<seconds>("99999999999999999999").error(), std::errc::result_out_of_range);
    EXPECT_EQ(snicholls::parse_unix<seconds>("1.0000000001").error(), std::errc::result_out_of_range);   // finer than ns
    EXPECT_EQ(snicholls::parse_unix<seconds>("1.0000000000")->unix_seconds(), 1);
    constexpr snicholls::parse_options truncate{.precision = snicholls::precision_policy::truncate};
    EXPECT_EQ(snicholls::parse_unix<seconds>("1.0000000019", truncate)->unix_nanos(), 1000000001);
    EXPECT_EQ(snicholls::parse_unix<seconds>("-1.0000000001", truncate)->unix_nanos(), -1000000001);   // floor
    EXPECT_EQ(snicholls::parse_unix<seconds>("-0.9999999999", truncate)->unix_nanos(), -1000000000);
    EXPECT_EQ((snicholls::parse_unix<milliseconds, seconds>("1500").error()), std::errc::result_out_of_range);
    EXPECT_EQ((snicholls::parse_unix<milliseconds, seconds>("1500", truncate)->unix_seconds()), 1);
    EXPECT_EQ((snicholls::parse_unix<milliseconds, seconds>("-1500", truncate)->unix_seconds()), -2);
}

TEST(UtcUnixText, SyntaxErrors)
{
    for (const char* s : {"", "-", "+1", "1.", ".5", "1e9", "0x10", " 1", "1 ", "1.5.5", "--1", "1-", "١"}) {
        EXPECT_EQ(snicholls::parse_unix<seconds>(std::string_view(s)).error(), std::errc::invalid_argument) << s;
    }
    utc<> t;
    const std::string_view s = "1714486985123,next";
    const auto r = snicholls::from_unix_chars<milliseconds>(s.data(), s.data() + s.size(), t);
    EXPECT_EQ(r.ec, std::errc{});
    EXPECT_EQ(r.ptr, s.data() + 13);
}

TEST(UtcUnixText, SyntaxIsReportedBeforeRange)
{
    EXPECT_EQ(snicholls::parse_unix<seconds>("99999999999999999999x").error(), std::errc::invalid_argument);
    EXPECT_EQ(snicholls::parse_unix<seconds>("99999999999999999999").error(), std::errc::result_out_of_range);
    EXPECT_EQ(parse_error("1600-01-01T00:00:00Zjunk"), std::errc::invalid_argument);
    EXPECT_EQ(parse_error("1600-01-01T00:00:00Z"), std::errc::result_out_of_range);
}

// Reference: std::regex for the syntax, exact 128-bit rational arithmetic for the value.
template <typename Unit, typename D>
std::errc reference_unix(const std::string& s, const snicholls::parse_options& o, typename D::rep& out)
{
    static const std::regex re(R"(^(-?)(\d+)(?:\.(\d+))?$)");
    std::smatch m;
    if (s.find('\0') != std::string::npos || !std::regex_match(s, m, re)) return std::errc::invalid_argument;
    // Contract: the integer part, in Unit, must fit in int64 (INT64_MIN included).
    int128 integer_part = 0;
    for (char c : std::string(m[2])) integer_part = integer_part * 10 + (c - '0');
    const int128 int64_limit = static_cast<int128>(std::numeric_limits<std::int64_t>::max()) + (m[1].length() != 0 ? 1 : 0);
    if (m[2].length() > 30 || integer_part > int64_limit) return std::errc::result_out_of_range;
    const std::string digits = std::string(m[2]) + std::string(m[3]);
    if (digits.size() > 36) return std::errc::result_out_of_range;   // the corpus stays below this
    int128 n = 0;
    for (char c : digits) n = n * 10 + (c - '0');
    if (m[1].length() != 0) n = -n;
    // value = n / 10^f Units = n * 10^(9 - u) / 10^f ns; ticks of D = floor(value / (ns per tick)).
    const int u = utc<Unit>::fraction_digits;
    const int f = static_cast<int>(m[3].length());
    int128 num = n;
    for (int i = 0; i < 9 - u; ++i) num *= 10;
    int128 den = 1;
    for (int i = 0; i < f; ++i) den *= 10;
    using period = typename D::period;
    den *= static_cast<int128>(1000000000) * period::num / period::den;   // ns per tick
    int128 q = num / den;
    const int128 r = num % den;
    if (r != 0 && num < 0) --q;
    if (r != 0 && o.precision == snicholls::precision_policy::reject) return std::errc::result_out_of_range;
    if (q < std::numeric_limits<typename D::rep>::min() || q > std::numeric_limits<typename D::rep>::max())
        return std::errc::result_out_of_range;
    out = static_cast<typename D::rep>(q);
    return std::errc{};
}

template <typename Unit, typename D>
void unix_differential(const std::vector<std::string>& corpus, const snicholls::parse_options& o)
{
    for (const std::string& s : corpus) {
        if (s.size() > 30) continue;
        typename D::rep expected{};
        const std::errc ref = reference_unix<Unit, D>(s, o, expected);
        const auto got = snicholls::parse_unix<Unit, D>(s, o);
        ASSERT_EQ(got.error(), ref) << '"' << s << '"';
        if (ref == std::errc{}) {
            ASSERT_EQ(got->time_since_epoch().count(), expected) << '"' << s << '"';
        }
    }
}

TEST(UtcUnixTextDifferential, MutationCorpusAgreesWithReferenceParser)
{
    const std::vector<std::string> seeds = {"1714486985", "-1.5", "1714486985123.456789", "9223372036854775807",
                                            "-9223372036854775808", "0.0000000001", "-0.9999999999", "-9223372036.854775808",
                                            "1500", "-0"};
    const std::string alphabet = std::string("0159-.x") + '\0';
    std::vector<std::string> corpus(seeds);
    for (const std::string& s : seeds)
        for (std::size_t i = 0; i <= s.size(); ++i) {
            corpus.push_back(s.substr(0, i));
            if (i < s.size()) corpus.push_back(s.substr(0, i) + s.substr(i + 1));
            for (char c : alphabet) {
                corpus.push_back(s.substr(0, i) + c + s.substr(i));
                if (i < s.size()) {
                    std::string r = s;
                    r[i] = c;
                    corpus.push_back(r);
                }
            }
        }
    for (const snicholls::parse_options& o : {snicholls::parse_options{}, snicholls::parse_options::lenient()}) {
        unix_differential<seconds, nanoseconds>(corpus, o);
        unix_differential<milliseconds, nanoseconds>(corpus, o);
        unix_differential<nanoseconds, nanoseconds>(corpus, o);
        unix_differential<seconds, milliseconds>(corpus, o);
        unix_differential<milliseconds, seconds>(corpus, o);
        unix_differential<nanoseconds, seconds>(corpus, o);
        unix_differential<microseconds, minutes>(corpus, o);
    }
}

// ---- dates ------------------------------------------------------------------------------------------------

TEST(UtcRfc3339Date, FormatAndParse)
{
    EXPECT_EQ(snicholls::to_string(2024y / April / 30), "2024-04-30");
    EXPECT_EQ(snicholls::to_text(0y / January / 1).view(), "0000-01-01");
    EXPECT_EQ(*snicholls::parse_date("2024-02-29"), 2024y / February / 29);
    EXPECT_EQ(*snicholls::parse_date("0000-01-01"), year(0) / January / 1);
    for (const char* s : {"2023-02-29", "2024-13-01", "2024-1-01", "2024-01-01T00:00:00Z", "", "2024-01-01 ", "20240101"})
        EXPECT_EQ(snicholls::parse_date(std::string_view(s)).error(), std::errc::invalid_argument) << s;
    char buf[10];
    EXPECT_EQ(snicholls::to_chars(buf, buf + 9, 2024y / April / 30).ec, std::errc::value_too_large);
    EXPECT_EQ(snicholls::to_chars(buf, buf + 10, 2024y / February / 30).ec, std::errc::invalid_argument);
    EXPECT_EQ(snicholls::to_chars(buf, buf + 10, year(-1) / January / 1).ec, std::errc::value_too_large);
    EXPECT_EQ(snicholls::to_chars(buf, buf + 10, year(10000) / January / 1).ec, std::errc::value_too_large);
    EXPECT_EQ(snicholls::to_string(year(10000) / January / 1), "");
    static_assert(snicholls::to_text(2024y / April / 30).view() == "2024-04-30");
    // Every valid date of 0000..9999 round trips.
    for (auto d = sys_days{year(0) / January / 1}; d <= sys_days{year(9999) / December / 31}; d += days(1)) {
        const year_month_day ymd(d);
        ASSERT_EQ(*snicholls::parse_date(snicholls::to_text(ymd).view()), ymd);
    }
}

// ---- parse_result and the literal -------------------------------------------------------------------------

TEST(UtcParseResult, BehavesLikeExpected)
{
    const auto ok = snicholls::parse_utc("2024-04-30T14:23:05Z");
    const auto bad = snicholls::parse_utc("2024-04-31T14:23:05Z");
    const auto range = snicholls::parse_utc("1600-01-01T00:00:00Z");
    EXPECT_TRUE(ok.has_value() && static_cast<bool>(ok) && ok.error() == std::errc{});
    EXPECT_EQ(ok.value().unix_seconds(), 1714486985);
    EXPECT_EQ(ok->unix_seconds(), 1714486985);
    EXPECT_FALSE(bad.has_value() || static_cast<bool>(bad));
    EXPECT_EQ(bad.error(), std::errc::invalid_argument);
    EXPECT_EQ(bad.value_or(utc<>::max()), utc<>::max());
    EXPECT_EQ(ok.value_or(utc<>::max()).unix_seconds(), 1714486985);
    try {
        (void)bad.value();
        ADD_FAILURE() << "value() did not throw";
    } catch (const snicholls::utc_error& e) {
        EXPECT_EQ(e.code(), std::errc::invalid_argument);
        EXPECT_NE(std::string_view(e.what()).find("RFC 3339"), std::string_view::npos);
    }
    try {
        (void)range.value();
        ADD_FAILURE() << "value() did not throw";
    } catch (const snicholls::utc_error& e) {
        EXPECT_EQ(e.code(), std::errc::result_out_of_range);
    }
    static_assert(noexcept(snicholls::parse_utc("")) && noexcept(snicholls::parse_unix<seconds>("")) && noexcept(snicholls::parse_date("")));
    static_assert(std::is_trivially_copyable_v<snicholls::parse_result<utc<>>>);
}

TEST(UtcLiteral, CheckedAtCompileTime)
{
    using namespace snicholls::literals;
    constexpr utc<> t = "2024-04-30T14:23:05.123456789Z"_utc;
    static_assert(t.unix_nanos() == 1714486985123456789);
    static_assert("2024-04-30T10:23:05-04:00"_utc == "2024-04-30T14:23:05Z"_utc);
    static_assert(snicholls::to_text("1999-12-31T23:59:59.5Z"_utc, 1).view() == "1999-12-31T23:59:59.5Z");
    // "2024-02-30T00:00:00Z"_utc does not compile.
    EXPECT_EQ(t.unix_millis(), 1714486985123);
}

// ---- portability fixtures (tests/fixtures/time/, from Python's datetime and exact integers) ---------------------------
//
// The same files are read by these same tests on every platform; see scripts/generate_time_fixtures.py.

template <typename F>
void with_resolution(const std::string& res, F&& f)
{
    if (res == "s") f(std::type_identity<seconds>{});
    else if (res == "ms") f(std::type_identity<milliseconds>{});
    else if (res == "us") f(std::type_identity<microseconds>{});
    else if (res == "ns") f(std::type_identity<nanoseconds>{});
    else FAIL() << "unknown resolution " << res;
}

TEST(UtcRfc3339Fixtures, FormatMatchesPythonAtEveryResolutionAndPrecision)
{
    const auto rows = snicholls::test::read_time_fixture("rfc3339_format.tsv");
    for (const auto& row : rows) {
        const auto& f = row.fields;
        ASSERT_EQ(f.size(), 4U) << "line " << row.line;
        with_resolution(f[0], [&]<typename D>(std::type_identity<D>) {
            const utc<D> t(sys_time<D>(D(snicholls::test::to_i64(f[1]))));
            const int precision = f[2] == "-" ? utc<D>::fraction_digits : std::stoi(f[2]);
            EXPECT_EQ(snicholls::to_text(t, precision).view(), f[3]) << "line " << row.line;
            if (f[2] == "-") {
                // And back: the canonical text parses to the same ticks.
                const auto back = snicholls::parse_utc<D>(f[3]);
                ASSERT_TRUE(back) << "line " << row.line;
                EXPECT_EQ(back->time_since_epoch().count(), t.time_since_epoch().count()) << "line " << row.line;
            }
        });
    }
    EXPECT_GT(rows.size(), 1000U);
}

TEST(UtcRfc3339Fixtures, ParseMatchesPythonUnderStrictAndLenientOptions)
{
    const auto rows = snicholls::test::read_time_fixture("rfc3339_parse.tsv");
    for (const auto& row : rows) {
        const auto& f = row.fields;
        ASSERT_EQ(f.size(), 4U) << "line " << row.line;
        const snicholls::parse_options o = f[1] == "strict" ? snicholls::parse_options{} : snicholls::parse_options::lenient();
        with_resolution(f[0], [&]<typename D>(std::type_identity<D>) {
            const auto r = snicholls::parse_utc<D>(f[2], o);
            const std::string got = r ? std::to_string(r->time_since_epoch().count()) : snicholls::test::errc_name(r.error());
            EXPECT_EQ(got, f[3]) << "line " << row.line << ": " << f[0] << ' ' << f[1] << " \"" << f[2] << '"';
        });
    }
    EXPECT_GT(rows.size(), 900U);
}

TEST(UtcUnixTextFixtures, ExactFloorMatchesPythonsIntegerArithmetic)
{
    const auto rows = snicholls::test::read_time_fixture("unix_text.tsv");
    for (const auto& row : rows) {
        const auto& f = row.fields;
        ASSERT_EQ(f.size(), 5U) << "line " << row.line;
        const snicholls::parse_options o = f[2] == "strict" ? snicholls::parse_options{} : snicholls::parse_options::lenient();
        with_resolution(f[0], [&]<typename Unit>(std::type_identity<Unit>) {
            with_resolution(f[1], [&]<typename D>(std::type_identity<D>) {
                const auto r = snicholls::parse_unix<Unit, D>(f[3], o);
                const std::string got = r ? std::to_string(r->time_since_epoch().count()) : snicholls::test::errc_name(r.error());
                EXPECT_EQ(got, f[4]) << "line " << row.line << ": " << f[0] << " -> " << f[1] << ' ' << f[2] << " \"" << f[3] << '"';
            });
        });
    }
    EXPECT_GT(rows.size(), 1000U);
}

} // namespace
