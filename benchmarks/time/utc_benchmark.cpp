//
//  utc_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Benchmark for <snicholls/core/time/rfc3339.hpp>.
//

// Throughput of snicholls::utc text and calendar conversions versus the alternatives (Google Benchmark).
//
// Like-for-like rules (enforced by verify_like_for_like() before anything is timed):
//  * every formatter writes the byte-identical 30-character RFC 3339 text "YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ" for the
//    same 4096 instants into a caller-provided buffer (except the *_string variants, which all return std::string);
//  * every parser reads the same pre-built std::string inputs and must produce identical nanosecond values;
//  * calendar variants must produce identical (y, m, d) / day counts.
// Only Release numbers mean anything (presets macos-release / ubuntu-release).
//
// Baselines:
//  * std_format_to_n      std::format_to_n("{:%FT%TZ}", sys_time<ns>) - the standard library's chrono formatter;
//  * gmtime_snprintf      gmtime_r + snprintf - what a typical logger does;
//  * gmtime_strftime      gmtime_r + strftime + the fraction by hand - the C library way;
//  * original_ostream     the original UTC::As() shape: a std::ostringstream per call (locale), the chrono
//                         formatter, then a std::string. (date::format itself is not vendored here; its
//                         implementation is the same ostream + locale + string pattern.)
//  * original_exact       the original parse_rfc3339_exact, copied verbatim (portable:: -> std::chrono::), i.e.
//                         the original's *fast* path; its generic fallback is slower still (up to ~25
//                         istringstreams per string);
//  * istream_get_time     std::istringstream + std::get_time + timegm (the original's generic-loop mechanism,
//                         one format);
//  * sscanf_timegm        sscanf + timegm.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <snicholls/core/time/civil.hpp>
#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/utc.hpp>

namespace {

using namespace std::chrono;
using snicholls::utc;

constexpr std::size_t count = 4096;
constexpr std::size_t text_size = 30;

const std::vector<utc<>>& instants()
{
    static const std::vector<utc<>> v = [] {
        std::mt19937_64 rng(20240430);
        const std::int64_t lo = utc<>(sys_days{2000y / January / 1}).unix_nanos();
        const std::int64_t hi = utc<>(sys_days{2030y / January / 1}).unix_nanos();
        std::uniform_int_distribution<std::int64_t> dist(lo, hi);
        std::vector<utc<>> out;
        for (std::size_t i = 0; i < count; ++i) out.push_back(utc<>::from_unix_nanos(dist(rng)));
        return out;
    }();
    return v;
}

const std::vector<std::string>& texts()
{
    static const std::vector<std::string> v = [] {
        std::vector<std::string> out;
        for (const utc<>& t : instants()) out.push_back(snicholls::to_string(t));
        return out;
    }();
    return v;
}

// ---- formatters (all write 30 chars at out) ---------------------------------------------------------------

inline void format_core(utc<> t, char* out) noexcept { (void)snicholls::to_chars(out, out + text_size, t); }

inline void format_std(utc<> t, char* out) { std::format_to_n(out, text_size, "{:%FT%TZ}", t.sys_time()); }

inline void split(utc<> t, std::tm& tm, long& ns) noexcept
{
    const std::time_t s = static_cast<std::time_t>(t.unix_seconds());
    ns = static_cast<long>(t.unix_nanos() - t.unix_seconds() * 1000000000);
    ::gmtime_r(&s, &tm);
}

inline void format_snprintf(utc<> t, char* out) noexcept
{
    std::tm tm{};
    long ns = 0;
    split(t, tm, ns);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d.%09ldZ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec, ns);
    std::copy_n(buf, text_size, out);
}

inline void format_strftime(utc<> t, char* out) noexcept
{
    std::tm tm{};
    long ns = 0;
    split(t, tm, ns);
    std::strftime(out, 20, "%Y-%m-%dT%H:%M:%S", &tm);
    out[19] = '.';
    for (int i = 28; i >= 20; --i, ns /= 10) out[i] = static_cast<char>('0' + ns % 10);
    out[29] = 'Z';
}

inline std::string format_original(utc<> t)
{
    std::ostringstream os;
    std::format_to(std::ostreambuf_iterator<char>(os), "{:%FT%TZ}", t.sys_time());
    return os.str();
}

// ---- parsers ----------------------------------------------------------------------------------------------

// The original implementation's parse_rfc3339_exact, verbatim except portable:: -> std::chrono::.
template <typename Duration>
static std::optional<std::chrono::sys_time<Duration>> parse_rfc3339_exact(const std::string& s)
{
    const auto is_digit = [&](size_t i) { return i < s.size() && s[i] >= '0' && s[i] <= '9'; };
    const auto num = [&](size_t pos, size_t n) {
        long long v = 0;
        for (size_t i = pos; i < pos + n; ++i) v = v * 10 + (s[i] - '0');
        return v;
    };
    if (s.size() < 19) return std::nullopt;
    static constexpr size_t digit_positions[] = {0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18};
    for (size_t i : digit_positions) {
        if (!is_digit(i)) return std::nullopt;
    }
    if (s[4] != '-' || s[7] != '-' || (s[10] != 'T' && s[10] != ' ') || s[13] != ':' || s[16] != ':') {
        return std::nullopt;
    }
    const long long Y = num(0, 4);
    const long long Mo = num(5, 2);
    const long long D = num(8, 2);
    const long long H = num(11, 2);
    const long long Mi = num(14, 2);
    const long long S = num(17, 2);
    if (H > 23 || Mi > 59 || S > 60) return std::nullopt;
    size_t pos = 19;
    std::chrono::nanoseconds frac{0};
    if (pos < s.size() && s[pos] == '.') {
        ++pos;
        const size_t start = pos;
        while (is_digit(pos)) ++pos;
        if (pos == start) return std::nullopt;
        long long ns = 0;
        size_t ndig = 0;
        for (size_t i = start; i < pos && ndig < 9; ++i, ++ndig) {
            ns = ns * 10 + (s[i] - '0');
        }
        while (ndig < 9) {
            ns *= 10;
            ++ndig;
        }
        frac = std::chrono::nanoseconds{ns};
    }
    std::chrono::seconds offset{0};
    if (pos < s.size()) {
        if (s[pos] == 'Z' && pos + 1 == s.size()) {
        } else if (s[pos] == '+' || s[pos] == '-') {
            const bool neg = (s[pos] == '-');
            ++pos;
            if (!(is_digit(pos) && is_digit(pos + 1))) return std::nullopt;
            const long long oh = num(pos, 2);
            pos += 2;
            if (pos < s.size() && s[pos] == ':') ++pos;
            if (!(is_digit(pos) && is_digit(pos + 1))) return std::nullopt;
            const long long om = num(pos, 2);
            pos += 2;
            if (pos != s.size() || oh > 23 || om > 59) return std::nullopt;
            const long long total = (oh * 3600 + om * 60) * (neg ? -1 : 1);
            offset = std::chrono::seconds{total};
        } else {
            return std::nullopt;
        }
    }
    const std::chrono::year_month_day ymd{std::chrono::year{static_cast<int>(Y)}, std::chrono::month{static_cast<unsigned>(Mo)},
                                          std::chrono::day{static_cast<unsigned>(D)}};
    if (!ymd.ok()) return std::nullopt;
    const auto tp_ns = std::chrono::sys_days{ymd} + std::chrono::hours{H} + std::chrono::minutes{Mi} +
                       std::chrono::seconds{std::min<long long>(S, 59)} + frac - offset;
    return std::chrono::time_point_cast<Duration>(tp_ns);
}

inline std::int64_t parse_core(const std::string& s) noexcept
{
    utc<> t;
    (void)snicholls::from_chars(s.data(), s.data() + s.size(), t);
    return t.unix_nanos();
}

inline std::int64_t parse_original(const std::string& s)
{
    return parse_rfc3339_exact<nanoseconds>(s)->time_since_epoch().count();
}

inline std::int64_t parse_get_time(const std::string& s)
{
    std::istringstream in(s);
    std::tm tm{};
    in >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    char dot = 0, z = 0;
    long long frac = 0;
    in >> dot >> frac >> z;
    return static_cast<std::int64_t>(::timegm(&tm)) * 1000000000 + frac;
}

inline std::int64_t parse_sscanf(const std::string& s) noexcept
{
    std::tm tm{};
    long long frac = 0;
    std::sscanf(s.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d.%9lldZ", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min,
                &tm.tm_sec, &frac);
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    return static_cast<std::int64_t>(::timegm(&tm)) * 1000000000 + frac;
}

// ---- fairness check ---------------------------------------------------------------------------------------

int verify_like_for_like()
{
    int failures = 0;
    const auto fail = [&](const char* what, std::size_t i) {
        if (failures++ < 5) std::cerr << "like-for-like check failed: " << what << " at " << i << '\n';
    };
    for (std::size_t i = 0; i < count; ++i) {
        const utc<> t = instants()[i];
        const std::string& ref = texts()[i];
        char a[text_size], b[text_size], c[text_size], d[text_size + 1];
        format_core(t, a);
        format_std(t, b);
        format_snprintf(t, c);
        format_strftime(t, d);
        if (ref.size() != text_size || std::string(a, text_size) != ref) fail("snicholls::to_chars", i);
        if (std::string(b, text_size) != ref) fail("std::format_to_n", i);
        if (std::string(c, text_size) != ref) fail("gmtime_r + snprintf", i);
        if (std::string(d, text_size) != ref) fail("gmtime_r + strftime", i);
        if (format_original(t) != ref || snicholls::to_string(t) != ref || std::format("{:%FT%TZ}", t.sys_time()) != ref)
            fail("string formatters", i);
        const std::int64_t ns = t.unix_nanos();
        if (parse_core(ref) != ns) fail("snicholls::from_chars", i);
        if (parse_original(ref) != ns) fail("original parse_rfc3339_exact", i);
        if (parse_get_time(ref) != ns) fail("istream get_time", i);
        if (parse_sscanf(ref) != ns) fail("sscanf + timegm", i);
        const std::int64_t day = t.days().time_since_epoch().count();
        const snicholls::civil_date cd = snicholls::civil_from_days(day);
        const year_month_day ymd{t.days()};
        if (cd.year != static_cast<int>(ymd.year()) || cd.month != static_cast<unsigned>(ymd.month()) ||
            cd.day != static_cast<unsigned>(ymd.day()) || snicholls::days_from_civil(cd.year, cd.month, cd.day) != day)
            fail("calendar", i);
    }
    if (failures != 0) std::cerr << failures << " like-for-like failures: refusing to benchmark\n";
    return failures;
}

// ---- format: into a caller buffer ---------------------------------------------------------------------------

// Each variant is a functor type (not a function pointer), so every compiler inlines all of them alike.
template <typename Format>
void format_into_buffer(benchmark::State& state)
{
    const auto& in = instants();
    char out[text_size];
    for (auto _ : state) {
        for (const utc<>& t : in) {
            Format{}(t, out);
            benchmark::DoNotOptimize(out);
            benchmark::ClobberMemory();
        }
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * in.size()));
}

template <auto F>
struct as_functor
{
    void operator()(utc<> t, char* out) const { F(t, out); }
};

BENCHMARK(format_into_buffer<as_functor<format_core>>)->Name("format_core_to_chars");
BENCHMARK(format_into_buffer<as_functor<format_std>>)->Name("format_std_format_to_n");
BENCHMARK(format_into_buffer<as_functor<format_snprintf>>)->Name("format_gmtime_snprintf");
BENCHMARK(format_into_buffer<as_functor<format_strftime>>)->Name("format_gmtime_strftime");

// ---- format: returning std::string ------------------------------------------------------------------------

void format_string_core(benchmark::State& state)
{
    for (auto _ : state)
        for (const utc<>& t : instants()) benchmark::DoNotOptimize(snicholls::to_string(t));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * count));
}

void format_string_std(benchmark::State& state)
{
    for (auto _ : state)
        for (const utc<>& t : instants()) benchmark::DoNotOptimize(std::format("{:%FT%TZ}", t.sys_time()));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * count));
}

void format_string_original_ostream(benchmark::State& state)
{
    for (auto _ : state)
        for (const utc<>& t : instants()) benchmark::DoNotOptimize(format_original(t));
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * count));
}

BENCHMARK(format_string_core);
BENCHMARK(format_string_std);
BENCHMARK(format_string_original_ostream);

// ---- parse ------------------------------------------------------------------------------------------------

template <std::int64_t (*Parse)(const std::string&)>
void parse_text(benchmark::State& state)
{
    const auto& in = texts();
    for (auto _ : state) {
        std::int64_t sum = 0;
        for (const std::string& s : in) sum += Parse(s);
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * in.size()));
}

BENCHMARK(parse_text<parse_core>)->Name("parse_core_from_chars");
BENCHMARK(parse_text<parse_original>)->Name("parse_original_exact");
BENCHMARK(parse_text<parse_sscanf>)->Name("parse_sscanf_timegm");
BENCHMARK(parse_text<parse_get_time>)->Name("parse_istream_get_time");

// ---- calendar ---------------------------------------------------------------------------------------------

void civil_core(benchmark::State& state)
{
    for (auto _ : state) {
        std::int64_t sum = 0;
        for (const utc<>& t : instants()) {
            const snicholls::civil_date c = snicholls::civil_from_days(t.days().time_since_epoch().count());
            sum += c.year + c.month + c.day;
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * count));
}

void civil_std(benchmark::State& state)
{
    for (auto _ : state) {
        std::int64_t sum = 0;
        for (const utc<>& t : instants()) {
            const year_month_day c{t.days()};
            sum += static_cast<int>(c.year()) + static_cast<unsigned>(c.month()) + static_cast<unsigned>(c.day());
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * count));
}

BENCHMARK(civil_core)->Name("calendar_core_civil_from_days");
BENCHMARK(civil_std)->Name("calendar_std_year_month_day");

// ---- Unix ms -> instant -> Unix ms (zero overhead versus raw std::chrono) ------------------------------------

void unix_ms_core(benchmark::State& state)
{
    for (auto _ : state) {
        std::int64_t sum = 0;
        for (const utc<>& t : instants()) sum += utc<>::from_unix_millis(t.unix_nanos() / 1000000 + sum % 7).unix_millis();
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * count));
}

void unix_ms_std(benchmark::State& state)
{
    for (auto _ : state) {
        std::int64_t sum = 0;
        for (const utc<>& t : instants()) {
            const sys_time<nanoseconds> tp{floor<nanoseconds>(milliseconds(t.unix_nanos() / 1000000 + sum % 7))};
            sum += floor<milliseconds>(tp.time_since_epoch()).count();
        }
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * count));
}

BENCHMARK(unix_ms_core)->Name("unix_ms_core");
BENCHMARK(unix_ms_std)->Name("unix_ms_std_chrono");

} // namespace

int main(int argc, char** argv)
{
    if (verify_like_for_like() != 0) return 1;
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
