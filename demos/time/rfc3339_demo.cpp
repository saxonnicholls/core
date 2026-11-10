//
//  rfc3339_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/rfc3339.hpp>.
//

// Demo: interfaces/utc/rfc3339.hpp - allocation-free RFC 3339 text for snicholls::utc, exact both ways.

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#include <snicholls/core/time/rfc3339.hpp>

// Hides a value from the optimiser, so the demo runs the library at run time instead of letting the compiler fold
// a constant expression.
#if defined(_MSC_VER)
#define SNICHOLLS_DEMO_NOINLINE __declspec(noinline)
#else
#define SNICHOLLS_DEMO_NOINLINE [[gnu::noinline]]
#endif
template <typename T>
SNICHOLLS_DEMO_NOINLINE T opaque(T v)
{
    return v;
}

//! [demo]
int main()
{
    using namespace std::chrono;
    using namespace std::chrono_literals;
    using snicholls::utc;
    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << '\n';
        failures += !ok;
    };
    const auto show = [](std::string_view label, std::errc ec) {
        std::cout << "    " << label << " -> " << (ec == std::errc{} ? "ok" : std::make_error_code(ec).message()) << '\n';
    };

    // ---- formatting ----
    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    char buf[snicholls::max_chars_v<nanoseconds>];
    const auto [end, ec] = snicholls::to_chars(buf, buf + sizeof buf, t);
    std::cout << "to_chars: " << std::string_view(buf, static_cast<std::size_t>(end - buf)) << '\n';
    check("to_chars writes 30 chars, ns precision", ec == std::errc{} && end - buf == 30);
    check("to_text(t).view()", snicholls::to_text(t).view() == "2024-04-30T14:23:05.123456789Z");
    check("precision 0, 3, 1 (floored)", snicholls::to_string(t, 0) == "2024-04-30T14:23:05Z" &&
                                             snicholls::to_string(t, 3) == "2024-04-30T14:23:05.123Z" &&
                                             snicholls::to_string(t, 1) == "2024-04-30T14:23:05.1Z");
    check("coarser types print their own digits", snicholls::to_string(utc<milliseconds>(floor<milliseconds>(t.sys_time()))) == "2024-04-30T14:23:05.123Z" &&
                                                      snicholls::to_string(utc<seconds>{}) == "1970-01-01T00:00:00Z" &&
                                                      snicholls::to_string(utc<days>{}) == "1970-01-01T00:00:00Z");
    check("zero-padded when finer than the type", snicholls::to_string(utc<seconds>{}, 2) == "1970-01-01T00:00:00.00Z");
    check("before 1970 is floored", snicholls::to_string(utc<>::from_unix_nanos(-1), 3) == "1969-12-31T23:59:59.999Z");
    check("buffer too small: value_too_large", snicholls::to_chars(buf, buf + 29, t).ec == std::errc::value_too_large);
    check("precision 10: invalid_argument", snicholls::to_chars(buf, buf + sizeof buf, t, 10).ec == std::errc::invalid_argument);
    const auto year_12000 = utc<seconds>(sys_days{year(12000) / January / 1});
    const auto year_minus_1 = utc<seconds>(sys_days{year(-1) / January / 1});
    check("years outside 0000..9999: value_too_large, empty text",
          snicholls::to_chars(buf, buf + sizeof buf, year_12000).ec == std::errc::value_too_large &&
              snicholls::to_chars(buf, buf + sizeof buf, opaque(utc<microseconds>(year_minus_1))).ec == std::errc::value_too_large &&
              !snicholls::to_text(year_12000) && snicholls::to_string(year_12000).empty() &&
              snicholls::to_chars(buf, buf + sizeof buf, utc<days>::max()).ec == std::errc::value_too_large);
    std::ostringstream os;
    os << t;
    std::ostringstream bad;
    bad << year_12000;
    check("operator<< writes the text; failbit when it cannot", os.str() == snicholls::to_string(t) && bad.fail());
    const snicholls::rfc3339_text epoch_text = snicholls::to_text(utc<seconds>{});   // the view below refers into it
    const std::string_view view = epoch_text;
    check("rfc3339_text converts to string_view", view == "1970-01-01T00:00:00Z");

    // ---- parsing (strict RFC 3339 by default) ----
    const auto r = snicholls::parse_utc("2024-04-30T10:23:05.123456789-04:00");
    std::cout << "parse_utc(\"2024-04-30T10:23:05.123456789-04:00\") = " << *r << '\n';
    check("offsets are applied", r && *r == t);
    check("t, z and space separators", snicholls::parse_utc("2024-04-30t14:23:05.123456789z").value() == t &&
                                           snicholls::parse_utc("2024-04-30 14:23:05.123456789Z").value() == t);
    for (const char* s : {"2024-04-30T14:23Z", "2024-02-30T00:00:00Z", "2024-04-30T14:23:05", "2024-04-30T14:23:05+0100",
                          "2024-04-30T14:23:05.Z", "2024-04-30T14:23:05+24:00", "2024-04-30T14:23:05Zjunk",
                          "1600-01-01T00:00:00Z", "2024-04-30T14:23:05.1234567891Z", "2016-12-31T23:59:60Z",
                          "2016-12-31T23:58:60Z", "2024-04-30T14:23:05+01", "2024-04-30T14:23:05+01:0"})
        show(s, snicholls::parse_utc(std::string_view(s)).error());
    check("missing seconds rejected (never silently zeroed)", snicholls::parse_utc("2024-04-30T14:23Z").error() == std::errc::invalid_argument);
    check("invalid date rejected", snicholls::parse_utc("2024-02-30T00:00:00Z").error() == std::errc::invalid_argument);
    check("missing offset rejected (strict)", snicholls::parse_utc("2024-04-30T14:23:05").error() == std::errc::invalid_argument);
    check("basic offset rejected (strict)", snicholls::parse_utc("2024-04-30T14:23:05+0100").error() == std::errc::invalid_argument);
    check("truncated offsets rejected", snicholls::parse_utc("2024-04-30T14:23:05+01").error() == std::errc::invalid_argument &&
                                            snicholls::parse_utc("2024-04-30T14:23:05+01:0").error() == std::errc::invalid_argument);
    check("before 1677 is out of range for ns", snicholls::parse_utc("1600-01-01T00:00:00Z").error() == std::errc::result_out_of_range);
    check("10 fraction digits: out of range for ns", snicholls::parse_utc("2024-04-30T14:23:05.1234567891Z").error() == std::errc::result_out_of_range);
    check("leap second rejected by default; :60 elsewhere is invalid",
          snicholls::parse_utc("2016-12-31T23:59:60Z").error() == std::errc::result_out_of_range &&
              snicholls::parse_utc("2016-12-31T23:58:60Z").error() == std::errc::invalid_argument);

    const auto lenient = opaque(snicholls::parse_options::lenient());
    check("lenient: truncate excess digits", snicholls::parse_utc("2024-04-30T14:23:05.1234567891Z", lenient)->unix_nanos() == 1714486985123456789);
    check("lenient: leap second clamps like utc_clock::to_sys",
          *snicholls::parse_utc("2016-12-31T23:59:60.5Z", lenient) == utc<>(sys_days{2017y / January / 1}) - 1ns &&
              *snicholls::parse_utc<milliseconds>("2016-12-31T23:59:60Z", lenient) == utc<milliseconds>(sys_days{2017y / January / 1}) - 1ms &&
              *snicholls::parse_utc<minutes>("2016-12-31T23:59:60Z", lenient) == utc<minutes>(sys_days{2017y / January / 1}) - 1min &&
              *snicholls::parse_utc<minutes>("2016-12-31T23:59:30Z", lenient) == utc<minutes>(sys_days{2017y / January / 1}) - 1min);
    using seven_seconds = duration<std::int64_t, std::ratio<7>>;   // does not divide a day: the floor is already earlier
    check("lenient: leap second into 7-second ticks floors",
          snicholls::parse_utc<seven_seconds>(opaque(std::string_view("2016-12-31T23:59:60Z")), lenient)->sys_time() ==
              floor<seven_seconds>(sys_seconds{sys_days{2016y / December / 31} + 86399s}));
    check("lenient: no offset = UTC, basic offset", snicholls::parse_utc("2024-04-30 14:23:05", lenient)->unix_seconds() == 1714486985 &&
                                                       snicholls::parse_utc("2024-04-30T15:23:05+0100", lenient)->unix_seconds() == 1714486985);
    check("excess precision into coarser types", snicholls::parse_utc<seconds>("2024-04-30T14:23:05.5Z").error() == std::errc::result_out_of_range &&
                                                     snicholls::parse_utc<seconds>("2024-04-30T14:23:05.5Z", lenient)->unix_seconds() == 1714486985);
    check("ns boundaries are exact", *snicholls::parse_utc("1677-09-21T00:12:43.145224192Z") == utc<>::min() &&
                                         *snicholls::parse_utc("2262-04-11T23:47:16.854775807Z") == utc<>::max() &&
                                         !snicholls::parse_utc("2262-04-11T23:47:16.854775808Z"));

    // from_chars: a prefix, like std::from_chars; untouched on failure.
    const std::string_view line = "2024-04-30T14:23:05Z,EURUSD,1.0856";
    utc<> parsed;
    const auto fc = snicholls::from_chars(line.data(), line.data() + line.size(), parsed);
    check("from_chars stops after the timestamp", fc.ec == std::errc{} && fc.ptr == line.data() + 20 && parsed.unix_seconds() == 1714486985);
    check("parse_utc requires the whole string", snicholls::parse_utc(line).error() == std::errc::invalid_argument);
    check("too short: invalid_argument", snicholls::parse_utc("2024").error() == std::errc::invalid_argument);
    check("date and time syntax errors", snicholls::parse_utc(opaque(std::string_view("2024/04/30T14:23:05Z"))).error() == std::errc::invalid_argument &&
                                             snicholls::parse_utc(opaque(std::string_view("2024-04-30T14-23-05Z"))).error() == std::errc::invalid_argument);
    // A 32-bit seconds count ends in January 2038 (the Y2038 problem): out of range, never wrapped.
    using int32_seconds = duration<std::int32_t>;
    check("utc<int32 seconds> cannot hold 2040", snicholls::parse_utc<int32_seconds>(opaque(std::string_view("2040-01-01T00:00:00Z"))).error() ==
                                                     std::errc::result_out_of_range &&
                                                 snicholls::parse_utc<int32_seconds>(opaque(std::string_view("2038-01-19T03:14:07Z")))->unix_seconds() == 2147483647);

    // Unix numbers as text: exact, no double.
    check("parse_unix<ms>(\"1714486985123\")", snicholls::parse_unix<milliseconds>("1714486985123")->unix_millis() == 1714486985123);
    check("parse_unix<s>(\"1714486985.123456789\") is exact", snicholls::parse_unix<seconds>("1714486985.123456789")->unix_nanos() == 1714486985123456789);
    check("negative fractions floor", snicholls::parse_unix<seconds>(opaque(std::string_view("-1.5")))->unix_millis() == -1500 &&
                                          snicholls::parse_unix<seconds>("-0.0000000001", lenient)->unix_nanos() == -1 &&
                                          snicholls::parse_unix<seconds>("-2")->unix_seconds() == -2);
    check("INT64 limits in ns", *snicholls::parse_unix<nanoseconds>("-9223372036854775808") == utc<>::min() &&
                                    snicholls::parse_unix<nanoseconds>("9223372036854775808").error() == std::errc::result_out_of_range);
    check("seconds INT64_MIN into utc<s>; one more is out of range",
          (snicholls::parse_unix<seconds, seconds>("-9223372036854775808")->unix_seconds() == std::numeric_limits<std::int64_t>::min()) &&
              (snicholls::parse_unix<seconds, seconds>("-9223372036854775808.5").error() == std::errc::result_out_of_range));
    check("unix syntax errors", snicholls::parse_unix<seconds>("1e9").error() == std::errc::invalid_argument &&
                                    snicholls::parse_unix<seconds>("1.").error() == std::errc::invalid_argument &&
                                    snicholls::parse_unix<seconds>("-").error() == std::errc::invalid_argument &&
                                    snicholls::parse_unix<seconds>("").error() == std::errc::invalid_argument);
    check("excess unix digits rejected", snicholls::parse_unix<seconds>("1.0000000001").error() == std::errc::result_out_of_range);

    // Dates.
    check("dates both ways", snicholls::to_string(2024y / April / 30) == "2024-04-30" && *snicholls::parse_date(opaque(std::string_view("2024-02-29"))) == 2024y / February / 29 &&
                                 snicholls::to_text(2024y / April / 30).view() == "2024-04-30");
    check("bad dates", snicholls::parse_date(opaque(std::string_view("2023-02-29"))).error() == std::errc::invalid_argument &&
                           snicholls::parse_date(opaque(std::string_view("2024-01-01x"))).error() == std::errc::invalid_argument &&
                           snicholls::parse_date(opaque(std::string_view("2024/01/01"))).error() == std::errc::invalid_argument &&
                           snicholls::to_chars(buf, buf + 10, 2024y / February / 30).ec == std::errc::invalid_argument &&
                           snicholls::to_chars(buf, buf + 10, year(10000) / January / 1).ec == std::errc::value_too_large &&
                           snicholls::to_string(year(10000) / January / 1).empty());

    // parse_result: has_value / error / value_or / value() throwing snicholls::utc_error.
    const auto bad_result = snicholls::parse_utc("not a time");
    const auto range_result = snicholls::parse_utc("1600-01-01T00:00:00Z");
    check("value_or", bad_result.value_or(utc<>::max()) == utc<>::max() && r.value_or(utc<>{}) == t);
    try {
        (void)bad_result.value();
        check("value() throws", false);
    } catch (const snicholls::utc_error& e) {
        std::cout << "    value() threw: " << e.what() << '\n';
        check("value() throws snicholls::utc_error(invalid_argument)", e.code() == std::errc::invalid_argument);
    }
    try {
        (void)range_result.value();
        check("value() throws", false);
    } catch (const snicholls::utc_error& e) {
        check("value() throws snicholls::utc_error(result_out_of_range)", e.code() == std::errc::result_out_of_range);
    }

    // Compile-time literal.
    using namespace snicholls::literals;
    constexpr utc<> literal = "2024-04-30T14:23:05.123456789Z"_utc;
    static_assert(literal.unix_nanos() == 1714486985123456789);
    check("\"...\"_utc is checked at compile time", literal == t);

    std::cout << (failures == 0 ? "OK\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
//! [demo]
