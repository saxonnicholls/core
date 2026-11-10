//
//  utc_format_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/utc_format.hpp>.
//

// interfaces/utc/utc_format.hpp: std::formatter<snicholls::utc<D>>.

#include <chrono>
#include <format>
#include <locale>
#include <random>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <snicholls/core/time/utc_format.hpp>

namespace {

using namespace std::chrono;
using snicholls::utc;

TEST(UtcFormat, EmptySpecIsTheRfc3339FastPath)
{
    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    EXPECT_EQ(std::format("{}", t), "2024-04-30T14:23:05.123456789Z");
    EXPECT_EQ(std::format("[{}]", utc<milliseconds>::from_unix_millis(-1)), "[1969-12-31T23:59:59.999Z]");
    EXPECT_EQ(std::format("{0} {0}", utc<seconds>{}), "1970-01-01T00:00:00Z 1970-01-01T00:00:00Z");
    std::mt19937_64 rng(11);
    for (int i = 0; i < 10000; ++i) {
        const utc<> r = utc<>::from_unix_nanos(static_cast<std::int64_t>(rng()));
        ASSERT_EQ(std::format("{}", r), snicholls::to_string(r));
    }
}

TEST(UtcFormat, ChronoSpecsAreForwardedToTheStandardFormatter)
{
    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    EXPECT_EQ(std::format("{:%F}", t), "2024-04-30");
    EXPECT_EQ(std::format("{:%a %d %b %Y}", t), "Tue 30 Apr 2024");
    EXPECT_EQ(std::format("{:%H:%M}", t), "14:23");
    EXPECT_EQ(std::format("{:%FT%TZ}", t), std::format("{:%FT%TZ}", t.sys_time()));
    EXPECT_EQ(std::format("{:%T}", utc<milliseconds>(floor<milliseconds>(t.sys_time()))), "14:23:05.123");
}

TEST(UtcFormat, OutOfRangeYearsThrowFormatError)
{
    const auto far = utc<seconds>(sys_days{year(12000) / January / 1});
    EXPECT_THROW((void)std::format("{}", far), std::format_error);
    EXPECT_NO_THROW((void)std::format("{:%Y}", far));   // the chrono formatter handles it
}

// ---- portability: no text form depends on the locale -------------------------------------------------------------

struct odd_punctuation : std::numpunct<char>
{
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '\''; }
    std::string do_grouping() const override { return "\3"; }
};

TEST(UtcFormatPortability, NoTextFormDependsOnTheGlobalOrStreamLocale)
{
    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    const std::locale odd(std::locale::classic(), new odd_punctuation);
    const std::locale previous = std::locale::global(odd);
    std::ostringstream os;
    os.imbue(odd);
    os << t;
    const std::string streamed = os.str();
    const std::string formatted = std::format("{}", t);
    const std::string chrono_spec = std::format("{:%F %T}", t);    // no L: the C locale, by the standard
    const std::string names = std::format("{:%a %d %b %Y}", t);
    const std::string text = snicholls::to_string(t);
    std::locale::global(previous);
    EXPECT_EQ(streamed, "2024-04-30T14:23:05.123456789Z");
    EXPECT_EQ(formatted, "2024-04-30T14:23:05.123456789Z");
    EXPECT_EQ(text, "2024-04-30T14:23:05.123456789Z");
    EXPECT_EQ(chrono_spec, "2024-04-30 14:23:05.123456789");
    EXPECT_EQ(names, "Tue 30 Apr 2024");
}

} // namespace
