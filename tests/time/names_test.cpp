//
//  names_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/time/names.hpp>: every month and weekday name and abbreviation, read as attributes and
//  through the std::chrono helpers, at compile time; the JSON export.
//

#include <snicholls/core/time/names.hpp>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <sstream>
#include <string_view>

namespace {

using namespace std::chrono;
using snicholls::day_of_week;
using snicholls::month_of_year;

static_assert(snicholls::month_name(November) == "November" && snicholls::month_abbreviation(September) == "Sep");
static_assert(snicholls::weekday_name(Sunday) == "Sunday" && snicholls::weekday_abbreviation(Thursday) == "Thu");
static_assert(snicholls::month_name(month{13}).empty() && snicholls::weekday_name(weekday{9}).empty());
static_assert(snicholls::month_abbreviation(month{0}).empty() && snicholls::weekday_abbreviation(weekday{8}).empty());
static_assert(snicholls::to_day_of_week(Sunday) == day_of_week::sunday);   // ISO: Sunday is 7, not 0
static_assert(snicholls::to_month_of_year(January) == month_of_year::january);
static_assert(snicholls::attributes_of(month_of_year::may).get<snicholls::label>().text == "May");
static_assert(snicholls::has_attribute_v<snicholls::abbreviation, day_of_week>);
static_assert(snicholls::enum_name(day_of_week::friday) == "friday");
static_assert(noexcept(snicholls::month_name(January)));

constexpr std::array<std::string_view, 12> months{"January", "February", "March",     "April",   "May",      "June",
                                                  "July",    "August",   "September", "October", "November", "December"};
constexpr std::array<std::string_view, 7> weekdays{"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"};

TEST(TimeNames, EveryMonthHasItsNameAndThreeLetterAbbreviation)
{
    for (unsigned m = 1; m <= 12; ++m) {
        EXPECT_EQ(snicholls::month_name(month{m}), months[m - 1]);
        EXPECT_EQ(snicholls::month_abbreviation(month{m}), months[m - 1].substr(0, 3));
    }
}

TEST(TimeNames, EveryWeekdayHasItsNameAndThreeLetterAbbreviation)
{
    for (unsigned d = 1; d <= 7; ++d) {
        const weekday w{d == 7 ? 0 : d};
        EXPECT_EQ(snicholls::weekday_name(w), weekdays[d - 1]);
        EXPECT_EQ(snicholls::weekday_abbreviation(w), weekdays[d - 1].substr(0, 3));
        EXPECT_EQ(snicholls::weekday_name(weekday{d}), weekdays[d - 1]);   // weekday{7} is Sunday too
    }
}

TEST(TimeNames, ExportsTheNamesAsJson)
{
    std::ostringstream os;
    snicholls::write_enum_json<day_of_week>(os);
    const std::string json = os.str();
    EXPECT_NE(json.find("\"label\": \"Wednesday\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"abbreviation\": \"Wed\""), std::string::npos) << json;
}

} // namespace
