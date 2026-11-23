//
//  names_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/names.hpp>: month and weekday names read from the attributes of month_of_year and
//  day_of_week, for std::chrono's types, printed through acout, and exported as JSON. Exits non-zero if a name
//  is not the English one.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/time/names.hpp>

#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    std::printf("snicholls-core names demo\n");
    using namespace std::chrono;

    //! [names]
    constexpr year_month_day d = 2026y / November / 23d;
    static_assert(snicholls::month_name(d.month()) == "November");
    static_assert(snicholls::weekday_abbreviation(weekday{sys_days{d}}) == "Mon");

    // the names are attributes of the enumerations, so every printer reads the same ones
    static_assert(snicholls::attributes_of(snicholls::month_of_year::may).get<snicholls::label>().text == "May");
    static_assert(snicholls::attributes_of(snicholls::day_of_week::sunday).get<snicholls::abbreviation>().text == "Sun");
    //! [names]

    snicholls::acout{} << "  " << snicholls::weekday_name(weekday{sys_days{d}}) << ' ' << static_cast<unsigned>(d.day()) << ' '
                       << snicholls::month_name(d.month()) << ' ' << static_cast<int>(d.year()) << '\n';
    for (unsigned m = 1; m <= 12; ++m)
        std::printf("  %2u %-9.*s %.*s\n", m, static_cast<int>(snicholls::month_name(month{m}).size()), snicholls::month_name(month{m}).data(),
                    static_cast<int>(snicholls::month_abbreviation(month{m}).size()), snicholls::month_abbreviation(month{m}).data());

    std::ostringstream json;
    snicholls::write_enum_json<snicholls::day_of_week>(json);
    std::printf("  %s\n", json.str().c_str());

    check(snicholls::weekday_name(Saturday) == "Saturday" && snicholls::month_abbreviation(February) == "Feb", "names and abbreviations");
    check(snicholls::to_day_of_week(Sunday) == snicholls::day_of_week::sunday &&
              snicholls::to_month_of_year(March) == snicholls::month_of_year::march,
          "std::chrono's types map to the enumerations");
    check(snicholls::month_name(month{13}).empty() && snicholls::weekday_abbreviation(weekday{8}).empty(), "an invalid value has no name");
    check(json.str().find("\"thursday\": {\"label\": \"Thursday\", \"abbreviation\": \"Thu\"}") != std::string::npos,
          "the JSON export carries both attributes");
    return failures == 0 ? 0 : 1;
}
