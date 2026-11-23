//
//  country_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/country.hpp>: the types, the lookups and the literal; and every country against the
//  oracle fixture tests/fixtures/countries/countries.tsv (pycountry, babel and phonenumbers, through
//  scripts/generate_country_fixtures.py, never core's own output).
//

#include <snicholls/core/concepts/country.hpp>
#include <snicholls/core/country.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef SNICHOLLS_CORE_FIXTURE_DIR
#error "SNICHOLLS_CORE_FIXTURE_DIR must name tests/fixtures (set by tests/CMakeLists.txt)"
#endif

namespace {

using namespace snicholls;
using snicholls::literals::operator""_country;

static_assert(country_like<country::AU>);
static_assert(country::AU::alpha3() == "AUS" && country::AU::numeric() == 36 && country::AU::name() == "Australia");
static_assert(country::AU::calling_code() == 61 && country::GB::calling_code() == 44 && country::US::calling_code() == 1);
static_assert(country::AU::currencies().size() == 1 && country::AU::currencies()[0] == "AUD");
static_assert(country::AQ::calling_code() == 0 && country::AQ::currencies().empty(), "Antarctica: no code, no currency");
static_assert(country::BG::currencies()[0] == "EUR", "SIX's list one: Bulgaria joined the euro in 2026");
static_assert(country::DO::alpha3() == "DOM" && country::IN::alpha3() == "IND", "capitals: do and in are keywords");
static_assert("AU"_country.numeric == 36 && "AUS"_country.alpha2 == "AU");
static_assert(find_country("NZ")->alpha3 == "NZL" && find_country("NZL")->alpha2 == "NZ");
static_assert(find_country(std::uint16_t{554})->alpha2 == "NZ");
static_assert(!find_country("ZZ") && !find_country("XK") && !find_country("A") && !find_country(std::uint16_t{0}));
static_assert(iso3166_cldr_version == "47");

TEST(CountryTable, IsSortedAndComplete)
{
    const auto rows = iso3166_entries();
    EXPECT_EQ(rows.size(), 249u);   // ISO 3166-1, as pycountry 26.2.16 has it
    for (std::size_t i = 1; i < rows.size(); ++i)
        EXPECT_LT(rows[i - 1].alpha2, rows[i].alpha2);
}

TEST(CountryLookup, ByEveryCodeAndNothingElse)
{
    for (const auto& r : iso3166_entries()) {
        ASSERT_NE(find_country(r.alpha2), nullptr) << r.alpha2;
        EXPECT_EQ(find_country(r.alpha2), &r);
        EXPECT_EQ(find_country(r.alpha3), &r) << r.alpha3;
        EXPECT_EQ(find_country(r.numeric), &r) << r.numeric;
    }
    EXPECT_EQ(find_country(""), nullptr);
    EXPECT_EQ(find_country("au"), nullptr);
    EXPECT_EQ(find_country("AUSS"), nullptr);
    EXPECT_EQ(find_country("XKK"), nullptr);
    EXPECT_EQ(find_country(std::uint16_t{999}), nullptr);
}

TEST(CountryCurrencies, AreTheInverseOfTheCurrenciesCountries)
{
    // Every country's currencies name it among their countries; every currency's countries name it, unless it is a
    // fund (SIX's IsFund codes, left out of a country's currencies: BOV, CHE, CHW, CLF, COU, MXV, USN, UYI, UYW).
    const std::set<std::string_view> funds{"BOV", "CHE", "CHW", "CLF", "COU", "MXV", "USN", "UYI", "UYW"};
    for (const auto& c : iso3166_entries())
        for (const auto code : c.currencies) {
            const auto* e = find_iso4217(code);
            ASSERT_NE(e, nullptr) << code;
            EXPECT_TRUE(std::ranges::find(e->countries, c.alpha2) != e->countries.end()) << c.alpha2 << " " << code;
        }
    for (const auto& e : iso4217_entries())
        for (const auto a2 : e.countries) {
            const auto* c = find_country(a2);
            ASSERT_NE(c, nullptr) << a2;
            const bool listed = std::ranges::find(c->currencies, e.code) != c->currencies.end();
            EXPECT_EQ(listed, !funds.contains(e.code)) << a2 << " " << e.code;
        }
}

struct oracle_row {
    std::string alpha2, alpha3;
    int numeric = 0, calling_code = 0;
    std::string six, cldr, name;
};

std::vector<oracle_row> oracle()
{
    std::ifstream in(SNICHOLLS_CORE_FIXTURE_DIR "/countries/countries.tsv", std::ios::binary);
    std::vector<oracle_row> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream fields(line);
        oracle_row r;
        std::string numeric, call;
        std::getline(fields, r.alpha2, '\t');
        std::getline(fields, r.alpha3, '\t');
        std::getline(fields, numeric, '\t');
        std::getline(fields, call, '\t');
        std::getline(fields, r.six, '\t');
        std::getline(fields, r.cldr, '\t');
        std::getline(fields, r.name);
        r.numeric = std::stoi(numeric);
        r.calling_code = std::stoi(call);
        rows.push_back(r);
    }
    return rows;
}

std::string joined(std::span<const std::string_view> codes)
{
    std::string out;
    for (const auto c : codes)
        out += (out.empty() ? "" : " ") + std::string(c);
    return out.empty() ? "-" : out;
}

TEST(CountryOracle, AgreesWithPycountryBabelAndPhonenumbers)
{
    const auto rows = oracle();
    ASSERT_EQ(rows.size(), iso3166_entries().size()) << "the fixture is missing or incomplete";
    for (const auto& o : rows) {
        SCOPED_TRACE("tests/fixtures/countries/countries.tsv: " + o.alpha2);
        const auto* c = find_country(o.alpha2);
        ASSERT_NE(c, nullptr);
        EXPECT_EQ(c->alpha3, o.alpha3);
        EXPECT_EQ(c->numeric, o.numeric);
        EXPECT_EQ(c->name, o.name);
        EXPECT_EQ(c->calling_code, o.calling_code);
        EXPECT_EQ(joined(c->currencies), o.six);
    }
}

TEST(CountryOracle, DiffersFromCldrsCurrenciesOnlyWhereKnown)
{
    // SIX's list one of 2026-09-17 against CLDR 47 (babel 2.18.0): Bulgaria's euro is newer than CLDR 47; CLDR adds
    // currencies in use unofficially (GBP in South Georgia, ILS and JOD in Palestine, USD in Zimbabwe) and leaves out
    // two SIX lists (SVC, VED).
    const std::set<std::string> known{"BG", "GS", "PS", "SV", "VE", "ZW"};
    std::set<std::string> differ;
    for (const auto& o : oracle())
        if (o.six != o.cldr)
            differ.insert(o.alpha2);
    EXPECT_EQ(differ, known);
}

} // namespace
