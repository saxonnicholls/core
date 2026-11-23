//
//  iso4217_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/money/iso4217.hpp>. The table itself is checked against SIX's XML by the CTest
//  generated.iso4217_up_to_date (scripts/generate_iso4217.py --check); these tests check the lookups and the types,
//  and the countries, cash rules and symbols against the oracle fixture tests/fixtures/money/currencies.tsv (pycountry
//  and babel, through scripts/generate_country_fixtures.py, never core's own output).
//

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/meta/attributes.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef SNICHOLLS_CORE_FIXTURE_DIR
#error "SNICHOLLS_CORE_FIXTURE_DIR must name tests/fixtures (set by tests/CMakeLists.txt)"
#endif

namespace {

using namespace snicholls;
using snicholls::literals::operator""_iso4217;

static_assert(currency_like<currency::AUD>);
static_assert(currency::AUD::numeric() == 36 && currency::AUD::scale() == 2);
static_assert(currency::JPY::scale() == 0 && currency::KWD::scale() == 3 && currency::BHD::scale() == 3);
static_assert(!asset_like<currency::XAU>, "XAU has no ISO minor units");
static_assert(asset_like<with_scale<currency::XAU, 18>>);
static_assert(currency::ANG::withdrawn() == "2025-03", "historic codes resolve");
static_assert(currency::XCG::scale() == 2, "the Caribbean guilder, which replaced ANG");
static_assert(currency::TRY::code() == "TRY", "capitals: lowercase would be the keyword try");
static_assert("AUD"_iso4217 == asset_v<currency::AUD>);
static_assert(iso4217_asset("EUR")->scale() == 2);
static_assert(iso4217_published.size() == 10);
static_assert(iso4217_cldr_version == "47");

// CLDR's cash rules, where it gives one: CHF cash is a multiple of 0.05, DKK of 0.50, SEK of whole kronor.
static_assert(currency::CHF::cash_scale() == 2 && currency::CHF::cash_rounding() == 5);
static_assert(currency::DKK::cash_scale() == 2 && currency::DKK::cash_rounding() == 50);
static_assert(currency::SEK::cash_scale() == 0 && currency::SEK::cash_rounding() == 1);
template <typename C>
concept has_cash_rule = requires { C::cash_scale(); };
static_assert(!has_cash_rule<currency::AUD> && !has_cash_rule<currency::JPY>, "only where CLDR gives one");

// The countries using it, from SIX's list one; none for a historic code.
static_assert(currency::CHF::countries().size() == 2 && currency::CHF::countries()[0] == "CH");
static_assert(currency::XAU::countries().empty());
template <typename C>
concept has_countries = requires { C::countries(); };
static_assert(!has_countries<currency::ANG>, "historic: list three is not read for countries");

// The symbol is presentation: an attribute found by ADL, with the minor units.
static_assert(has_attributes<currency::AUD>);
static_assert(attribute_or(currency::AUD{}, label{""}).text == "A$");
static_assert(attribute_or(currency::EUR{}, label{""}).text == "€");
static_assert(attribute_or(currency::JPY{}, label{""}).text == "JP¥", "the root locale: unambiguous, not en's ¥");
static_assert(attribute_or(currency::CHF{}, label{""}).text == "CHF", "no root symbol: the code");
static_assert(attribute_or(currency::KWD{}, minor_units{-1}).digits == 3);
static_assert(!has_attribute_v<minor_units, currency::XAU>, "no ISO scale, no minor_units");

TEST(Iso4217Table, IsSortedAndComplete)
{
    const auto rows = iso4217_entries();
    ASSERT_GT(rows.size(), 300u);
    for (std::size_t i = 1; i < rows.size(); ++i)
        EXPECT_LT(rows[i - 1].code, rows[i].code);
    std::size_t current = 0;
    for (const auto& r : rows)
        current += r.withdrawn.empty() ? 1 : 0;
    EXPECT_EQ(current, 178u);   // SIX list one of 2026-09-17: 178 distinct codes
    EXPECT_EQ(iso4217_published, "2026-09-17");
}

TEST(Iso4217Lookup, ByCodeAndByNumber)
{
    ASSERT_NE(find_iso4217("AUD"), nullptr);
    EXPECT_EQ(find_iso4217("AUD")->name, "Australian Dollar");
    EXPECT_EQ(find_iso4217("ZZZ"), nullptr);
    EXPECT_EQ(find_iso4217(std::uint16_t{36})->code, "AUD");
    EXPECT_EQ(find_iso4217(std::uint16_t{532})->code, "XCG");   // the current code wins over the withdrawn ANG
    EXPECT_EQ(find_iso4217(std::uint16_t{999})->code, "XXX");
    EXPECT_EQ(find_iso4217(std::uint16_t{1}), nullptr);
}

TEST(Iso4217Lookup, AssetsOnlyWhereIsoGivesAScale)
{
    EXPECT_EQ(iso4217_asset("JPY")->scale(), 0);
    EXPECT_FALSE(iso4217_asset("XAU"));   // N.A.
    EXPECT_FALSE(iso4217_asset("ANG"));   // historic: list three has no minor units
    EXPECT_FALSE(iso4217_asset("aud"));
}

TEST(Iso4217Metadata, RowsCarryTheSameAsTheTypes)
{
    const auto* chf = find_iso4217("CHF");
    ASSERT_NE(chf, nullptr);
    EXPECT_EQ(chf->cash_scale, 2);
    EXPECT_EQ(chf->cash_rounding, 5);
    EXPECT_EQ(chf->symbol, "CHF");
    EXPECT_EQ(chf->countries.data(), currency::CHF::countries().data());
    const auto* aud = find_iso4217("AUD");
    EXPECT_EQ(aud->cash_scale, -1);
    EXPECT_EQ(aud->cash_rounding, 0);
    EXPECT_EQ(aud->symbol, "A$");
    EXPECT_EQ(find_iso4217("ANG")->countries.size(), 0u);
}

struct oracle_row {
    std::string code;
    int digits = 0, rounding = 0, cash_digits = 0, cash_rounding = 0;
    std::string symbol, countries;
};

std::vector<oracle_row> oracle()
{
    std::ifstream in(SNICHOLLS_CORE_FIXTURE_DIR "/money/currencies.tsv", std::ios::binary);
    std::vector<oracle_row> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream fields(line);
        oracle_row r;
        std::string f[4];
        std::getline(fields, r.code, '\t');
        for (auto& x : f)
            std::getline(fields, x, '\t');
        std::getline(fields, r.symbol, '\t');
        std::getline(fields, r.countries);
        r.digits = std::stoi(f[0]);
        r.rounding = std::stoi(f[1]);
        r.cash_digits = std::stoi(f[2]);
        r.cash_rounding = std::stoi(f[3]);
        rows.push_back(r);
    }
    return rows;
}

TEST(Iso4217Oracle, AgreesWithBabelAndPycountry)
{
    const auto rows = oracle();
    ASSERT_EQ(rows.size(), iso4217_entries().size()) << "the fixture is missing or incomplete";
    for (const auto& o : rows) {
        SCOPED_TRACE("tests/fixtures/money/currencies.tsv: " + o.code);
        const auto* e = find_iso4217(o.code);
        ASSERT_NE(e, nullptr);
        EXPECT_EQ(e->symbol, o.symbol);
        if (e->cash_scale >= 0) {   // CLDR gives a cash rule: babel's cash fractions, a rounding of 0 meaning 1
            EXPECT_EQ(e->cash_scale, o.cash_digits);
            EXPECT_EQ(e->cash_rounding, std::max(o.cash_rounding, 1));
        } else {   // none: babel's cash fractions are its ordinary ones
            EXPECT_EQ(o.cash_digits, o.digits);
            EXPECT_EQ(o.cash_rounding, o.rounding);
        }
        std::string countries;
        for (const auto a2 : e->countries)
            countries += (countries.empty() ? "" : " ") + std::string(a2);
        EXPECT_EQ(countries.empty() ? "-" : countries, o.countries);
    }
}

} // namespace
