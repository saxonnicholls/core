//
//  e164_country_codes_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for the generated <snicholls/core/detail/e164_country_codes.hpp>: the table against the committed snapshot
//  it was generated from (scripts/data/e164/country_codes.txt), and its invariants (ascending, 1 to 3 digits,
//  prefix-free, the shared +1 and +7 zones and the non-geographic codes present).
//

#include <snicholls/core/detail/e164_country_codes.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace e = snicholls::detail::e164;

static_assert(std::string_view(e::source) == "phonenumbers 9.0.44");
static_assert(std::string_view(e::snapshot_date).size() == 10);
static_assert(e::country_codes[0].code == 1);

TEST(E164CountryCodes, AreAscendingAndOneToThreeDigits)
{
    for (std::size_t i = 0; i < std::size(e::country_codes); ++i) {
        EXPECT_GE(e::country_codes[i].code, 1U);
        EXPECT_LE(e::country_codes[i].code, 999U);
        if (i > 0) {
            EXPECT_LT(e::country_codes[i - 1].code, e::country_codes[i].code);
        }
    }
}

TEST(E164CountryCodes, ArePrefixFree)
{
    for (const auto& a : e::country_codes)
        for (const auto& b : e::country_codes) {
            if (a.code == b.code) continue;
            const std::string sa = std::to_string(a.code), sb = std::to_string(b.code);
            EXPECT_NE(sb.rfind(sa, 0), 0U) << sa << " is a prefix of " << sb;
        }
}

TEST(E164CountryCodes, IncludeTheSharedZonesAndNonGeographicCodes)
{
    auto find = [](unsigned code) {
        for (const auto& c : e::country_codes)
            if (c.code == code) return true;
        return false;
    };
    for (unsigned code : {1U, 7U, 44U, 61U, 86U, 800U, 808U, 870U, 878U, 881U, 882U, 883U, 888U, 979U})
        EXPECT_TRUE(find(code)) << code;
    for (unsigned code : {0U, 28U, 42U, 80U, 999U})
        EXPECT_FALSE(find(code)) << code;
}

TEST(E164CountryCodes, MatchTheCommittedSnapshot)
{
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/../../scripts/data/e164/country_codes.txt");
    ASSERT_TRUE(in) << "scripts/data/e164/country_codes.txt not found";
    std::vector<std::pair<unsigned, bool>> rows;
    std::string line;
    bool dated = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line.front() == '#') {
            if (line.find(std::string("extracted ") + e::snapshot_date + " from " + e::source) != std::string::npos)
                dated = true;
            continue;
        }
        std::istringstream fields(line);
        unsigned code = 0, trunk_zero = 0;
        fields >> code >> trunk_zero;
        rows.emplace_back(code, trunk_zero != 0);
    }
    EXPECT_TRUE(dated) << "the snapshot's header names the same date and source";
    ASSERT_EQ(rows.size(), std::size(e::country_codes));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        EXPECT_EQ(rows[i].first, e::country_codes[i].code);
        EXPECT_EQ(rows[i].second, e::country_codes[i].trunk_zero) << rows[i].first;
    }
}

TEST(E164CountryCodes, MarkTheTrunkZeroCountries)
{
    auto trunk_zero = [](unsigned code) {
        for (const auto& c : e::country_codes)
            if (c.code == code) return c.trunk_zero;
        return false;
    };
    EXPECT_TRUE(trunk_zero(44));   // United Kingdom: 020 ... nationally
    EXPECT_TRUE(trunk_zero(61));   // Australia
    EXPECT_TRUE(trunk_zero(49));   // Germany
    EXPECT_FALSE(trunk_zero(39));   // Italy: the 0 is part of the number
    EXPECT_FALSE(trunk_zero(1));   // North America: the trunk prefix is 1
    EXPECT_FALSE(trunk_zero(7));   // Russia: 8
    EXPECT_FALSE(trunk_zero(800));
}

} // namespace
