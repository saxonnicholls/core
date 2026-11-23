//
//  iso4217_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/money/iso4217.hpp>. The table itself is checked against SIX's XML by the CTest
//  generated.iso4217_up_to_date (scripts/generate_iso4217.py --check); these tests check the lookups and the types.
//

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <gtest/gtest.h>

#include <string_view>

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

} // namespace
