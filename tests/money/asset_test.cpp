//
//  asset_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/money/asset.hpp>.
//

#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using namespace snicholls;

struct eeau {   // a product's own token, declared as a type
    static constexpr std::string_view code() noexcept { return "EEAU"; }
    static constexpr std::uint8_t scale() noexcept { return 18; }
};

static_assert(std::is_trivially_copyable_v<asset> && std::is_standard_layout_v<asset>);
static_assert(sizeof(asset) == 14 && alignof(asset) == 1);
static_assert(asset_v<eeau>.code() == "EEAU" && asset_v<eeau>.scale() == 18);
static_assert(asset_v<currency::AUD> == *make_asset("AUD", 2));
static_assert(asset_v<native::ETH>.scale() == 18);
static_assert(asset_v<with_scale<currency::XAU, 18>>.code() == "XAU");

template <asset A>
struct tagged {};
static_assert(sizeof(tagged<asset_v<currency::AUD>>) == 1, "asset is structural: usable as a template argument");

TEST(AssetCode, AcceptsTheAllowedCharacters)
{
    for (const char* code : {"AUD", "ETH", "USDC:56", "EEAU2029", "A", "X.Y-Z_1", "ABCDEFGHIJKL"})
        EXPECT_TRUE(make_asset(code, 2)) << code;
}

TEST(AssetCode, RefusesMalformedCodesAndScales)
{
    for (const char* code : {"", "aud", "AUD ", "ABCDEFGHIJKLM", "A/B", "€"})
        EXPECT_FALSE(make_asset(code, 2)) << code;
    EXPECT_FALSE(make_asset("AUD", max_scale + 1));
    EXPECT_TRUE(make_asset("AUD", max_scale));
}

TEST(AssetValue, ComparesCodeAndScale)
{
    EXPECT_EQ(*make_asset("USDC:1", 6), *make_asset("USDC:1", 6));
    EXPECT_NE(*make_asset("USDC:1", 6), *make_asset("USDC:56", 6));
    EXPECT_NE(*make_asset("USDC:56", 6), *make_asset("USDC:56", 18));
}

} // namespace
