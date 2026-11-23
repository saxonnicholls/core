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

#include <algorithm>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;

struct eeau {   // a product's own token, declared as a type
    static constexpr std::string_view code() noexcept { return "EEAU"; }
    static constexpr std::uint8_t scale() noexcept { return 18; }
};

static_assert(std::is_trivially_copyable_v<asset> && std::is_standard_layout_v<asset>);
static_assert(sizeof(asset) == 26 && alignof(asset) == 1);
static_assert(asset_v<eeau>.code() == "EEAU" && asset_v<eeau>.scale() == 18);
static_assert(asset_v<currency::AUD> == *make_asset("AUD", 2));
static_assert(asset_v<native::ETH>.scale() == 18);
static_assert(asset_v<with_scale<currency::XAU, 18>>.code() == "XAU");

static_assert(checked_asset("USDC:8453", 6) == *make_asset("USDC:8453", 6));
static_assert(checked_asset("AB", 2) < checked_asset("ABC", 0), "a prefix sorts first");
static_assert(checked_asset("AUD", 2) < checked_asset("AUD", 6) && checked_asset("AUD", 6) < checked_asset("EUR", 2));
template <class C>
concept checked_compiles = requires { typename std::integral_constant<std::uint8_t, checked_asset(C::code, 2).scale()>; };
struct good_code { static constexpr std::string_view code = "USDC:1"; };
struct bad_code { static constexpr std::string_view code = "usdc"; };
static_assert(checked_compiles<good_code> && !checked_compiles<bad_code>, "a bad code is a compile error");

template <asset A>
struct tagged {};
static_assert(sizeof(tagged<asset_v<currency::AUD>>) == 1, "asset is structural: usable as a template argument");

TEST(AssetCode, AcceptsTheAllowedCharacters)
{
    for (const char* code : {"AUD", "ETH", "USDC:56", "EEAU2029:42161", "WSTETH:42161", "A", "X.Y-Z_1", "ABCDEFGHIJKLMNOPQRSTUVWX"})
        EXPECT_TRUE(make_asset(code, 2)) << code;
}

TEST(AssetCode, RefusesMalformedCodesAndScales)
{
    for (const char* code : {"", "aud", "AUD ", "ABCDEFGHIJKLMNOPQRSTUVWXY", "A/B", "€"})
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

TEST(AssetOrdering, SortsByCodeThenScale)
{
    std::vector<asset> v{checked_asset("USDC:56", 6), checked_asset("AUD", 2), checked_asset("USDC:1", 6),
                         checked_asset("AUD", 0)};
    std::ranges::sort(v);
    EXPECT_EQ(v[0], checked_asset("AUD", 0));
    EXPECT_EQ(v[1], checked_asset("AUD", 2));
    EXPECT_EQ(v[2], checked_asset("USDC:1", 6));
    EXPECT_EQ(v[3], checked_asset("USDC:56", 6));
}

} // namespace
