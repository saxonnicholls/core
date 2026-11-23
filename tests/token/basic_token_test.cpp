//
//  basic_token_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/token/basic_token.hpp>: a token declared in one line, its code built at compile time.
//  (A wrong address, decimals above max_scale or a bad symbol are static_assert failures; the checks behind them are
//  tested in tests/detail/token_checks_test.cpp.)
//

#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/interfaces/token_conformance.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/token/basic_token.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using namespace snicholls;

// EIP-55's own example address, standing in for a product's token
struct SAX : basic_token<chain::base, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", 18, "SAX"> {};
// a single-supply issuer token keeps its bare symbol
struct SAXG : basic_token<chain::arbitrum, "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359", 6, "SAXG"> {
    static constexpr std::string_view code() noexcept { return "SAXG"; }
};
// TRON, with the longest identifier of the major chains
struct TUSD : basic_token<chain::tron, "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t", 6, "TUSD"> {};
// the longest code: a 13-character symbol and a 10-digit identifier would be 24; a 9-digit one is 23
struct LONG : basic_token<chain::tron, "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t", 36, "ABCDEFGHIJKLM"> {};
struct ZERO : basic_token<chain::ethereum, "0x52908400098527886E0F7030069857D2E4169EE7", 0, "Z.1-A_B"> {};

static_assert(token_like<SAX> && token_like<SAXG> && token_like<TUSD>);
static_assert(SAX::code() == "SAX:8453" && SAX::scale() == 18 && SAX::symbol() == "SAX");
static_assert(SAX::contract() == "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed");
static_assert(std::is_same_v<decltype(SAX::chain()), chain::base>);
static_assert(SAXG::code() == "SAXG");
static_assert(TUSD::code() == "TUSD:728126428" && TUSD::chain().kind() == chain_kind::tron);
static_assert(LONG::code() == "ABCDEFGHIJKLM:728126428" && LONG::code().size() == 23 && LONG::scale() == max_scale);
static_assert(ZERO::code() == "Z.1-A_B:1" && ZERO::scale() == 0);
static_assert(asset_v<SAX> == *make_asset("SAX:8453", 18));
static_assert(noexcept(SAX::code()) && noexcept(SAX::contract()) && noexcept(SAX::chain()));
static_assert(std::is_empty_v<SAX> && std::is_trivially_copyable_v<SAX>);
static_assert(!check_token(SAX{}) && !check_token(SAXG{}) && !check_token(TUSD{}) && !check_token(LONG{}));

TEST(BasicToken, DeclaresAConformingTokenInOneLine)
{
    EXPECT_EQ(SAX::code(), "SAX:8453");
    EXPECT_EQ(SAX{}.chain().chain_id(), 8453u);
    EXPECT_FALSE(check_token(SAX{}));
    EXPECT_FALSE(check_token(ZERO{}));
}

TEST(BasicToken, OneSymbolOnTwoChainsIsTwoAssets)
{
    struct on_base : basic_token<chain::base, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", 18, "SAX"> {};
    struct on_bsc : basic_token<chain::bsc, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", 18, "SAX"> {};
    EXPECT_NE(asset_v<on_base>, asset_v<on_bsc>);
    EXPECT_EQ(on_bsc::code(), "SAX:56");
}

} // namespace
