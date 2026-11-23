//
//  tokens_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/token/tokens.hpp>: every token type against tests/fixtures/tokens/tokens.txt, which
//  scripts/generate_tokens.py writes from the committed data (scripts/data/tokens/issuers.json: the issuers' and
//  bridge operators' own published addresses). Addresses are compared case-exactly, so an EIP-55 checksum that drifted would fail here as
//  well as in the compiler.
//

#include <snicholls/core/interfaces/token.hpp>
#include <snicholls/core/interfaces/token_conformance.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/token/tokens.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace {

using namespace snicholls;

static_assert(meta::size_v<token::all> == 19);
static_assert(token::ethereum::USDC::code() == "USDC:1");
static_assert(token::bsc::USDC::code() == "USDC:56" && token::bsc::USDC::scale() == 18);
static_assert(token::bsc::USDT::scale() == 18, "USDT has 18 decimals on BNB Smart Chain");
static_assert(token::tron::USDT::code() == "USDT:728126428");
static_assert(token::tron::USDT::contract() == "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t");
static_assert(token::ethereum::WBTC::scale() == 8 && token::ethereum::WETH::scale() == 18);
static_assert(token::arbitrum::USDT0::code() == "USDT0:42161");
static_assert(asset_v<token::base::USDC> != asset_v<token::ethereum::USDC>);
static_assert(token::polygon::USDT0::code() == "USDT0:137");
static_assert(token::bsc::USDC::kind() == token::issuance::pegged && token::bsc::USDT::kind() == token::issuance::pegged);
static_assert(token::ethereum::USDC::kind() == token::issuance::native && token::ethereum::USDC::issuer() == "Circle");
static_assert(token::base::USDT::kind() == token::issuance::bridged && token::optimism::USDT::kind() == token::issuance::bridged);
static_assert(token::ethereum::WETH::kind() == token::issuance::wrapped && token::ethereum::WBTC::kind() == token::issuance::wrapped);
static_assert(noexcept(token::bsc::USDC::issuer()) && noexcept(token::bsc::USDC::kind()) && noexcept(token::bsc::USDC::source()));

struct row {
    unsigned decimals = 0;
    std::string address;
    std::string kind;
    std::string name;
};

std::map<std::pair<std::uint64_t, std::string>, row> read_fixture()
{
    std::map<std::pair<std::uint64_t, std::string>, row> rows;
    std::ifstream in(SNICHOLLS_CORE_FIXTURE_DIR "/tokens/tokens.txt");
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream fields(line);
        std::string id, symbol, decimals;
        row r;
        std::getline(fields, id, '\t');
        std::getline(fields, symbol, '\t');
        std::getline(fields, decimals, '\t');
        std::getline(fields, r.address, '\t');
        std::getline(fields, r.kind, '\t');
        std::getline(fields, r.name);
        r.decimals = static_cast<unsigned>(std::stoul(decimals));
        rows.emplace(std::make_pair(std::stoull(id), symbol), r);
    }
    return rows;
}

TEST(Tokens, EveryTypeMatchesTheCommittedData)
{
    const auto rows = read_fixture();
    ASSERT_EQ(rows.size(), meta::size_v<token::all>);
    std::set<std::pair<std::uint64_t, std::string>> seen;
    meta::for_each_type<token::all>([&]<typename T>() {
        const auto key = std::make_pair(std::uint64_t{T::chain().chain_id()}, std::string{T::symbol()});
        seen.insert(key);
        const auto it = rows.find(key);
        ASSERT_NE(it, rows.end()) << T::code();
        EXPECT_EQ(T::contract(), it->second.address) << T::code();
        EXPECT_EQ(T::scale(), it->second.decimals) << T::code();
        EXPECT_EQ(T::name(), it->second.name) << T::code();
        EXPECT_EQ(enum_name(T::kind()), it->second.kind) << T::code();
        EXPECT_FALSE(T::issuer().empty()) << T::code();
        EXPECT_TRUE(T::source().starts_with("https://")) << T::code();
        EXPECT_EQ(T::code(), std::string{T::symbol()} + ":" + std::to_string(T::chain().chain_id()));
        EXPECT_FALSE(check_token(T{})) << T::code();
        EXPECT_FALSE(check_token(token_model<T>{})) << T::code();
    });
    EXPECT_EQ(seen.size(), rows.size()) << "two types share a chain and symbol";
}

TEST(Tokens, CodesAreUnique)
{
    std::set<std::string> codes;
    meta::for_each_type<token::all>([&]<typename T>() { codes.insert(std::string{T::code()}); });
    EXPECT_EQ(codes.size(), meta::size_v<token::all>);
}

TEST(Tokens, KindSaysWhoStandsBehindTheToken)
{
    EXPECT_EQ(token::bsc::USDC::issuer(), "Binance (Binance-Peg USD Coin), not Circle");
    EXPECT_EQ(token::tron::USDT::issuer(), "Tether");
    EXPECT_EQ(token::arbitrum::USDT0::kind(), token::issuance::bridged);
    EXPECT_EQ(token::ethereum::USDC::source(), "https://developers.circle.com/stablecoins/usdc-contract-addresses");
    int native = 0;
    meta::for_each_type<token::all>([&]<typename T>() { native += T::kind() == token::issuance::native; });
    EXPECT_EQ(native, 11);
}

TEST(Tokens, ChainsAreTheChainTypes)
{
    EXPECT_EQ(token::ethereum::DAI::chain().name(), "Ethereum");
    EXPECT_EQ(token::optimism::USDC::chain().chain_id(), 10u);
    EXPECT_EQ(token::polygon::USDT0::chain().chain_id(), 137u);
    EXPECT_EQ(token::avalanche::USDC::chain().chain_id(), 43114u);
    EXPECT_EQ(token::tron::USDT::chain().kind(), chain_kind::tron);
}

} // namespace
