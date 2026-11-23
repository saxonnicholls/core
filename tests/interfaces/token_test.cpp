//
//  token_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/token.hpp>: the adapter from a type to the interface, and the interface as a
//  token_like in templates written over the concept.
//

#include <snicholls/core/interfaces/token.hpp>
#include <snicholls/core/interfaces/token_conformance.hpp>
#include <snicholls/core/token/tokens.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

using namespace snicholls;

constexpr token_model<token::bsc::USDC> bsc_usdc{};
static_assert(bsc_usdc.code() == "USDC:56" && bsc_usdc.scale() == 18);
static_assert(bsc_usdc.chain().chain_id() == 56u && bsc_usdc.chain().kind() == chain_kind::evm);
static_assert(bsc_usdc.chain().native_asset().code() == "BNB");
static_assert(bsc_usdc.symbol() == "USDC" && bsc_usdc.contract() == token::bsc::USDC::contract());
static_assert(!check_token(bsc_usdc));
static_assert(to_asset(bsc_usdc) == asset_v<token::bsc::USDC>);
static_assert(token_like<token_interface> && token_like<token_model<token::tron::USDT>>);

// a template over the concept, called with both forms
template <token_like T>
std::string describe(const T& t)
{
    return std::string{t.symbol()} + " on " + std::string{t.chain().name()};
}

// a run-time registry row behind the interface, as servers' registry would be
class registry_row final : public token_interface {
public:
    registry_row(std::string code, std::uint8_t scale, std::string contract, std::string symbol)
        : code_(std::move(code)), scale_(scale), contract_(std::move(contract)), symbol_(std::move(symbol)) {}
    std::string_view code() const noexcept override { return code_; }
    std::uint8_t scale() const noexcept override { return scale_; }
    const chain_interface& chain() const noexcept override { return chain_; }
    std::string_view contract() const noexcept override { return contract_; }
    std::string_view symbol() const noexcept override { return symbol_; }

private:
    std::string code_;
    std::uint8_t scale_;
    std::string contract_;
    std::string symbol_;
    chain_model<chain::base> chain_;
};

TEST(TokenInterface, ChoosesATokenAtRunTime)
{
    std::vector<std::unique_ptr<token_interface>> wallet;
    wallet.push_back(std::make_unique<token_model<token::ethereum::USDC>>());
    wallet.push_back(std::make_unique<token_model<token::tron::USDT>>());
    wallet.push_back(std::make_unique<registry_row>("AERO:8453", 18, "0x940181a94A35A4569E4529A3CDfB74e38FD98631", "AERO"));
    EXPECT_EQ(wallet[0]->code(), "USDC:1");
    EXPECT_EQ(*wallet[1]->chain().chain_id(), 728126428u);
    EXPECT_EQ(wallet[1]->chain().kind(), chain_kind::tron);
    EXPECT_EQ(wallet[2]->chain().name(), "Base");
    for (const auto& t : wallet) {
        EXPECT_FALSE(check_token(*t)) << t->code();
    }
}

TEST(TokenInterface, BothFormsGoWhereTheConceptIsExpected)
{
    const token_model<token::ethereum::WETH> model;
    EXPECT_EQ(describe(token::ethereum::WETH{}), "WETH on Ethereum");
    EXPECT_EQ(describe(static_cast<const token_interface&>(model)), "WETH on Ethereum");
}

} // namespace
