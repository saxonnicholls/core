//
//  token_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/token_conformance.hpp>: the kit catches each broken rule.
//

#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/interfaces/token.hpp>
#include <snicholls/core/interfaces/token_conformance.hpp>
#include <snicholls/core/money/native.hpp>
#include <snicholls/core/token/basic_token.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

namespace {

using namespace snicholls;

// A token whose every answer is a template argument, so each test breaks one rule.
template <typename Chain, std::uint8_t Scale = 6>
struct fake {
    std::string_view code_ = "USDC:1";
    std::string_view contract_ = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";
    std::string_view symbol_ = "USDC";
    constexpr std::string_view code() const noexcept { return code_; }
    static constexpr std::uint8_t scale() noexcept { return Scale; }
    static constexpr Chain chain() noexcept { return {}; }
    constexpr std::string_view contract() const noexcept { return contract_; }
    constexpr std::string_view symbol() const noexcept { return symbol_; }
};
using eth = fake<chain::ethereum>;

struct zero_chain {
    static constexpr std::string_view name() noexcept { return "Zero"; }
    static constexpr chain_kind kind() noexcept { return chain_kind::evm; }
    static constexpr std::uint64_t chain_id() noexcept { return 0; }
    static constexpr native::ETH native_asset() noexcept { return {}; }
};
struct other_chain {
    static constexpr std::string_view name() noexcept { return "Other"; }
    static constexpr chain_kind kind() noexcept { return chain_kind::other; }
    static constexpr native::ETH native_asset() noexcept { return {}; }
};

// changes its contract between calls
struct fickle {
    mutable int calls = 0;
    static constexpr std::string_view code() noexcept { return "USDC:1"; }
    static constexpr std::uint8_t scale() noexcept { return 6; }
    static constexpr chain::ethereum chain() noexcept { return {}; }
    std::string_view contract() const noexcept
    {
        return ++calls == 1 ? "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48" : "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";
    }
    static constexpr std::string_view symbol() noexcept { return "USDC"; }
};

static_assert(!check_token(eth{}));

TEST(TokenConformance, AcceptsAConformingToken)
{
    EXPECT_FALSE(check_token(eth{}));
    EXPECT_FALSE(check_token(fake<chain::tron>{"USDT:728126428", "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t", "USDT"}));
    EXPECT_FALSE(check_token(eth{"USDC", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC"}));   // a bare symbol
    EXPECT_FALSE(check_token(fake<other_chain>{"OTHER", "anything", "OTHER"}));
}

TEST(TokenConformance, NamesTheFailedCheck)
{
    EXPECT_EQ(*check_token(eth{"usdc:1"}), "asset: code() is not 1 to 24 of A-Z 0-9 : . - _");
    EXPECT_EQ(*check_token(fake<chain::ethereum, 37>{}), "asset: scale() is above max_scale");
    EXPECT_EQ(*check_token(fake<zero_chain>{}), "chain: chain_id() is zero");
    EXPECT_EQ(*check_token(eth{":1", "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", ""}), "token: symbol() is empty");
    EXPECT_EQ(*check_token(eth{"USDC:56"}), "token: code() is neither symbol() nor symbol():chain_id");
    EXPECT_EQ(*check_token(fake<other_chain>{"OTHER:1", "x", "OTHER"}),
              "token: code() is neither symbol() nor symbol():chain_id");
    EXPECT_EQ(*check_token(eth{"USDC:1", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"}),
              "token: contract() fails its EIP-55 checksum");
    EXPECT_EQ(*check_token(eth{"USDC:1", "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"}),
              "token: contract() is not 0x and 40 hex digits");
    EXPECT_EQ(*check_token(fake<chain::tron>{"USDT:728126428", "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6u", "USDT"}),
              "token: contract() fails its base58check checksum");
    EXPECT_EQ(*check_token(fake<other_chain>{"OTHER", "", "OTHER"}), "token: contract() is empty");
    EXPECT_EQ(*check_token(fickle{}), "token: symbol() or contract() changes between calls");
}

TEST(TokenConformance, ChecksAnObjectBehindTheInterface)
{
    struct USDC : basic_token<chain::ethereum, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6, "USDC"> {};
    const token_model<USDC> model;
    EXPECT_FALSE(check_token(static_cast<const token_interface&>(model)));
}

} // namespace
