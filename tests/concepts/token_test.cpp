//
//  token_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/token.hpp>: what models token_like and what does not.
//

#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/concepts/token.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/native.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>

namespace {

using namespace snicholls;

struct hand_written {
    static constexpr std::string_view code() noexcept { return "USDC:8453"; }
    static constexpr std::uint8_t scale() noexcept { return 6; }
    static constexpr chain::base chain() noexcept { return {}; }
    static constexpr std::string_view contract() noexcept { return "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"; }
    static constexpr std::string_view symbol() noexcept { return "USDC"; }
};
struct no_contract {
    static constexpr std::string_view code() noexcept { return "X:1"; }
    static constexpr std::uint8_t scale() noexcept { return 6; }
    static constexpr chain::ethereum chain() noexcept { return {}; }
    static constexpr std::string_view symbol() noexcept { return "X"; }
};
struct chain_is_not_a_chain {
    static constexpr std::string_view code() noexcept { return "X:1"; }
    static constexpr std::uint8_t scale() noexcept { return 6; }
    static constexpr int chain() noexcept { return 1; }
    static constexpr std::string_view contract() noexcept { return "0x"; }
    static constexpr std::string_view symbol() noexcept { return "X"; }
};

static_assert(token_like<hand_written>);
static_assert(token_like<const hand_written&>);
static_assert(!token_like<no_contract>);
static_assert(!token_like<chain_is_not_a_chain>);
static_assert(!token_like<native::ETH>, "a native asset is an asset, not a token");
static_assert(!token_like<asset>);
static_assert(!token_like<int>);

TEST(TokenConcept, AHandWrittenTypeIsAnAssetToo)
{
    EXPECT_EQ(asset_v<hand_written>.code(), "USDC:8453");
    EXPECT_EQ(hand_written{}.chain().chain_id(), 8453u);
}

} // namespace
