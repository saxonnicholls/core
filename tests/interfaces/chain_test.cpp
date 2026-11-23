//
//  chain_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/chain.hpp>.
//

#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/interfaces/chain.hpp>
#include <snicholls/core/interfaces/chain_conformance.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace {

using namespace snicholls;

constexpr chain_model<chain::base> base_model{};
static_assert(base_model.chain_id() == 8453u && base_model.native_asset() == asset_v<native::ETH>);
static_assert(!chain_model<chain::solana>{}.chain_id());

TEST(ChainInterface, ChoosesAChainAtRunTime)
{
    std::vector<std::unique_ptr<chain_interface>> wallet;
    wallet.push_back(std::make_unique<chain_model<chain::ethereum>>());
    wallet.push_back(std::make_unique<chain_model<chain::bsc>>());
    wallet.push_back(std::make_unique<chain_model<chain::bitcoin>>());
    EXPECT_EQ(*wallet[0]->chain_id(), 1u);
    EXPECT_EQ(wallet[1]->native_asset().code(), "BNB");
    EXPECT_FALSE(wallet[2]->chain_id());
    EXPECT_EQ(wallet[2]->kind(), chain_kind::bitcoin);
    for (const auto& c : wallet)
        EXPECT_FALSE(check_chain(*c)) << c->name();
}

} // namespace
