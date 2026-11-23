//
//  chains_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/chain/chains.hpp>: every EIP-155 identifier and native currency against the chain
//  registry (github.com/ethereum-lists/chains, _data/chains/eip155-<id>.json: chainId, nativeCurrency.symbol and
//  .decimals), and the kinds.
//

#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/interfaces/chain_conformance.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

template <typename C>
constexpr bool evm(std::uint64_t id, std::string_view native, unsigned decimals)
{
    return evm_chain_like<C> && C::chain_id() == id && C::kind() == chain_kind::evm &&
           decltype(C::native_asset())::code() == native && decltype(C::native_asset())::scale() == decimals;
}

static_assert(evm<chain::ethereum>(1, "ETH", 18));
static_assert(evm<chain::optimism>(10, "ETH", 18));
static_assert(evm<chain::bsc>(56, "BNB", 18));
static_assert(evm<chain::polygon>(137, "POL", 18));
static_assert(evm<chain::base>(8453, "ETH", 18));
static_assert(evm<chain::arbitrum>(42161, "ETH", 18));
static_assert(evm<chain::avalanche>(43114, "AVAX", 18));
static_assert(evm<chain::base_sepolia>(84532, "ETH", 18));
static_assert(evm<chain::arbitrum_sepolia>(421614, "ETH", 18));
static_assert(evm<chain::sepolia>(11155111, "ETH", 18));
static_assert(chain::tron::chain_id() == 728126428 && chain::tron::kind() == chain_kind::tron);
static_assert(chain_like<chain::bitcoin> && !evm_chain_like<chain::bitcoin> && chain::bitcoin::kind() == chain_kind::bitcoin);
static_assert(chain_like<chain::solana> && !evm_chain_like<chain::solana>);

TEST(Chains, EveryChainConforms)
{
    EXPECT_FALSE(check_chain(chain::ethereum{}));
    EXPECT_FALSE(check_chain(chain::base{}));
    EXPECT_FALSE(check_chain(chain::arbitrum{}));
    EXPECT_FALSE(check_chain(chain::bsc{}));
    EXPECT_FALSE(check_chain(chain::tron{}));
    EXPECT_FALSE(check_chain(chain::bitcoin{}));
    EXPECT_FALSE(check_chain(chain::solana{}));
    EXPECT_EQ(chain::base::name(), "Base");
}

} // namespace
