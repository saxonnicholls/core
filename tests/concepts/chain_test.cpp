//
//  chain_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/chain.hpp>: who models chain_like and evm_chain_like, and who does not.
//

#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/concepts/chain.hpp>
#include <snicholls/core/interfaces/chain.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

struct no_native {
    static constexpr std::string_view name() noexcept { return "x"; }
    static constexpr chain_kind kind() noexcept { return chain_kind::other; }
};
struct native_not_an_asset {
    static constexpr std::string_view name() noexcept { return "x"; }
    static constexpr chain_kind kind() noexcept { return chain_kind::other; }
    static constexpr int native_asset() noexcept { return 0; }
};

static_assert(chain_like<chain::ethereum> && evm_chain_like<chain::ethereum>);
static_assert(chain_like<chain::bitcoin> && !evm_chain_like<chain::bitcoin>);
static_assert(chain_like<chain_interface> && !evm_chain_like<chain_interface>, "the interface's chain_id is optional");
static_assert(chain_like<chain_model<chain::base>>);
static_assert(!chain_like<no_native> && !chain_like<native_not_an_asset> && !chain_like<int>);

TEST(ChainConcepts, KindsAreDistinct)
{
    EXPECT_NE(chain::ethereum::kind(), chain::bitcoin::kind());
    EXPECT_NE(chain::tron::kind(), chain::ethereum::kind());
}

} // namespace
