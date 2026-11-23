//
//  chain_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/chain_conformance.hpp>: the kit catches each broken rule.
//

#include <snicholls/core/interfaces/chain_conformance.hpp>
#include <snicholls/core/money/native.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

struct zero_id {
    static constexpr std::string_view name() noexcept { return "Zero"; }
    static constexpr chain_kind kind() noexcept { return chain_kind::evm; }
    static constexpr std::uint64_t chain_id() noexcept { return 0; }
    static constexpr native::ETH native_asset() noexcept { return {}; }
};
struct bad_native {
    struct coin {
        static constexpr std::string_view code() noexcept { return "eth"; }
        static constexpr std::uint8_t scale() noexcept { return 18; }
    };
    static constexpr std::string_view name() noexcept { return "Bad"; }
    static constexpr chain_kind kind() noexcept { return chain_kind::other; }
    static constexpr coin native_asset() noexcept { return {}; }
};

TEST(ChainConformance, NamesTheFailedCheck)
{
    EXPECT_EQ(*check_chain(zero_id{}), "chain: chain_id() is zero");
    EXPECT_EQ(*check_chain(bad_native{}), "asset: code() is not 1 to 24 of A-Z 0-9 : . - _");
}

} // namespace
