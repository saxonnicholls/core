//
//  native_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/money/native.hpp>: the decimals against each protocol's specification, the coin types
//  against the SLIP-44 registry (github.com/satoshilabs/slips, slip-0044.md).
//

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/native.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

template <typename T>
constexpr bool is(std::string_view code, unsigned scale, std::uint32_t slip44)
{
    return asset_like<T> && has_name<T> && T::code() == code && T::scale() == scale && T::slip44() == slip44 &&
           asset_v<T>.code() == code;
}

static_assert(is<native::BTC>("BTC", 8, 0));
static_assert(is<native::LTC>("LTC", 8, 2));
static_assert(is<native::DOGE>("DOGE", 8, 3));
static_assert(is<native::ETH>("ETH", 18, 60));
static_assert(is<native::XRP>("XRP", 6, 144));
static_assert(is<native::TRX>("TRX", 6, 195));
static_assert(is<native::SOL>("SOL", 9, 501));
static_assert(is<native::TON>("TON", 9, 607));
static_assert(is<native::BNB>("BNB", 18, 714));
static_assert(is<native::POL>("POL", 18, 966));
static_assert(is<native::ADA>("ADA", 6, 1815));
static_assert(is<native::AVAX>("AVAX", 18, 9000));

TEST(NativeAssets, NamesAreNotEmpty)
{
    EXPECT_EQ(native::ETH::name(), "Ether");
    EXPECT_EQ(native::BTC::name(), "Bitcoin");
}

} // namespace
