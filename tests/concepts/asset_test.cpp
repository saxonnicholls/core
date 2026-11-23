//
//  asset_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/asset.hpp>: who models asset_like and currency_like, and who does not.
//

#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/interfaces/asset.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

struct code_only {
    static constexpr std::string_view code() noexcept { return "X"; }
};
struct scale_only {
    static constexpr std::uint8_t scale() noexcept { return 2; }
};
struct wrong_code {
    static constexpr int code() noexcept { return 0; }
    static constexpr std::uint8_t scale() noexcept { return 2; }
};

static_assert(asset_like<asset>, "the value is the reference implementation");
static_assert(asset_like<currency::AUD> && currency_like<currency::AUD>);
static_assert(asset_like<native::ETH> && !currency_like<native::ETH>, "ETH has no ISO numeric code");
static_assert(asset_like<asset_interface> && currency_like<currency_interface>);
static_assert(asset_like<asset_model<native::ETH>> && currency_like<currency_model<currency::EUR>>);
static_assert(!asset_like<currency::XAU> && asset_like<with_scale<currency::XAU, 18>>);
static_assert(!asset_like<code_only> && !asset_like<scale_only> && !asset_like<wrong_code>);
static_assert(!currency_like<asset>);

TEST(AssetConcepts, TheValueAndTheTypesAgree)
{
    EXPECT_EQ(asset_v<currency::AUD>.code(), currency::AUD{}.code());
    EXPECT_EQ(asset_v<currency::AUD>.scale(), currency::AUD{}.scale());
}

} // namespace
