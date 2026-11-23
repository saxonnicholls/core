//
//  asset_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/asset.hpp>.
//

#include <snicholls/core/interfaces/asset.hpp>
#include <snicholls/core/interfaces/asset_conformance.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace {

using namespace snicholls;

constexpr currency_model<currency::EUR> eur{};
static_assert(eur.numeric() == 978 && eur.scale() == 2 && eur.code() == "EUR");
static_assert(*to_asset(eur) == asset_v<currency::EUR>);
static_assert(*to_asset(native::ETH{}) == asset_v<native::ETH>);

TEST(AssetInterface, MixesTypesAtRunTime)
{
    std::vector<std::unique_ptr<asset_interface>> all;
    all.push_back(std::make_unique<asset_model<native::ETH>>());
    all.push_back(std::make_unique<asset_model<asset>>(*make_asset("USDC:56", 18)));
    all.push_back(std::make_unique<currency_model<currency::AUD>>());
    EXPECT_EQ(all[0]->code(), "ETH");
    EXPECT_EQ(all[1]->code(), "USDC:56");
    EXPECT_EQ(all[1]->scale(), 18);
    EXPECT_EQ(all[2]->scale(), 2);
    for (const auto& a : all)
        EXPECT_FALSE(check_asset(*a)) << a->code();
}

TEST(AssetInterface, TheKitPassesThroughBothForms)
{
    EXPECT_FALSE(check_currency(currency::AUD{}));
    const currency_interface& virtual_form = eur;
    EXPECT_FALSE(check_currency(virtual_form));
    EXPECT_FALSE(check_asset(asset_v<native::SOL>));
}

} // namespace
