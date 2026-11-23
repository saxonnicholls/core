//
//  onramp_provider_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/onramp_provider.hpp>: the models and the non-models, EVM addresses, exact value comparison and fact validation.
//

#include <snicholls/core/concepts/onramp_provider.hpp>
#include <snicholls/core/interfaces/onramp_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <gtest/gtest.h>

#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

static_assert(onramp_provider<memory_onramp_provider>);
static_assert(onramp_provider<onramp_provider_interface>);
static_assert(onramp_provider<onramp_provider_adapter<memory_onramp_provider>>);
static_assert(onramp_provider<onramp_provider_adapter<memory_onramp_provider&>>);
static_assert(onramp_provider<onramp_provider_ref>);

struct no_statement {
    std::vector<onramp_asset> assets() const;
    capability_result<onramp_quote> quote(const onramp_quote_request&);
    capability_result<order> buy(const buy_request&);
    capability_result<order> order_status(std::string_view);
    capability_result<withdrawal> withdraw(const withdraw_request&);
    capability_result<withdrawal> withdrawal_status(std::string_view);
    capability_result<std::vector<venue_balance>> balances();
};
static_assert(!onramp_provider<no_statement>);
static_assert(!onramp_provider<int>);

static_assert(is_evm_address("0x52908400098527886E0F7030069857D2E4169EE7"));
static_assert(!is_evm_address("0x1234") && !is_evm_address("52908400098527886E0F7030069857D2E4169EE7aa"));
static_assert(!is_evm_address("0x52908400098527886E0F7030069857D2E4169EEg"));

constexpr asset usdc = checked_asset("USDC", 6), usdc18 = checked_asset("USDC:MEM", 18), aud = checked_asset("AUD", 2);
static_assert(same_value(money::from_units(usdc, 1), money::from_units(usdc18, uint256::pow10(12))));
static_assert(!same_value(money::from_units(usdc, 1), money::from_units(usdc18, 1)));
static_assert(!same_value(money::from_units(usdc, 1), money::from_units(usdc, 1, true)));

onramp_fact fill()
{
    onramp_fact f;
    f.source = "okx";
    f.fact_id = "t-1";
    f.account = "trading";
    f.kind = onramp_fact_kind::trade_fill;
    f.debit = money::from_units(aud, 153);
    f.credit = money::from_units(usdc18, uint256::pow10(18));
    f.fee = money::zero(aud);
    f.price = *rate::parse(usdc, aud, "1.53");
    f.order_id = "o-1";
    return f;
}

TEST(OnrampProviderConcept, ValidateRefusesFactsTheLedgerCouldNotBook)
{
    EXPECT_TRUE(validate(fill()).is_ok());
    onramp_fact f = fill();
    f.price.reset();
    EXPECT_TRUE(validate(f).is_err());
    f = fill();
    f.credit = money::from_units(aud, 1);
    EXPECT_EQ(validate(f).unwrap_err().message, std::string_view{"a fill moves two assets"});
    f = fill();
    f.fee = money::from_units(aud, 1, true);
    EXPECT_TRUE(validate(f).is_err());
    f = fill();
    f.fact_id = "";
    EXPECT_TRUE(validate(f).is_err());

    onramp_fact w = fill();
    w.kind = onramp_fact_kind::withdrawal;
    w.debit = money::from_units(usdc18, uint256::pow10(12));
    w.credit = money::from_units(usdc, 1);
    w.chain_id = 8453;
    w.address = "0x52908400098527886E0F7030069857D2E4169EE7";
    w.withdrawal_id = "w-1";
    EXPECT_TRUE(validate(w).is_ok());
    w.credit = money::from_units(usdc, 2);
    EXPECT_TRUE(validate(w).is_err()) << "not 1:1";

    onramp_fact t = fill();
    t.kind = onramp_fact_kind::internal_transfer;
    t.credit = t.debit;
    t.to_account = "funding";
    EXPECT_TRUE(validate(t).is_ok());
    t.to_account = "trading";
    EXPECT_TRUE(validate(t).is_err());
}

} // namespace
