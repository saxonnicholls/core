//
//  security_provider_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/security_provider.hpp>: the models and the non-models, the kind predicates and
//  their stable values, instrument codes, cash_movement, and every rule of validate(security_fact), at compile time
//  where it can be.
//

#include <snicholls/core/concepts/security_provider.hpp>
#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/interfaces/security_provider_reference.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/money/asset.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

using namespace snicholls;

namespace {

static_assert(security_provider<memory_security_provider>);
static_assert(security_provider<security_provider_interface>);
static_assert(security_provider<security_provider_adapter<memory_security_provider>>);
static_assert(security_provider<security_provider_adapter<memory_security_provider&>>);
static_assert(security_provider<security_provider_ref>);

struct no_holdings {
    security_capabilities capabilities() const;
    capability_result<webhook_event> verify_webhook(const webhook_delivery&) const;
    capability_result<std::vector<security_fact>> resolve(const webhook_event&);
    capability_result<security_statement_page> statement(const security_statement_query&);
};
struct payment_facts {
    security_capabilities capabilities() const;
    capability_result<webhook_event> verify_webhook(const webhook_delivery&) const;
    capability_result<std::vector<payment_fact>> resolve(const webhook_event&);
    capability_result<security_statement_page> statement(const security_statement_query&);
    capability_result<std::vector<security_holding>> holdings(std::chrono::sys_days);
};
static_assert(!security_provider<no_holdings>);
static_assert(!security_provider<payment_facts>);
static_assert(!security_provider<memory_payment_provider>);
static_assert(!security_provider<int>);

using enum security_fact_kind;
static_assert(static_cast<int>(buy) == 1 && static_cast<int>(collateral_reuse) == 27 &&
              static_cast<int>(application) == 28 && static_cast<int>(distribution_reinvested) == 31,
              "the wire values never move");
static_assert(enum_count_v<security_fact_kind> == 31);
static_assert(static_cast<int>(event_kind::security_updated) == 6 && static_cast<int>(event_kind::other) == 5,
              "appended to the payment contract's kinds");
static_assert(is_repo(repo_open) && is_repo(reverse_repo_close) && is_repo(margin_call) && !is_repo(interest_accrual) &&
              !is_repo(buy));
static_assert(is_corporate_action(dividend_entitlement) && is_corporate_action(scrip_takeover) && !is_corporate_action(settle) &&
              !is_corporate_action(repo_open) && !is_corporate_action(application));
static_assert(needs_other_instrument(demerger) && needs_other_instrument(unit_switch) && !needs_other_instrument(merger));
static_assert(needs_ratio(split) && needs_ratio(rights_issue) && !needs_ratio(demerger));

static_assert(is_instrument_code("BHP:ASX") && is_instrument_code("VAS:XASX") && is_instrument_code("USDC:8453") &&
              is_instrument_code("BRK.B:XNYS"));
static_assert(!is_instrument_code("BHP") && !is_instrument_code(":ASX") && !is_instrument_code("BHP:") &&
              !is_instrument_code("BHP:asx") && !is_instrument_code("A:B:C") && !is_instrument_code(""));

constexpr asset bhp = checked_asset("BHP:ASX", 0), s32 = checked_asset("S32:ASX", 0), aud = checked_asset("AUD", 2);
constexpr asset aud_mills = checked_asset("AUD", 3), usd = checked_asset("USD", 2);
constexpr money units(asset a, std::int64_t v) { return money::from_units(a, static_cast<std::uint64_t>(v < 0 ? -v : v), v < 0); }
constexpr std::chrono::sys_days on(int y, unsigned m, unsigned d)
{
    return std::chrono::sys_days{std::chrono::year{y} / std::chrono::month{m} / std::chrono::day{d}};
}

constexpr security_fact a_buy()
{
    security_fact f;
    f.source = "broker";
    f.fact_id = "fill_1";
    f.kind = buy;
    f.instrument = bhp;
    f.quantity = units(bhp, 100);
    f.price = units(aud_mills, 45'125);         // 45.125
    f.consideration = units(aud, -451'250);     // paid
    f.fees = units(aud, 995);
    f.taxes.duty = units(aud, 0);
    f.dates.trade_at = utc_s::from_unix_seconds(1'800'000'000);
    f.dates.known_at = f.dates.trade_at;
    f.dates.settles_on = f.dates.trade_at.days() + std::chrono::days{2};
    return f;
}

static_assert(validate(a_buy()).is_ok());
static_assert(*cash_movement(a_buy()) == units(aud, -452'245));
static_assert(noexcept(validate(std::declval<const security_fact&>())) && noexcept(cash_movement(std::declval<const security_fact&>())));

template <class Change>
constexpr std::string_view refused(security_fact f, Change change)
{
    change(f);
    const auto r = validate(f);
    return r.is_ok() ? std::string_view{} : r.unwrap_err_unchecked().message.view();
}

TEST(SecurityProviderConcept, ValidateRefusesWhatTheLedgerCouldNotBook)
{
    const security_fact f = a_buy();
    EXPECT_EQ(refused(f, [](auto& g) { g.fact_id = {}; }), "a fact needs its source and fact id");
    EXPECT_EQ(refused(f, [](auto& g) { g.kind = static_cast<security_fact_kind>(0); }), "not a security fact kind");
    EXPECT_EQ(refused(f, [](auto& g) { g.kind = static_cast<security_fact_kind>(32); }), "not a security fact kind");
    EXPECT_EQ(refused(f, [](auto& g) { g.dates.known_at = utc_s{}; }), "a fact is known before it happened");
    EXPECT_EQ(refused(f, [](auto& g) { g.instrument = aud; g.quantity = units(aud, 1); }), "an instrument is <SYMBOL>:<VENUE>");
    EXPECT_EQ(refused(f, [](auto& g) { g.quantity = units(s32, 100); }), "the quantity is in the instrument");
    EXPECT_EQ(refused(f, [](auto& g) { g.isin = isin_text{"AU000000BHP5"}; }), "not a valid ISIN");
    EXPECT_EQ(refused(f, [](auto& g) { g.isin = isin_text{"AU000000BHP4"}; }), "");
    EXPECT_EQ(refused(f, [](auto& g) { g.fees = units(aud, -1); }), "fees and taxes are not negative, in the cash currency");
    EXPECT_EQ(refused(f, [](auto& g) { g.fees = units(usd, 1); }), "fees and taxes are not negative, in the cash currency");
    EXPECT_EQ(refused(f, [](auto& g) { g.taxes.withheld = units(usd, 1); }), "fees and taxes are not negative, in the cash currency");
    EXPECT_EQ(refused(f, [](auto& g) { g.price = units(usd, 4512); }), "a price is not negative, in the cash currency");
    EXPECT_EQ(refused(f, [](auto& g) { g.price = units(checked_asset("AUD", 1), 451); }),
              "a price is not negative, in the cash currency") << "coarser than the currency";
    EXPECT_EQ(refused(f, [](auto& g) { g.quantity = units(bhp, -100); }), "a purchase adds units and pays; a sale removes and receives");
    EXPECT_EQ(refused(f, [](auto& g) { g.consideration = units(aud, 451'250); }),
              "a purchase adds units and pays; a sale removes and receives");
    EXPECT_EQ(refused(f, [](auto& g) { g.kind = sell; }), "a purchase adds units and pays; a sale removes and receives");
    EXPECT_EQ(refused(f, [](auto& g) { g.kind = sell; g.quantity = units(bhp, -100); g.consideration = units(aud, 451'250); }), "");
    EXPECT_EQ(refused(f, [](auto& g) { g.taxes.franking_credit = units(aud, 10); }), "franking is a dividend's");
    EXPECT_EQ(refused(f, [](auto& g) { g.record_date = on(2027, 1, 1); }), "ex, record and pay dates are a corporate action's");
}

security_fact action(security_fact_kind k)
{
    security_fact f = a_buy();
    f.kind = k;
    f.quantity = units(bhp, 0);
    f.price.reset();
    f.consideration = units(aud, 0);
    f.fees = units(aud, 0);
    f.record_date = on(2027, 3, 6);
    return f;
}

TEST(SecurityProviderConcept, CorporateActionsCarryTheirTermsAndDates)
{
    security_fact dividend = action(dividend_paid);
    dividend.consideration = units(aud, 72'000);
    dividend.taxes.franking_credit = units(aud, 30'857);
    dividend.taxes.franked = finance::ratio{1, 1};
    dividend.ex_date = on(2027, 3, 5);
    dividend.pay_date = on(2027, 3, 26);
    EXPECT_TRUE(validate(dividend).is_ok());
    EXPECT_EQ(*cash_movement(dividend), units(aud, 72'000)) << "a franking credit is never cash";
    EXPECT_EQ(refused(dividend, [](auto& g) { g.taxes.franked = finance::ratio{3, 2}; }), "the franked part is from 0 to 1");
    EXPECT_EQ(refused(dividend, [](auto& g) { g.ex_date = on(2027, 3, 7); }), "ex, then record, then pay");
    EXPECT_EQ(refused(dividend, [](auto& g) { g.pay_date = on(2027, 3, 1); }), "ex, then record, then pay");
    EXPECT_EQ(refused(dividend, [](auto& g) { g.record_date.reset(); g.pay_date = on(2027, 3, 1); }), "ex, then record, then pay");
    EXPECT_EQ(refused(action(dividend_entitlement), [](auto& g) { g.record_date.reset(); }), "an entitlement has its record date");

    security_fact three_for_two = action(split);
    three_for_two.quantity = units(bhp, 50);
    three_for_two.new_for_old = finance::ratio{3, 2};
    EXPECT_TRUE(validate(three_for_two).is_ok());
    EXPECT_EQ(refused(three_for_two, [](auto& g) { g.new_for_old.reset(); }), "this action has a new-for-old ratio");
    EXPECT_EQ(refused(three_for_two, [](auto& g) { g.new_for_old = finance::ratio{1, 2}; }),
              "a ratio is positive, a split's above 1, a consolidation's below");
    security_fact one_for_five = action(consolidation);
    one_for_five.new_for_old = finance::ratio{1, 5};
    EXPECT_TRUE(validate(one_for_five).is_ok());
    EXPECT_EQ(refused(one_for_five, [](auto& g) { g.new_for_old = finance::ratio{5, 1}; }),
              "a ratio is positive, a split's above 1, a consolidation's below");
    EXPECT_EQ(refused(action(bonus), [](auto& g) { g.new_for_old = finance::ratio{}; }),
              "a ratio is positive, a split's above 1, a consolidation's below");

    security_fact spin_off = action(demerger);
    spin_off.other_instrument = s32;
    spin_off.other_quantity = units(s32, 100);
    spin_off.cost_apportionment = *finance::ratio::parse("13.24%");
    EXPECT_TRUE(validate(spin_off).is_ok());
    EXPECT_EQ(refused(spin_off, [](auto& g) { g.cost_apportionment.reset(); }), "a demerger apportions the cost base");
    EXPECT_EQ(refused(spin_off, [](auto& g) { g.cost_apportionment = finance::ratio{-1, 10}; }), "a cost apportionment is from 0 to 1");
    EXPECT_EQ(refused(spin_off, [](auto& g) { g.other_instrument.reset(); g.other_quantity.reset(); }),
              "this action names its other instrument");
    EXPECT_EQ(refused(spin_off, [](auto& g) { g.other_instrument = bhp; g.other_quantity = units(bhp, 1); }),
              "the other instrument is another <SYMBOL>:<VENUE>");
    EXPECT_EQ(refused(spin_off, [](auto& g) { g.other_quantity = units(bhp, 1); }), "the other quantity is in the other instrument");
    EXPECT_EQ(refused(a_buy(), [](auto& g) { g.cost_apportionment = finance::ratio{1, 2}; }),
              "a cost apportionment goes to the other instrument");
    EXPECT_EQ(refused(a_buy(), [](auto& g) { g.other_quantity = units(s32, 1); }), "the other quantity is in the other instrument");
}

TEST(SecurityProviderConcept, ReposCarryTheirTermsAndFundsDeal)
{
    security_fact repo = a_buy();
    repo.kind = repo_open;
    repo.quantity = units(bhp, -10'000);          // the securities go out
    repo.consideration = units(aud, 45'000'000);  // the cash comes in
    repo.price.reset();
    repo.repo = repo_terms{"5493001KJTIIGC8Y1R12", "gmra-2026-001", *finance::ratio::parse("4.35%"),
                           finance::day_count_basis::act_365_fixed, on(2027, 2, 15)};
    EXPECT_TRUE(validate(repo).is_ok());
    EXPECT_EQ(refused(repo, [](auto& g) { g.repo.reset(); }), "a repo fact carries its terms");
    EXPECT_EQ(refused(repo, [](auto& g) { g.repo->agreement_id = {}; }), "a repo names its counterparty and agreement");
    EXPECT_EQ(refused(repo, [](auto& g) { g.repo->day_count = static_cast<finance::day_count_basis>(9); }), "not a day-count basis");
    EXPECT_EQ(refused(repo, [](auto& g) { g.repo->maturity = on(2020, 1, 1); }), "a repo matures after it opens");
    EXPECT_EQ(refused(repo, [](auto& g) { g.repo->maturity.reset(); }), "") << "an open repo has no maturity";

    security_fact apply = a_buy();
    apply.kind = application;
    apply.instrument = checked_asset("MFUND1:APIR", 4);
    apply.quantity = units(apply.instrument, 0);   // units unknown until priced
    apply.price.reset();
    EXPECT_TRUE(validate(apply).is_ok());
    EXPECT_EQ(refused(apply, [](auto& g) { g.consideration = units(aud, 1); }),
              "a purchase adds units and pays; a sale removes and receives");
    security_fact redeem = apply;
    redeem.kind = redemption;
    redeem.quantity = units(apply.instrument, -1'000'000);
    redeem.consideration = units(aud, 1'012'345);
    EXPECT_TRUE(validate(redeem).is_ok());
    security_fact to_other_fund = redeem;
    to_other_fund.kind = unit_switch;
    EXPECT_EQ(refused(to_other_fund, [](auto&) {}), "this action names its other instrument");
    to_other_fund.other_instrument = checked_asset("MFUND2:APIR", 4);
    EXPECT_TRUE(validate(to_other_fund).is_ok());
}

TEST(SecurityProviderConcept, KindsPrintByName)
{
    EXPECT_EQ(enum_name(dividend_reinvested), "dividend_reinvested");
    EXPECT_EQ(enum_name(unit_switch), "unit_switch");
    EXPECT_EQ(enum_name(event_kind::security_updated), "security_updated");
}

} // namespace
