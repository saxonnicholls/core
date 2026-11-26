//
//  money_ratio_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/money_ratio.hpp>: the fund case (1,000.00 AUD at a NAV per unit of 1.234567 is
//  exactly 1,000,000,000/1,234,567 units, then 4 places under every rounding mode, and back), signs, zero, overflow,
//  ratio_of, the literals, and the constexpr and noexcept claims.
//

#include <snicholls/core/finance/money_ratio.hpp>
#include <snicholls/core/money/asset.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

using namespace snicholls;
using namespace snicholls::finance;
using namespace snicholls::finance::literals;

namespace {

constexpr asset aud = checked_asset("AUD", 2), units4 = checked_asset("MFUND1:APIR", 4), wei = checked_asset("ETH", 18);
constexpr ratio nav = "1.234567"_ratio;

static_assert(nav == ratio{1234567, 1000000});
static_assert("3/2"_ratio == ratio{3, 2} && "4.35%"_ratio == ratio{87, 2000});
static_assert(noexcept(ratio{1, 2}) && noexcept(ratio{}));
static_assert(noexcept(divide(std::declval<const money&>(), std::declval<const ratio&>(), aud, rounding::half_even)) &&
              noexcept(exact_value(std::declval<const money&>())) && noexcept(ratio_of(money{}, money{})));

// the fund case, at compile time
static_assert(*divide(*parse_money(aud, "1000.00"), nav) == ratio{1'000'000'000, 1'234'567});
static_assert(*divide(*parse_money(aud, "1000.00"), nav, units4, rounding::half_even) == *parse_money(units4, "810.0006"));

TEST(FinanceMoneyRatio, ASubscriptionIsExactUnitsThenRoundedOnce)
{
    const money subscription = *parse_money(aud, "1000.00");
    const auto exact = divide(subscription, nav);
    ASSERT_TRUE(exact.has_value());
    EXPECT_EQ(*exact, (ratio{1'000'000'000, 1'234'567})) << "810.000591...";
    EXPECT_EQ(*to_money(*exact, units4, rounding::half_even), *parse_money(units4, "810.0006"));
    EXPECT_EQ(*divide(subscription, nav, units4, rounding::half_even), *parse_money(units4, "810.0006"));
    EXPECT_EQ(*divide(subscription, nav, units4, rounding::toward_zero), *parse_money(units4, "810.0005"));
    EXPECT_EQ(*divide(subscription, nav, units4, rounding::floor), *parse_money(units4, "810.0005"));
    EXPECT_EQ(*divide(subscription, nav, units4, rounding::ceiling), *parse_money(units4, "810.0006"));
    // and back: the units issued, priced at the NAV, to the cent
    const money issued = *parse_money(units4, "810.0005");
    EXPECT_EQ(*multiply(issued, nav), (ratio{810'0005LL * 1'234'567, 10'000LL * 1'000'000}));
    EXPECT_EQ(*multiply(issued, nav, aud, rounding::half_even), *parse_money(aud, "1000.00"));   // 999.99988...
    EXPECT_EQ(*multiply(issued, nav, aud, rounding::toward_zero), *parse_money(aud, "999.99"));
}

TEST(FinanceMoneyRatio, SignsZeroAndTies)
{
    const money refund = -*parse_money(aud, "10.00");
    EXPECT_EQ(*multiply(refund, ratio{1, 3}, aud, rounding::floor), -*parse_money(aud, "3.34"));
    EXPECT_EQ(*multiply(refund, ratio{1, 3}, aud, rounding::ceiling), -*parse_money(aud, "3.33"));
    EXPECT_EQ(*multiply(refund, ratio{-1, 4}, aud, rounding::half_even), *parse_money(aud, "2.50"));
    EXPECT_EQ(*multiply(*parse_money(aud, "0.05"), ratio{1, 2}, aud, rounding::half_even), *parse_money(aud, "0.02"));
    EXPECT_EQ(*multiply(*parse_money(aud, "0.05"), ratio{1, 2}, aud, rounding::half_up), *parse_money(aud, "0.03"));
    EXPECT_FALSE(multiply(*parse_money(aud, "-0.01"), ratio{1, 3}, aud, rounding::toward_zero)->is_negative())
        << "a zero result is never negative";
    EXPECT_EQ(*divide(money::zero(aud), nav), ratio{});
    EXPECT_FALSE(divide(*parse_money(aud, "1.00"), ratio{}).has_value());
    EXPECT_FALSE(divide(*parse_money(aud, "1.00"), ratio{}, units4, rounding::half_even).has_value());
    EXPECT_EQ(*to_money(ratio{-1, 3}, aud, rounding::half_even), -*parse_money(aud, "0.33"));
}

TEST(FinanceMoneyRatio, LargeAmountsUseTheMoneyForms)
{
    const money ether = *parse_money(wei, "12345678.123456789012345678");
    EXPECT_FALSE(exact_value(ether).has_value()) << "beyond 63 bits of wei";
    EXPECT_FALSE(multiply(ether, ratio{1, 2}).has_value());
    EXPECT_EQ(*multiply(ether, ratio{1, 2}, wei, rounding::half_even), *parse_money(wei, "6172839.061728394506172839"));
    EXPECT_FALSE(multiply(money::from_units(wei, uint256::max()), ratio{3, 1}, wei, rounding::half_even).has_value());
}

TEST(FinanceMoneyRatio, TheRatioOfTwoAmountsIsTheNavPerUnit)
{
    const money net_assets = *parse_money(aud, "1234567.00"), outstanding = *parse_money(units4, "1000000.0000");
    EXPECT_EQ(*ratio_of(net_assets, outstanding), nav);
    EXPECT_FALSE(ratio_of(net_assets, money::zero(units4)).has_value());
    EXPECT_EQ(*exact_value(*parse_money(aud, "-12.34")), (ratio{-617, 50}));
}

} // namespace
