//
//  rate_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/money/rate.hpp>. convert is checked against tests/fixtures/money/convert.txt, written by
//  scripts/generate_money_fixtures.py from Python's decimal, for every rounding mode, with the exact residual.
//

#include <snicholls/core/detail/mul_div.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>
#include <snicholls/core/money/rate.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

namespace {

using namespace snicholls;
using snicholls::literals::operator""_iso4217;

constexpr asset usd = "USD"_iso4217;
constexpr asset eth = asset_v<native::ETH>;

static_assert(noexcept(convert(money{}, *rate::make(eth, usd, 1, 1), rounding::half_even)));

rounding mode_of(const std::string& s)
{
    if (s == "half_even") return rounding::half_even;
    if (s == "half_up") return rounding::half_up;
    if (s == "half_down") return rounding::half_down;
    if (s == "toward_zero") return rounding::toward_zero;
    if (s == "away_from_zero") return rounding::away_from_zero;
    if (s == "floor") return rounding::floor;
    return rounding::ceiling;
}

TEST(RateParse, HoldsMoreThanEighteenDigitsExactly)
{
    const auto r = rate::parse(eth, usd, "3000.123456789012345678");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->numerator(), *uint256::from_decimal("3000123456789012345678"));
    EXPECT_EQ(r->denominator(), uint256::pow10(18));
    const auto c = convert(*parse_money(eth, "1"), *r, rounding::half_even);
    ASSERT_TRUE(c);
    EXPECT_EQ(to_string(c->converted), "3000.12");
    EXPECT_FALSE(c->residual.is_zero());
    EXPECT_EQ(rate::parse(eth, usd, "1.2.3").err(), money_errc::malformed);
    EXPECT_EQ(rate::parse(eth, usd, "-1").err(), money_errc::malformed);
    EXPECT_EQ(rate::parse(eth, usd, "").err(), money_errc::malformed);
    EXPECT_EQ(rate::make(eth, usd, 1, 0).err(), money_errc::malformed);
}

TEST(RateFromDecimal, IsExact)
{
    const auto r = rate::from_decimal(eth, usd, *decimal::parse("3000.25"));
    ASSERT_TRUE(r);
    EXPECT_EQ(r->numerator(), uint256(300025));
    EXPECT_EQ(r->denominator(), uint256(100));
    EXPECT_EQ(rate::from_decimal(eth, usd, *decimal::parse("-1")).err(), money_errc::malformed);
    EXPECT_EQ(rate::from_decimal(eth, usd, *decimal::parse("2000000"))->numerator(), uint256(2000000));
}

TEST(RateInverse, SwapsTheAssetsAndTheRatio)
{
    const auto r = *rate::make(eth, usd, 3, 2);
    const auto i = r.inverse();
    ASSERT_TRUE(i);
    EXPECT_EQ(i->from(), usd);
    EXPECT_EQ(i->to(), eth);
    EXPECT_EQ(i->numerator(), uint256(2));
    EXPECT_EQ(rate::make(eth, usd, 0, 1)->inverse().err(), money_errc::malformed);
}

TEST(RateConvert, RefusesTheWrongSourceAsset)
{
    const auto r = *rate::make(eth, usd, 1, 1);
    EXPECT_EQ(convert(*parse_money(usd, "1"), r, rounding::half_even).err(), money_errc::asset_mismatch);
}

TEST(RateConvert, ResidualAtAFinerScale)
{
    const auto r = *rate::make(usd, usd, 1, 3);   // a third
    const auto c = *convert(*parse_money(usd, "1"), r, rounding::half_even);
    EXPECT_EQ(to_string(c.converted), "0.33");
    const auto fine = c.residual.at_scale(usd, 6);
    ASSERT_TRUE(fine);
    EXPECT_EQ(fine->asset_of().scale(), 8);
    EXPECT_EQ(to_string(*fine), "0.00333333");   // a third of a cent, truncated at six more places
}

TEST(RateConvert, TypedConversion)
{
    const auto r = *rate::make(eth, usd, 2000, 1);
    const auto got = convert_to<currency::USD>(money_of<native::ETH>::from_units(uint256::pow10(18)), r, rounding::floor);
    ASSERT_TRUE(got);
    EXPECT_EQ(got->first.magnitude(), uint256(200000));
    EXPECT_EQ(convert_to<currency::EUR>(money_of<native::ETH>::from_units(1), r, rounding::floor).err(),
              money_errc::asset_mismatch);
}

TEST(RateConvert, MatchesDecimalInEveryModeWithTheExactResidual)
{
    std::ifstream in(SNICHOLLS_CORE_FIXTURE_DIR "/money/convert.txt");
    ASSERT_TRUE(in) << "tests/fixtures/money/convert.txt";
    std::string line;
    int cases = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream is(line);
        unsigned fs = 0, ts = 0;
        std::string num, den, sign, units, mode, bar, qs, qu, rs, ru;
        is >> fs >> ts >> num >> den >> sign >> units >> mode >> bar >> qs >> qu >> rs >> ru;
        const auto from = *make_asset("FROM", fs);
        const auto to = *make_asset("TO", ts);
        const auto r = *rate::make(from, to, *uint256::from_decimal(num), *uint256::from_decimal(den));
        const auto amount = money::from_units(from, *uint256::from_decimal(units), sign == "-");
        const auto c = convert(amount, r, mode_of(mode));
        ASSERT_TRUE(c) << line;
        EXPECT_EQ(c->converted, money::from_units(to, *uint256::from_decimal(qu), qs == "-")) << line;
        // the residual is (exact - converted) * d / d, with d = denominator * 10^from_scale
        EXPECT_EQ(c->residual.numerator, *uint256::from_decimal(ru)) << line;
        EXPECT_EQ(c->residual.negative, rs == "-" && ru != "0") << line;
        EXPECT_EQ(c->residual.denominator, r.denominator() * uint256::pow10(fs)) << line;
        EXPECT_LT(c->residual.numerator, c->residual.denominator) << line;
        ++cases;
    }
    EXPECT_GT(cases, 500);
}

TEST(MulDiv, DividesTheFull512BitProduct)
{
    const auto big = uint256::max();
    const auto qr = detail::mul_div(big, big, big);
    ASSERT_TRUE(qr);
    EXPECT_EQ(qr->first, big);
    EXPECT_TRUE(qr->second.is_zero());
    EXPECT_FALSE(detail::mul_div(big, big, 1));   // the quotient needs 512 bits
    EXPECT_FALSE(detail::mul_div(1, 1, 0));
    const auto small = detail::mul_div(7, 3, 2);
    EXPECT_EQ(small->first, uint256(10));
    EXPECT_EQ(small->second, uint256(1));
    static_assert(detail::mul_div(uint256::pow10(40), uint256::pow10(40), uint256::pow10(70))->first == uint256::pow10(10));
}

} // namespace
