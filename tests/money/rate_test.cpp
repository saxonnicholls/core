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

#include <array>
#include <cstdint>
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

TEST(RateRoundsAway, AppliesEveryModeToTheSignedFraction)
{
    const residual none{false, 0, 10}, below{false, 4, 10}, tie{false, 5, 10}, above{false, 6, 10};
    for (const auto m : {rounding::half_even, rounding::half_up, rounding::half_down, rounding::toward_zero,
                         rounding::away_from_zero, rounding::floor, rounding::ceiling})
        EXPECT_FALSE(rounds_away(3, none, m));   // nothing left: never rounds
    EXPECT_FALSE(rounds_away(3, below, rounding::half_up));
    EXPECT_TRUE(rounds_away(3, above, rounding::half_down));
    EXPECT_TRUE(rounds_away(3, tie, rounding::half_up));
    EXPECT_FALSE(rounds_away(3, tie, rounding::half_down));
    EXPECT_TRUE(rounds_away(3, tie, rounding::half_even));    // 3.5 to 4
    EXPECT_FALSE(rounds_away(2, tie, rounding::half_even));   // 2.5 to 2
    EXPECT_FALSE(rounds_away(3, below, rounding::toward_zero));
    EXPECT_TRUE(rounds_away(3, below, rounding::away_from_zero));
    EXPECT_FALSE(rounds_away(3, below, rounding::floor));
    EXPECT_TRUE(rounds_away(3, below, rounding::ceiling));
    const residual negative{true, 4, 10};   // -3.4: floor goes to -4, ceiling to -3
    EXPECT_TRUE(rounds_away(3, negative, rounding::floor));
    EXPECT_FALSE(rounds_away(3, negative, rounding::ceiling));
    static_assert(rounds_away(0, residual{false, 1, 2}, rounding::half_up));
    static_assert(noexcept(rounds_away(0, residual{}, rounding::half_up)));
}

TEST(RateRoundsAway, AgreesWithConvert)
{
    // 1 FROM at 0 places is n/d TO: convert rounds up exactly when rounds_away says so
    const auto from = *make_asset("FROM", 0);
    const auto to = *make_asset("TO", 0);
    for (std::uint64_t n = 0; n < 40; ++n)
        for (const bool neg : {false, true})
            for (const auto m : {rounding::half_even, rounding::half_up, rounding::floor, rounding::ceiling}) {
                const auto r = *rate::make(from, to, n, 8);
                const auto c = convert(money::from_units(from, 1, neg), r, m);
                ASSERT_TRUE(c);
                const bool away = rounds_away(n / 8, residual{neg, n % 8, 8}, m);
                EXPECT_EQ(c->converted.magnitude(), uint256{n / 8 + (away ? 1 : 0)}) << n << " " << neg;
            }
}

TEST(RateSumResiduals, AddsSignedFractionsExactly)
{
    const std::array parts{residual{false, 3, 10}, residual{true, 7, 10}, residual{false, 9, 10}, residual{true, 0, 10}};
    const auto s = sum_residuals(parts, 10);
    ASSERT_TRUE(s);
    EXPECT_EQ(*s, (residual{false, 5, 10}));   // 3 - 7 + 9 - 0
    const std::array down{residual{true, 9, 10}, residual{true, 8, 10}};
    EXPECT_EQ(*sum_residuals(down, 10), (residual{true, 17, 10}));   // the total can exceed one unit
    const std::array cancel{residual{false, 4, 10}, residual{true, 4, 10}};
    EXPECT_EQ(*sum_residuals(cancel, 10), (residual{false, 0, 10}));   // zero is never negative
    EXPECT_EQ(*sum_residuals({}, 7), (residual{false, 0, 7}));
    constexpr std::array two{residual{false, 1, 3}, residual{false, 1, 3}};
    static_assert(sum_residuals(two, 3)->numerator == uint256{2});
    static_assert(noexcept(sum_residuals(two, 3)));
}

TEST(RateSumResiduals, RefusesAnotherDenominatorAZeroOneAndOverflow)
{
    const std::array mixed{residual{false, 1, 10}, residual{false, 1, 100}};
    EXPECT_EQ(sum_residuals(mixed, 10).err(), money_errc::malformed);
    EXPECT_EQ(sum_residuals({}, 0).err(), money_errc::malformed);
    const std::array huge{residual{false, uint256::max(), uint256::max()}, residual{false, 1, uint256::max()}};
    EXPECT_EQ(sum_residuals(huge, uint256::max()).err(), money_errc::overflow);
}

TEST(MulDiv, SixtyFourBitFactorsGiveTheSameBitsAsTheGeneralPath)
{
    std::uint64_t x = 0x13198a2e03707344ull;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (int i = 0; i < 5000; ++i) {
        const std::uint64_t a = next() >> (i % 64), b = next() >> (i % 59);
        const uint256 d = i % 3 == 0 ? uint256::from_limbs(next() | 1, next()) : uint256{(next() >> (i % 50)) | 1};
        const auto fast = detail::mul_div(a, b, d);
        // a 2^64 * b / (d 2^64): the general path, the same quotient and the remainder shifted up a limb
        const auto general = detail::mul_div(uint256{a} << 64, b, d << 64);
        ASSERT_TRUE(fast && general);
        ASSERT_EQ(fast->first, general->first);
        ASSERT_EQ(fast->second, general->second >> 64);
    }
    static_assert(detail::mul_div(~0ull, ~0ull, 3)->first == uint256::from_limbs(0xaaaaaaaaaaaaaaabull, 0x5555555555555554ull));
}

TEST(RateConvertRate, MatchesTheRoundingTableInEveryMode)
{
    // a positive row of the convert table is a rate priced to to_scale places: (units * numerator) /
    // (denominator * 10^from_scale) to to_scale decimals, with the same quotient and the same residual
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
        if (sign != "+")
            continue;
        const auto n = uint256::checked_mul(*uint256::from_decimal(units), *uint256::from_decimal(num));
        const auto d = uint256::checked_mul(*uint256::from_decimal(den), uint256::pow10(fs));
        if (!n || !d)
            continue;
        const auto r = *rate::make(eth, usd, *n, *d);
        const auto p = convert_rate(r, ts, mode_of(mode));
        ASSERT_TRUE(p) << line;
        EXPECT_EQ(p->rounded.numerator(), *uint256::from_decimal(qu)) << line;
        EXPECT_EQ(p->rounded.denominator(), uint256::pow10(ts)) << line;
        EXPECT_EQ(p->rounded.from(), eth);
        EXPECT_EQ(p->rounded.to(), usd);
        EXPECT_EQ(p->residual.numerator, *uint256::from_decimal(ru)) << line;
        EXPECT_EQ(p->residual.negative, rs == "-" && ru != "0") << line;
        EXPECT_EQ(p->residual.denominator, *d) << line;
        ++cases;
    }
    EXPECT_GT(cases, 200);
}

TEST(RateConvertRate, PricesANavPerUnitInEachMode)
{
    const auto nav = *rate::parse(eth, usd, "1.23456");   // to four places
    const auto at = [&](rounding m) { return convert_rate(nav, 4, m)->rounded.numerator(); };
    EXPECT_EQ(at(rounding::half_even), uint256(12346));
    EXPECT_EQ(at(rounding::half_up), uint256(12346));
    EXPECT_EQ(at(rounding::half_down), uint256(12346));
    const auto tie = *rate::parse(eth, usd, "1.23455");   // 12345.5 units of 10^-4
    EXPECT_EQ(convert_rate(tie, 4, rounding::half_down)->rounded.numerator(), uint256(12345));
    EXPECT_EQ(convert_rate(tie, 4, rounding::half_up)->rounded.numerator(), uint256(12346));
    EXPECT_EQ(convert_rate(tie, 4, rounding::half_even)->rounded.numerator(), uint256(12346));
    EXPECT_EQ(at(rounding::toward_zero), uint256(12345));
    EXPECT_EQ(at(rounding::floor), uint256(12345));
    EXPECT_EQ(at(rounding::away_from_zero), uint256(12346));
    EXPECT_EQ(at(rounding::ceiling), uint256(12346));
    const auto p = convert_rate(nav, 4, rounding::half_even);
    EXPECT_EQ(p->residual, (residual{true, 40000, 100000}));   // 1.23456 - 1.2346 = -0.4 of 10^-4
    const auto exact = convert_rate(nav, 6, rounding::half_even);
    EXPECT_TRUE(exact->residual.is_zero());
    EXPECT_EQ(exact->rounded.numerator(), uint256(1234560));
    EXPECT_EQ(convert_rate(nav, 78, rounding::half_even).err(), money_errc::overflow);
    EXPECT_EQ(convert_rate(*rate::make(eth, usd, uint256::max(), 1), 1, rounding::half_even).err(), money_errc::overflow);
    static_assert(convert_rate(*rate::make(eth, usd, 2, 3), 2, rounding::half_up)->rounded.numerator() == uint256(67));
    static_assert(noexcept(convert_rate(*rate::make(eth, usd, 1, 1), 2, rounding::half_up)));
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
