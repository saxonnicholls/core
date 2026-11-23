//
//  money_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/money/money.hpp>. Parsing and formatting are checked against
//  tests/fixtures/money/parse.txt, written by scripts/generate_money_fixtures.py from Python's decimal.
//

#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/money/native.hpp>

#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;
using snicholls::literals::operator""_iso4217;

constexpr asset aud = "AUD"_iso4217;
constexpr asset eth = asset_v<native::ETH>;
constexpr asset usdt = *make_asset("USDT", 6);

money m(const asset& a, std::string_view text)
{
    auto r = parse_money(a, text);
    EXPECT_TRUE(r.is_ok()) << text;
    return r ? *r : money{};
}

std::string text(const money& x, money_digits d = money_digits::minimal) { return to_string(x, d); }

// layout and the zero-overhead contract
static_assert(std::is_trivially_copyable_v<money>);
static_assert(std::is_trivially_copyable_v<money_of<currency::AUD>>);
static_assert(sizeof(money_of<currency::AUD>) == sizeof(uint256) + alignof(uint256), "money_of: a sign and a uint256");
static_assert(sizeof(money) <= sizeof(uint256) + 24, "money: the asset, a sign and a uint256");
static_assert(noexcept(add(money{}, money{})));
static_assert(noexcept(parse_money(aud, std::string_view{})));

// compile-time arithmetic
constexpr auto compile_time = [] {
    const auto a = *parse_money(eth, "1000.5");
    const auto b = *parse_money(eth, "-0.5");
    return *add(a, b);
}();
static_assert(compile_time == money::from_units(eth, uint256::pow10(21)));

TEST(MoneyRepresentation, ZeroIsNeverNegative)
{
    EXPECT_FALSE(money::from_units(aud, 0, true).is_negative());
    EXPECT_EQ(money::from_units(aud, 0, true), money::zero(aud));
    EXPECT_EQ(-money::zero(aud), money::zero(aud));
    EXPECT_EQ(money::from_units(aud, 5, true).signum(), -1);
    EXPECT_EQ(money::zero(aud).signum(), 0);
}

TEST(MoneyRepresentation, HoldsEveryUint256EitherSign)
{
    const auto top = money::from_units(eth, uint256::max(), true);
    EXPECT_EQ(top.magnitude(), uint256::max());
    EXPECT_TRUE(top.is_negative());
    const auto back = parse_money(eth, text(top));
    ASSERT_TRUE(back);
    EXPECT_EQ(*back, top);   // the maximum in wei round-trips at scale 18
    const auto at0 = money::from_units(*make_asset("WEI", 0), uint256::max());
    EXPECT_EQ(text(at0), uint256::max().to_string());
}

TEST(MoneyArithmetic, AddsAndSubtractsWithinOneAsset)
{
    EXPECT_EQ(text(*add(m(aud, "10.25"), m(aud, "-3.30"))), "6.95");
    EXPECT_EQ(text(*add(m(aud, "-10.25"), m(aud, "3.30"))), "-6.95");
    EXPECT_EQ(text(*subtract(m(aud, "1"), m(aud, "1"))), "0");
    EXPECT_EQ(text(*subtract(m(aud, "1"), m(aud, "2.5"))), "-1.5");
    EXPECT_EQ(text(*multiply(m(usdt, "1.5"), 3)), "4.5");
}

TEST(MoneyArithmetic, RefusesMixedAssetsAndOverflow)
{
    EXPECT_EQ(add(m(aud, "1"), m(eth, "1")).err(), money_errc::asset_mismatch);
    EXPECT_EQ(compare(m(aud, "1"), m(eth, "1")).err(), money_errc::asset_mismatch);
    const auto top = money::from_units(eth, uint256::max());
    EXPECT_EQ(add(top, money::from_units(eth, 1)).err(), money_errc::overflow);
    EXPECT_EQ(add(-top, money::from_units(eth, 1, true)).err(), money_errc::overflow);
    EXPECT_TRUE(add(top, money::from_units(eth, 1, true)));   // opposite signs never overflow
    EXPECT_EQ(multiply(top, 2).err(), money_errc::overflow);
}

TEST(MoneyArithmetic, ComparesByValue)
{
    EXPECT_EQ(*compare(m(aud, "-2"), m(aud, "-1")), std::strong_ordering::less);
    EXPECT_EQ(*compare(m(aud, "-1"), m(aud, "1")), std::strong_ordering::less);
    EXPECT_EQ(*compare(m(aud, "2"), m(aud, "1")), std::strong_ordering::greater);
    EXPECT_EQ(*compare(m(aud, "0"), m(aud, "-0")), std::strong_ordering::equal);
}

TEST(MoneyOf, TypedAmountsKeepTheirAssetInTheType)
{
    using aud_t = money_of<currency::AUD>;
    const auto a = aud_t::from_units(1025);
    const auto b = aud_t::from_units(330, true);
    EXPECT_EQ(add(a, b)->magnitude(), uint256(695));
    EXPECT_LT(b, a);
    EXPECT_EQ(a.to_money(), m(aud, "10.25"));
    EXPECT_EQ(*aud_t::from_money(m(aud, "10.25")), a);
    EXPECT_EQ(aud_t::from_money(m(eth, "1")).err(), money_errc::asset_mismatch);
    EXPECT_EQ(aud_t::asset_of(), aud);
    using gold = money_of<with_scale<currency::XAU, 18>>;
    EXPECT_EQ(gold::asset_of().scale(), 18);
    EXPECT_EQ(gold::asset_of().code(), "XAU");
}

TEST(MoneyText, ToCharsReportsASmallBuffer)
{
    std::array<char, 3> small{};
    EXPECT_EQ(to_chars(small.data(), small.data() + small.size(), m(aud, "1000.5")).ec, std::errc::value_too_large);
    std::ostringstream os;
    os << m(aud, "-1000.5").magnitude().to_string();
    m(aud, "-1000.5").print(os);
    EXPECT_NE(os.str().find("-1000.5 AUD"), std::string::npos);
}

TEST(MoneyText, MatchesDecimalOnEveryFixture)
{
    std::ifstream in(SNICHOLLS_CORE_FIXTURE_DIR "/money/parse.txt");
    ASSERT_TRUE(in) << "tests/fixtures/money/parse.txt";
    std::string line;
    int parsed = 0, formatted = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream is(line);
        std::string kind, scale_s, bar;
        is >> kind >> scale_s;
        const auto a = *make_asset("X", static_cast<unsigned>(std::stoul(scale_s)));
        if (kind == "F") {
            std::string sign, units, minimal, all;
            is >> sign >> units >> bar >> minimal >> all;
            const auto x = money::from_units(a, *uint256::from_decimal(units), sign == "-");
            EXPECT_EQ(text(x), minimal) << line;
            EXPECT_EQ(text(x, money_digits::all), all) << line;
            ++formatted;
            continue;
        }
        std::string input, mode_s, outcome;
        is >> input >> mode_s >> bar >> outcome;
        if (input == "<empty>")
            input.clear();
        std::optional<rounding> mode;
        static const std::pair<const char*, rounding> modes[] = {
            {"half_even", rounding::half_even}, {"half_up", rounding::half_up}, {"half_down", rounding::half_down},
            {"toward_zero", rounding::toward_zero}, {"away_from_zero", rounding::away_from_zero},
            {"floor", rounding::floor}, {"ceiling", rounding::ceiling}};
        for (const auto& [name, r] : modes)
            if (mode_s == name)
                mode = r;
        const auto got = parse_money(a, input, mode);
        if (outcome == "malformed") {
            EXPECT_EQ(got.err(), money_errc::malformed) << line;
        } else if (outcome == "too_many_digits") {
            EXPECT_EQ(got.err(), money_errc::too_many_digits) << line;
        } else {
            std::string units;
            is >> units;
            ASSERT_TRUE(got) << line;
            EXPECT_EQ(*got, money::from_units(a, *uint256::from_decimal(units), outcome == "-")) << line;
        }
        ++parsed;
    }
    EXPECT_GT(parsed, 250);
    EXPECT_GT(formatted, 250);
}

TEST(MoneyText, ParsesZerosPastTheScaleWithoutRounding)
{
    EXPECT_EQ(m(usdt, "0.00100000"), money::from_units(usdt, 1000));   // eight places for a six-place token
    EXPECT_EQ(parse_money(usdt, "0.0000001").err(), money_errc::too_many_digits);
    EXPECT_EQ(*parse_money(usdt, "0.0000005", rounding::half_even), money::zero(usdt));
    EXPECT_EQ(*parse_money(usdt, "0.0000015", rounding::half_even), money::from_units(usdt, 2));
    EXPECT_EQ(*parse_money(usdt, "-0.0000001", rounding::floor), money::from_units(usdt, 1, true));
}

TEST(MoneyDecimal, ConvertsBothWaysExactly)
{
    const auto d = *decimal::parse("1000.5");
    EXPECT_EQ(*from_decimal(eth, d), m(eth, "1000.5"));
    EXPECT_EQ(*from_decimal(aud, *decimal::parse("-72.5")), m(aud, "-72.50"));
    EXPECT_EQ(from_decimal(aud, *decimal::parse("0.005")).err(), money_errc::too_many_digits);
    EXPECT_EQ(*from_decimal(aud, *decimal::parse("0.005"), rounding::half_even), m(aud, "0"));
    EXPECT_EQ(*from_decimal(aud, *decimal::parse("0.015"), rounding::half_even), m(aud, "0.02"));
    EXPECT_EQ(*to_decimal(m(eth, "1000.5")), d);
    EXPECT_EQ(*to_decimal(m(aud, "-6.95")), *decimal::parse("-6.95"));
    EXPECT_EQ(*to_decimal(money::zero(eth)), *decimal::parse("0"));
    EXPECT_FALSE(to_decimal(m(eth, "1000.000000000000000001")));   // 22 significant digits: not a decimal
    static_assert(to_decimal(money::from_units(asset_v<native::ETH>, uint256::pow10(18)))->mantissa() == 1);
}

} // namespace
