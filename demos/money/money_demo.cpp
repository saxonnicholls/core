//
//  money_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/money/money.hpp>.
//

// exact amounts: arithmetic, comparison, text
#include <cstdio>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/money/native.hpp>
#include <iostream>
#include <array>
#include <compare>
#include <system_error>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    using namespace snicholls;

    //! [money]
    const asset eth = asset_v<native::ETH>;
    const money a = *parse_money(eth, "1000.5");                      // 1000500000000000000000 wei, exactly
    const money b = *parse_money(eth, "-0.000000000000000001");       // one wei, negative
    const auto sum = add(a, b);                                        // result<money, money_errc>
    const auto mixed = add(a, *parse_money(asset_v<currency::AUD>, "1"));   // a different asset: an error
    using aud = money_of<currency::AUD>;                               // the asset in the type
    const auto typed = add(aud::from_units(1025), aud::from_units(330, true));
    //! [money]
    std::puts("Money");
    std::cout << "  " << to_string(*sum) << " ETH\n";
    check(to_string(*sum) == "1000.499999999999999999", "exact to the wei");
    check(mixed.err() == money_errc::asset_mismatch, "mixing assets is refused");
    check(typed && to_string(typed->to_money()) == "6.95", "typed AUD arithmetic");
    check(*subtract(a, a) == money::zero(eth), "subtract");
    check(*multiply(*parse_money(eth, "1.5"), 2) == *parse_money(eth, "3"), "multiply");
    check(*compare(b, a) == std::strong_ordering::less, "compare");
    check(to_string(a, money_digits::all) == "1000.500000000000000000", "every digit on request");
    check(parse_money(eth, "1.0000000000000000001").err() == money_errc::too_many_digits, "no silent rounding");
    check(*parse_money(eth, "1.0000000000000000005", rounding::half_even) == *parse_money(eth, "1"), "rounding when asked");
    check(money::from_units(eth, uint256::max()).magnitude() == uint256::max(), "the whole uint256 range");
    std::array<char, 4> small{};
    check(to_chars(small.data(), small.data() + small.size(), a).ec == std::errc::value_too_large, "a small buffer");
    a.print(std::cout);
    std::cout << '\n';
    check(aud::from_money(a).err() == money_errc::asset_mismatch && aud::asset_of().code() == "AUD", "typed from run time");
    check(*from_decimal(eth, *decimal::parse("1000.5")) == a && *to_decimal(a) == *decimal::parse("1000.5"), "decimal, both ways");
    check(!to_decimal(*sum), "an amount beyond 18 digits is not a decimal");
    check(-a != a && (-money::zero(eth)).signum() == 0, "negation, and zero is never negative");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
