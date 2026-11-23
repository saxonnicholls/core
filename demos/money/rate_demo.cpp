//
//  rate_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/money/rate.hpp>.
//

// exact conversion with a named rounding and the residual
#include <cstdio>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>
#include <snicholls/core/money/rate.hpp>

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

    //! [rate]
    const asset eth = asset_v<native::ETH>, usd = asset_v<currency::USD>;
    const rate r = *rate::parse(eth, usd, "3000.123456789012345678");   // 22 digits, exact
    const auto c = convert(*parse_money(eth, "1.5"), r, rounding::half_even);
    const auto finer = c->residual.at_scale(usd, 18);                  // to post to a rounding account
    //! [rate]
    std::puts("Rates");
    check(to_string(c->converted) == "4500.19", "1.5 ETH in USD, to the cent");
    check(finer && !finer->is_zero(), "the residual, at 20 places");
    check(r.inverse() && r.inverse()->from() == usd, "inverse");
    check(*rate::make(eth, usd, 3, 2) != r, "equality");
    check(convert(*parse_money(usd, "1"), r, rounding::floor).err() == money_errc::asset_mismatch, "the wrong source asset");
    const auto typed = convert_to<currency::USD>(money_of<native::ETH>::from_units(uint256::pow10(18)), r, rounding::floor);
    check(typed && typed->first.magnitude() == uint256(300012), "typed conversion");
    check(!rate::make(eth, usd, 1, 0), "a zero denominator is refused");
    check(rate::from_decimal(eth, usd, *decimal::parse("3000.25"))->denominator() == uint256(100), "from a decimal");
    check(r.numerator() > r.denominator() && r.to() == usd, "accessors");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
