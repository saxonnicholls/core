//
//  rate_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/money/rate.hpp>.
//

// exact conversion with a named rounding and the residual
#include <array>
#include <cstddef>
#include <cstdint>
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

    //! [helpers]
    // a dividend of 1/3 cent per unit to three holders, each line rounded half-even by hand
    const uint256 d = 3;
    std::array<residual, 3> left{};
    uint256 paid = 0;
    const std::array<std::uint64_t, 3> holdings{100, 200, 400};
    for (std::size_t i = 0; i < holdings.size(); ++i) {
        const auto [q, rem] = uint256{holdings[i]}.divmod(d);   // one hardware division: both fit in 64 bits
        const residual fraction{false, rem, d};
        const bool up = rounds_away(q, fraction, rounding::half_even);
        paid += up ? q + 1 : q;
        left[i] = up ? residual{true, d - rem, d} : fraction;   // exact minus paid
    }
    const auto total = sum_residuals(left, d);   // 1/3 - 1/3 + 1/3: the issuer keeps one third of a cent
    //! [helpers]
    std::puts("Rounding by hand");
    check(paid == uint256(233), "33 + 67 + 133 cents paid");
    check(total && *total == (residual{false, 1, 3}), "one third of a cent left over, exactly");
    check(sum_residuals(left, 7).err() == money_errc::malformed, "another denominator is refused");
    std::puts("Pricing a rate");
    const rate nav_exact = *rate::parse(eth, usd, "1.23456");
    const auto nav = convert_rate(nav_exact, 4, rounding::half_even);   // NAV per unit to four places
    check(nav && nav->rounded == *rate::make(eth, usd, 12346, 10000), "1.23456 priced to 1.2346");
    check(nav && nav->residual == (residual{true, 40000, 100000}), "and 0.4 of 10^-4 over the exact rate");
    check(convert_rate(nav_exact, 78, rounding::floor).err() == money_errc::overflow, "at most 77 places");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
