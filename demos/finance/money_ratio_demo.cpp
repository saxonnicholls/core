//
//  money_ratio_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/finance/money_ratio.hpp>: a fund subscription priced at a NAV per unit, the units exact and
//  then rounded to four places, the units priced back to cents, the NAV per unit from net assets and units
//  outstanding, and the literals. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/finance/money_ratio.hpp>
#include <snicholls/core/money/asset.hpp>

#include <cstdio>

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
    std::printf("snicholls-core finance money_ratio demo\n");
    using namespace snicholls;
    using namespace snicholls::finance;
    using namespace snicholls::finance::literals;
    constexpr asset aud = checked_asset("AUD", 2), units = checked_asset("MFUND1:APIR", 4);

    //! [fund]
    constexpr ratio nav_per_unit = "1.234567"_ratio;                     // a pure rational, no asset
    const money subscription = *parse_money(aud, "1000.00");
    const auto exact = divide(subscription, nav_per_unit);               // 1000000000/1234567 units, exactly
    const auto issued = divide(subscription, nav_per_unit, units, rounding::toward_zero);   // 810.0005, one rounding
    const auto value = multiply(*issued, nav_per_unit, aud, rounding::toward_zero);         // 999.99 (of 999.99988)
    //! [fund]

    std::printf("  exact units %lld/%lld = %.10f; issued %s; worth %s\n", static_cast<long long>(exact->numerator()),
                static_cast<long long>(exact->denominator()), exact->to_double(), to_string(*issued).c_str(),
                to_string(*value).c_str());
    check(*exact == ratio{1'000'000'000, 1'234'567}, "the units are exact");
    check(*issued == *parse_money(units, "810.0005"), "issued to four places, rounded toward zero");
    check(*to_money(*exact, units, rounding::half_even) == *parse_money(units, "810.0006"), "or to nearest, half even");
    check(*value == *parse_money(aud, "999.99"), "the issued units are worth 999.99 at the NAV, truncated");
    check(*multiply(subscription, ratio{1, 2}) == ratio{500, 1}, "an amount times a ratio, exact");
    check(*ratio_of(*parse_money(aud, "1234567.00"), *parse_money(units, "1000000")) == nav_per_unit,
          "net assets over units outstanding is the NAV per unit");
    check(*exact_value(*parse_money(aud, "12.34")) == ratio{617, 50}, "an amount as a ratio of whole units");
    check(!divide(subscription, ratio{}), "nothing divides by zero");
    return failures == 0 ? 0 : 1;
}
