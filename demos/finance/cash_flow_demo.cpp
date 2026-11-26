//
//  cash_flow_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  finance::cash_flow: one dated, signed, exact flow, shared by the liquidity ladder and the return measures.
//

#include <snicholls/core/finance/cash_flow.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <chrono>
#include <cstdio>

int main()
{
    using namespace snicholls;
    using namespace std::chrono_literals;
    constexpr asset aud = asset_v<currency::AUD>;
    std::puts("finance::cash_flow");
    //! [cash_flow]
    // a settlement credit of 1,250.00 due on 4 March, and a payout of 300.00 on the 5th: positive in, negative out
    const finance::cash_flow due[] = {{std::chrono::sys_days{2026y / 3 / 4}, money::from_units(aud, 125'000)},
                                      {std::chrono::sys_days{2026y / 3 / 5}, money::from_units(aud, -30'000)}};
    // the same span goes to finance::make_cash_ladder (liquidity) or finance::modified_dietz (performance)
    //! [cash_flow]
    const bool ok = !due[0].amount.is_negative() && due[1].amount.is_negative() && due[0].on < due[1].on;
    std::printf("  [%s] in positive, out negative, dated\n%s\n", ok ? "ok" : "FAIL", ok ? "all checks passed" : "FAILED");
    return ok ? 0 : 1;
}
