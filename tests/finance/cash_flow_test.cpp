//
//  cash_flow_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/finance/cash_flow.hpp>: the one dated flow the ladder and the return measures share.
//

#include <snicholls/core/finance/cash_flow.hpp>
#include <snicholls/core/finance/cash_ladder.hpp>
#include <snicholls/core/finance/returns.hpp>
#include <snicholls/core/money/iso4217.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <span>
#include <type_traits>

namespace {

using namespace snicholls;
using finance::cash_flow;
using namespace std::chrono_literals;

constexpr asset aud = asset_v<currency::AUD>;

static_assert(std::is_aggregate_v<cash_flow>);
static_assert(cash_flow{std::chrono::sys_days{2026y / 3 / 1}, money::from_units(aud, 100)} ==
              cash_flow{std::chrono::sys_days{2026y / 3 / 1}, money::from_units(aud, 100)});
static_assert(cash_flow{std::chrono::sys_days{2026y / 3 / 1}, money::from_units(aud, 100)} !=
              cash_flow{std::chrono::sys_days{2026y / 3 / 2}, money::from_units(aud, 100)});

TEST(FinanceCashFlow, OneFlowFeedsBothTheLadderAndTheReturnMeasures)
{
    // positive in, negative out, in both: a contribution of 500.00 then a withdrawal of 200.00
    const cash_flow flows[] = {{std::chrono::sys_days{2026y / 2 / 1}, money::from_units(aud, 50'000)},
                               {std::chrono::sys_days{2026y / 2 / 15}, money::from_units(aud, -20'000)}};
    const std::span<const cash_flow> view{flows};
    EXPECT_EQ(view.size(), 2u);
    EXPECT_FALSE(view[0].amount.is_negative());
    EXPECT_TRUE(view[1].amount.is_negative());
}

} // namespace
