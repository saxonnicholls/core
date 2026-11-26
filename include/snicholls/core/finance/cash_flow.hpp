//
//  cash_flow.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::finance::cash_flow`: one dated cash flow, exact, positive in and negative out.
/// \ingroup core
///
/// **Overview.** The one type for money that moves on a day, shared by everything in `finance/` that reads dated flows:
/// the liquidity ladder (`make_cash_ladder`, `cash_ladder.hpp`), which dates flows by their value date, and the return
/// measures (`modified_dietz`, `money_weighted_return`, `returns.hpp`), which date external flows by the day they
/// happened. One convention everywhere: a flow **in** (received, contributed, a settlement credit) is positive, a flow
/// **out** is negative. The amount is exact `money`; `xirr_flow` in `returns.hpp` is the `double` form for the root
/// finder.
///
/// \snippet finance/cash_flow_demo.cpp cash_flow
///
/// **Relationships.** `money/money.hpp`; `finance/cash_ladder.hpp`, `finance/returns.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One aggregate.
///
/// **Guarantees.** Trivially copyable when `money` is; `constexpr`; compares member by member.
///
/// **Errors.** None here: what a function does with flows of mixed assets is that function's (the ladder refuses them).
///
/// **Performance.** Not applicable: a value type.
///
/// **Tests and demos.** `tests/finance/cash_flow_test.cpp`, `demos/finance/cash_flow_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `money/money.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0 (it replaces the ladder's and the return measures' separate types).

#ifndef snicholls_core_finance_cash_flow_hpp
#define snicholls_core_finance_cash_flow_hpp

#include <snicholls/core/money/money.hpp>

#include <chrono>

namespace snicholls::finance {

    /// \brief One dated cash flow, exact: positive in, negative out.
    /// \ingroup core
    struct cash_flow {
        std::chrono::sys_days on{};  ///< the day: the value date for a ladder, the day it happened for a return
        money amount{};              ///< the signed amount: positive in, negative out

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const cash_flow&, const cash_flow&) noexcept = default;
    };

} // namespace snicholls::finance

#endif // snicholls_core_finance_cash_flow_hpp
