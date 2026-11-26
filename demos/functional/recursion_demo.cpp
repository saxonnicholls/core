//
//  recursion_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/recursion.hpp>: a cash-flow schedule built (ana), priced (cata) and fused (hylo); a deep recursion on a trampoline. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/recursion.hpp>

#include <array>
#include <cstdio>
#include <optional>
#include <utility>

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
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/recursion demo\n");

    //! [schemes]
    // ana: a bond's cash flows from a state (periods left): coupons, then the redemption.
    constexpr auto flows = [](int left) {
        return left == 0 ? std::nullopt : std::optional{std::pair{left == 1 ? 1025.0 : 25.0, left - 1}};
    };
    constexpr auto schedule = fn::ana<16>(flows, 4);
    static_assert(schedule.size() == 4 && schedule[3] == 1025.0);
    // cata: priced from the last flow back, each discounted one period.
    constexpr double price = fn::cata([](double cf, double later) { return (cf + later) / 1.025; }, 0.0, schedule);
    static_assert(price > 999.99 && price < 1000.01);
    // hylo: the undiscounted total, unfolded and folded in one pass, with no schedule in between.
    static_assert(fn::hylo([](double acc, double cf) { return acc + cf; }, 0.0, flows, 4) == 1100.0);
    //! [schemes]
    check(price > 999.99 && price < 1000.01, "a par bond prices at par");

    //! [trampoline]
    // Steps to reach 1 under Collatz's rule, at any depth, in constant stack.
    constexpr auto collatz = [](std::pair<long, int> s) -> fn::bounce<std::pair<long, int>, int> {
        if (s.first == 1) return fn::done(s.second);
        return fn::more(std::pair{s.first % 2 != 0 ? 3 * s.first + 1 : s.first / 2, s.second + 1});
    };
    static_assert(fn::trampoline(collatz, std::pair{27L, 0}) == 111);
    //! [trampoline]
    check(fn::trampoline(collatz, std::pair{837799L, 0}) == 524, "837799 takes 524 steps");
    check(fn::para([](int x, auto rest, int acc) { return acc + x * static_cast<int>(rest.size()); }, 0, std::array{1, 2, 3}) == 4,
          "para sees what follows each element");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
