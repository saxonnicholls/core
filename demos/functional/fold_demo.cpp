//
//  fold_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/fold.hpp>: folds over packs and sequences, a scan, an unfold and a
//  functional power, at compile time. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/fold.hpp>

#include <array>
#include <cstdio>
#include <tuple>
#include <utility>
#include <vector>

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
    std::printf("snicholls-core functional/fold demo\n");

    //! [fold]
    constexpr auto plus = [](auto a, auto b) { return a + b; };
    constexpr auto minus = [](auto a, auto b) { return a - b; };
    // Over a pack: left and right, and the accumulator may change type (int, then double).
    static_assert(fn::foldl(minus, 100, 10, 20) == 70);          // (100 - 10) - 20
    static_assert(fn::foldr(minus, 100, 10, 20) == 90);          // 10 - (20 - 100)
    static_assert(fn::foldl(plus, 0, 1, 2.5) == 3.5);
    // A running P&L: every partial sum.
    static_assert(fn::scan(plus, 0, 120, -40, 15) == std::array{0, 120, 80, 95});
    // Over a range or a tuple.
    static_assert(fn::foldl_of(plus, 0, std::array{1, 2, 3}) == 6);
    static_assert(fn::foldl_of(plus, 0.0, std::tuple{1, 0.5f, 2.0}) == 3.5);
    //! [fold]
    const std::vector<double> fills{10.5, 20.25};
    check(fn::foldl_of(plus, 0.0, fills) == 30.75, "a run-time vector folds the same way");
    check(fn::foldr_of(minus, 0.0, fills) == 10.5 - 20.25, "foldr_of walks from the back");

    //! [unfold]
    // unfold: a schedule from a state. Each step returns {value, next state}: a coupon, then the next period.
    constexpr auto coupons = fn::unfold<4>(std::pair{1, 1000.0}, [](const std::pair<int, double>& s) {
        return std::pair{s.second * 0.05 / 2, std::pair{s.first + 1, s.second}};
    });
    static_assert(coupons == std::array{25.0, 25.0, 25.0, 25.0});
    // iterate_n: a function applied n times, as a function.
    constexpr auto compound_5y = fn::iterate_n([](double b) { return b * 1.05; }, 5);
    static_assert(compound_5y(1000.0) > 1276.28 && compound_5y(1000.0) < 1276.29);
    //! [unfold]
    check(fn::iterate_n([](int x) { return 2 * x; }, 1, 10) == 1024, "iterate_n(f, x, n) applies it now");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
