//
//  combinators_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/combinators.hpp>: functions built from functions, at compile time and at
//  run time. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/combinators.hpp>

#include <cstdio>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

struct order {
    double price;
    int quantity;
};
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/combinators demo\n");

    //! [compose]
    constexpr auto to_bp = [](double r) { return r * 10'000.0; };
    constexpr auto spread = [](double bid, double ask) { return ask - bid; };

    // compose runs right to left, as in maths; pipe left to right, in the order the steps run.
    constexpr auto spread_bp = fn::compose(to_bp, spread);
    constexpr auto spread_bp_too = fn::pipe(spread, to_bp);
    static_assert(spread_bp(1.0850, 1.0852) == spread_bp_too(1.0850, 1.0852));

    // constant, flip, identity, juxt: small pieces that complete a composition.
    static_assert(fn::constant(42)("ignored", 1, 2.0) == 42);
    static_assert(fn::flip(std::minus<>{})(1, 10) == 9);
    static_assert(fn::identity(7) == 7);
    constexpr auto stats = fn::juxt([](int a, int b) { return a < b ? a : b; }, [](int a, int b) { return a < b ? b : a; });
    static_assert(stats(3, 9) == std::tuple{3, 9});
    //! [compose]
    check(spread_bp(1.0850, 1.0852) > 1.99 && spread_bp(1.0850, 1.0852) < 2.01, "compose: a 2 bp spread");
    check(stats(9, 3) == std::tuple{3, 9}, "juxt: min and max at once");

    //! [then]
    // A value through a pipe: x | then(f) | then(g) is g(f(x)).
    const double mid = 1.0851 | fn::then([](double m) { return m * 2; }) | fn::then([](double m) { return m / 2; });
    //! [then]
    check(mid == 1.0851, "then: a value through two steps");

    //! [on]
    // on(f, projection): compare orders by price without writing the comparison twice.
    const auto cheaper = fn::on(std::less<>{}, [](const order& o) { return o.price; });
    // apply: a function of two arguments over a pair.
    constexpr auto notional = fn::apply([](double p, int q) { return p * q; });
    //! [on]
    check(cheaper(order{99.5, 1}, order{100.0, 5}), "on: compares by price");
    check(notional(std::pair{100.0, 3}) == 300.0, "apply: a pair as two arguments");

    //! [fix]
    // fix: recursion for a lambda, without std::function, at compile time.
    constexpr auto factorial = fn::fix([](auto& self, int n) -> long { return n < 2 ? 1 : n * self(n - 1); });
    static_assert(factorial(10) == 3'628'800);
    //! [fix]
    check(factorial(5) == 120, "fix: 5! = 120");

    //! [tap]
    // tap looks without changing; when / unless change only some values.
    std::vector<double> seen;
    const auto record = fn::tap([&](double p) { seen.push_back(p); });
    const auto floor_at_zero = fn::when([](double p) { return p < 0; }, fn::constant(0.0));
    const auto keep_positive = fn::unless([](double p) { return p > 0; }, [](double p) { return -p; });
    const double cleaned = floor_at_zero(record(-3.0));
    //! [tap]
    check(cleaned == 0.0 && seen == std::vector<double>{-3.0}, "tap records, when floors at zero");
    check(floor_at_zero(2.0) == 2.0, "when leaves a positive price alone");
    check(keep_positive(-2.0) == 2.0 && keep_positive(2.0) == 2.0, "unless flips only what is not positive");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
