//
//  combinators_showcase_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/combinators.hpp>: point-free pipelines, OCaml's |>, anonymous recursion
//  through the fixed-point combinator (factorial, Ackermann, Fibonacci), Newton's method with `until`, and fan-out
//  with `juxt`, each computed by the compiler and checked again at run time. Exits non-zero on a mismatch.
//

#include <snicholls/core/functional/combinators.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <tuple>

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
    std::printf("snicholls-core functional/combinators showcase\n");

    //! [point_free]
    // Point-free style: a pricing step built from named pieces, with no argument ever named.
    constexpr auto to_usd = [](double eur) { return eur * 1.08; };
    constexpr auto add_fee = [](double x) { return x + 2.5; };
    constexpr auto round_cents = [](double x) { return static_cast<double>(static_cast<long>(x * 100 + 0.5)) / 100; };
    constexpr auto quote = fn::pipe(to_usd, add_fee, round_cents);   // read left to right
    constexpr auto same = fn::compose(round_cents, add_fee, to_usd);   // read right to left
    static_assert(quote(100.0) == 110.5 && same(100.0) == quote(100.0));
    // OCaml's  100. |> to_usd |> add_fee |> round_cents
    constexpr double piped = 100.0 | fn::then(to_usd) | fn::then(add_fee) | fn::then(round_cents);
    static_assert(piped == 110.5);
    //! [point_free]
    check(quote(19.99) == 24.09, "a 19.99 EUR item quotes at 24.09 USD");

    //! [fix]
    // The Y combinator: recursion for a lambda that has no name. `self` is the function itself.
    constexpr auto factorial = fn::fix([](auto self, std::uint64_t n) -> std::uint64_t { return n < 2 ? 1 : n * self(n - 1); });
    constexpr auto ackermann = fn::fix([](auto self, int m, int n) -> int {
        if (m == 0) return n + 1;
        if (n == 0) return self(m - 1, 1);
        return self(m - 1, self(m, n - 1));
    });
    static_assert(factorial(20) == 2'432'902'008'176'640'000ULL);
    static_assert(ackermann(2, 3) == 9 && ackermann(3, 3) == 61);
    //! [fix]
    check(factorial(10) == 3'628'800, "10! at run time");
    check(ackermann(3, 4) == 125, "A(3, 4) = 125 at run time");

    //! [until]
    // Newton's method for the square root: improve the guess until it is good enough (Haskell's `until`).
    constexpr auto sqrt_of = [](double a) {
        const auto good = [a](double x) { return (x * x - a < 0 ? a - x * x : x * x - a) < 1e-12 * a; };
        const auto improve = [a](double x) { return (x + a / x) / 2; };
        return fn::until(good, improve)(a);
    };
    static_assert(sqrt_of(2.0) > 1.414213562 && sqrt_of(2.0) < 1.414213563);
    //! [until]
    check(sqrt_of(1e6) > 999.999999 && sqrt_of(1e6) < 1000.000001, "the square root of a million");

    //! [juxt]
    // Fan-out: one value through several functions at once, and a comparison on a projection.
    constexpr auto stats = fn::juxt([](const std::array<int, 5>& v) { return v[0]; },
                                    [](const std::array<int, 5>& v) { return v[4]; },
                                    [](const std::array<int, 5>& v) { return v[0] + v[1] + v[2] + v[3] + v[4]; });
    static_assert(stats(std::array{1, 2, 3, 4, 5}) == std::tuple{1, 5, 15});
    constexpr auto closer_to_zero = fn::on(std::less<>{}, [](int x) { return x < 0 ? -x : x; });
    static_assert(closer_to_zero(-2, 3) && !closer_to_zero(-4, 3));
    //! [juxt]
    check(std::get<2>(stats(std::array{10, 20, 30, 40, 50})) == 150, "juxt: first, last and sum in one call");

    //! [conditionals]
    // Guards as functions: clamp a quantity, double the even ones, leave the rest alone.
    constexpr auto cap = fn::when([](int q) { return q > 1000; }, fn::constant(1000));
    constexpr auto shape = fn::pipe(cap, fn::when([](int q) { return q % 2 == 0; }, [](int q) { return q * 2; }));
    static_assert(shape(5000) == 2000 && shape(7) == 7 && shape(8) == 16);
    //! [conditionals]
    check(fn::unless([](int q) { return q > 0; }, fn::constant(0))(-5) == 0, "unless: negative quantities become zero");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
