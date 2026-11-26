//
//  fold_showcase_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/fold.hpp>: Horner's rule as a left fold, a list of functions composed by a
//  right fold, Fibonacci and Collatz by unfold, a running maximum drawdown by scan, a fold over a heterogeneous tuple,
//  and repeated squaring by iterate_n, all evaluated by the compiler. Exits non-zero on a mismatch.
//

#include <snicholls/core/functional/fold.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
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
    std::printf("snicholls-core functional/fold showcase\n");

    //! [horner]
    // Horner's rule: 2x^3 - 6x^2 + 2x - 1 at x = 3 is foldl (\acc c -> acc * x + c) 0 [2, -6, 2, -1].
    constexpr auto horner = [](double x, const auto& coefficients) {
        return fn::foldl_of([x](double acc, double c) { return acc * x + c; }, 0.0, coefficients);
    };
    static_assert(horner(3.0, std::array{2.0, -6.0, 2.0, -1.0}) == 5.0);
    //! [horner]
    check(horner(2.0, std::array{1.0, 0.0, 0.0, 0.0}) == 8.0, "x^3 at 2");

    //! [compose_by_fold]
    // A pipeline is a fold over functions: foldr (.) id [f, g, h], here over a tuple of different lambdas.
    constexpr auto steps = std::tuple{[](int x) { return x + 1; }, [](int x) { return x * 10; }, [](int x) { return x - 3; }};
    constexpr auto composed = fn::foldr_of([](auto f, auto acc) { return [f, acc](int x) { return f(acc(x)); }; },
                                           [](int x) { return x; }, steps);
    static_assert(composed(5) == (5 - 3) * 10 + 1);
    //! [compose_by_fold]
    check(composed(0) == -29, "f . g . h applies h first");

    //! [unfold]
    // unfoldr: Fibonacci from its pair of state, and the Collatz orbit of 6 (padded with 1 once it lands).
    constexpr auto fibonacci = fn::unfold<12>(std::pair{0, 1}, [](std::pair<int, int> s) {
        return std::pair{s.first, std::pair{s.second, s.first + s.second}};
    });
    static_assert(fibonacci == std::array{0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89});
    constexpr auto collatz = fn::unfold<10>(6, [](int n) { return std::pair{n, n == 1 ? 1 : n % 2 == 0 ? n / 2 : 3 * n + 1}; });
    static_assert(collatz == std::array{6, 3, 10, 5, 16, 8, 4, 2, 1, 1});
    //! [unfold]
    check(fibonacci[11] == 89, "the twelfth Fibonacci number");

    //! [scan]
    // A scan of a P&L stream gives the equity curve.
    constexpr auto equity = fn::scan([](int a, int x) { return a + x; }, 100, 20, -50, 10, 40, -30);
    // The maximum drawdown is one more fold, its state the running peak and the worst fall from it.
    constexpr auto max_drawdown = fn::foldl_of([](std::pair<int, int> s, int e) {
        const int top = s.first < e ? e : s.first;
        return std::pair{top, s.second < top - e ? top - e : s.second};
    }, std::pair{equity[0], 0}, equity).second;
    static_assert(equity == std::array{100, 120, 70, 80, 120, 90} && max_drawdown == 50);
    //! [scan]
    check(equity.back() == 90, "the equity curve ends at 90");

    //! [heterogeneous]
    // A fold over a tuple: the accumulator sees an int, a double and a float in turn.
    constexpr auto total = fn::foldl_of([](auto a, auto x) { return a + x; }, 0.0, std::tuple{1, 2.5, 0.25f});
    static_assert(total == 3.75);
    //! [heterogeneous]
    check(fn::foldl(std::plus<>{}, 0, 1, 2, 3, 4) == 10, "a fold over a parameter pack");

    //! [iterate]
    // Repeated squaring by iterate_n: 2^(2^5) = 2^32.
    constexpr auto square = [](std::uint64_t x) { return x * x; };
    static_assert(fn::iterate_n(square, std::uint64_t{2}, 5) == (std::uint64_t{1} << 32));
    //! [iterate]
    check(fn::iterate_n(square, 3)(std::uint64_t{3}) == 6561, "3 squared three times");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
