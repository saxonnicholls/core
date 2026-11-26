//
//  lazy_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/lazy.hpp>: infinite streams finished by the compiler, and a thunk evaluated once. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/lazy.hpp>

#include <array>
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
    namespace fn = snicholls::fn;
    namespace st = snicholls::fn::stream;
    std::printf("snicholls-core functional/lazy demo\n");

    //! [stream]
    // The Fibonacci numbers, infinite, then the first eight above 10 that are even.
    constexpr auto fib = st::unfold(std::pair{0L, 1L}, [](std::pair<long, long> s) {
        return std::pair{s.first, std::pair{s.second, s.first + s.second}};
    });
    constexpr auto even_big = st::filter([](long x) { return x % 2 == 0; }, st::drop_while([](long x) { return x <= 10; }, fib));
    static_assert(st::take<3>(even_big) == std::array{34L, 144L, 610L});
    // A compounding balance: scanl over an infinite stream of monthly rates, the first year taken.
    constexpr auto balances = st::scanl([](double b, double r) { return b * (1 + r); }, 1000.0, st::repeat(0.01));
    constexpr auto year = st::take<13>(balances);
    static_assert(year[12] > 1126.8 && year[12] < 1126.9);
    //! [stream]
    check(st::nth(fib, 10) == 55, "the tenth Fibonacci number");
    check(st::take_while<100>([](long x) { return x < 100; }, fib).size() == 12, "twelve below 100");
    check(st::take<4>(st::zip_with([](long a, long b) { return a * b; }, fib, st::iterate([](long n) { return n + 1; }, 0L)))[3] == 6,
          "zip_with two infinite streams");
    check(st::head(st::tail(st::cycle(std::array{'B', 'S'}))) == 'S', "cycle, tail and head");

    //! [lazy]
    // A thunk: the expensive calibration runs the first time it is needed, and once.
    int calibrations = 0;
    const fn::lazy vol{[&] { ++calibrations; return 0.2; }};
    const double a = vol.force(), b = vol.force();
    //! [lazy]
    check(a == b && calibrations == 1, "calibrated once, used twice");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
