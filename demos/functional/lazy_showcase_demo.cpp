//
//  lazy_showcase_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/lazy.hpp>: Haskell's infinite lists at compile time: the primes, the
//  Fibonacci numbers, a Collatz orbit taken while it lasts, pi from Leibniz's series by scanl, the triangular numbers
//  from two infinite streams zipped, and a schedule that cycles. Exits non-zero on a mismatch.
//

#include <snicholls/core/functional/lazy.hpp>

#include <array>
#include <cstdio>
#include <functional>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

constexpr bool is_prime(long n)
{
    if (n < 2) return false;
    for (long d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}
} // namespace

int main()
{
    namespace st = snicholls::fn::stream;
    std::printf("snicholls-core functional/lazy showcase\n");

    //! [primes]
    // primes = filter isPrime [2..]; take 10 primes; primes !! 999
    constexpr auto naturals = st::iterate([](long n) { return n + 1; }, 2L);
    constexpr auto primes = st::filter(is_prime, naturals);
    static_assert(st::take<10>(primes) == std::array{2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L, 29L});
    //! [primes]
    check(st::nth(primes, 999) == 7919, "the thousandth prime is 7919");

    //! [fibonacci]
    // fibs = unfoldr (\(a, b) -> Just (a, (b, a + b))) (0, 1); the first above a million
    constexpr auto fibs = st::unfold(std::pair{0L, 1L}, [](std::pair<long, long> s) { return std::pair{s.first, std::pair{s.second, s.first + s.second}}; });
    static_assert(st::head(st::drop_while([](long f) { return f <= 1'000'000; }, fibs)) == 1'346'269);
    //! [fibonacci]
    check(st::nth(fibs, 50) == 12'586'269'025L, "F(50)");

    //! [collatz]
    // takeWhile (/= 1) (iterate collatz 27): the orbit, as long as it lasts (at most 200 values kept).
    constexpr auto orbit = st::take_while<200>([](long n) { return n != 1; },
                                               st::iterate([](long n) { return n % 2 == 0 ? n / 2 : 3 * n + 1; }, 27L));
    static_assert(orbit.size() == 111);
    //! [collatz]
    check(orbit[0] == 27 && orbit[110] == 2, "27's orbit takes 111 steps to reach 1");

    //! [pi]
    // pi = 4 * sum [(-1)^k / (2k + 1)]: the partial sums by scanl over an infinite stream of terms.
    constexpr auto terms = st::map([](long k) { return (k % 2 == 0 ? 4.0 : -4.0) / static_cast<double>(2 * k + 1); },
                                   st::iterate([](long k) { return k + 1; }, 0L));
    constexpr auto partial_sums = st::scanl(std::plus<>{}, 0.0, terms);
    constexpr double pi_1000 = st::nth(partial_sums, 1000);
    static_assert(pi_1000 > 3.1405 && pi_1000 < 3.1426);   // slow: the error is about 1 / n
    // The mean of two neighbouring partial sums cancels the alternating error: six places from the same terms.
    constexpr double pi_mean = (st::nth(partial_sums, 1000) + st::nth(partial_sums, 1001)) / 2;
    static_assert(pi_mean > 3.141592 && pi_mean < 3.141593);
    //! [pi]
    check(pi_1000 > 3.14 && pi_1000 < 3.143, "a thousand terms of Leibniz's series give pi to two places");

    //! [zip]
    // The triangular numbers n(n+1)/2 as zipWith (*) of two infinite streams, halved.
    constexpr auto from0 = st::iterate([](long n) { return n + 1; }, 0L);
    constexpr auto triangular = st::map([](long x) { return x / 2; }, st::zip_with(std::multiplies<>{}, from0, st::drop(1, from0)));
    static_assert(st::take<6>(triangular) == std::array{0L, 1L, 3L, 6L, 10L, 15L});
    //! [zip]
    check(st::nth(triangular, 100) == 5050, "Gauss: 1 + ... + 100");

    //! [cycle]
    // A rota that repeats forever: cycle ["Mon", ...]; who is on the 10th day?
    constexpr std::array<char, 3> rota{'A', 'B', 'C'};
    static_assert(st::nth(st::cycle(rota), 9) == 'A');
    static_assert(st::take<5>(st::zip(st::cycle(rota), st::repeat(1)))[4] == std::pair{'B', 1});
    //! [cycle]
    check(st::head(st::tail(st::cycle(rota))) == 'B', "the second day is B's");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
