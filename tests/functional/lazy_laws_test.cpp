//
//  lazy_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Laws for <snicholls/core/functional/lazy.hpp>: infinite streams obey the list laws on every prefix (map identity
//  and fusion, filter fusion, nth = head . drop, cycle's period, scanl against a fold), and a thunk is forced once.
//

#include <snicholls/core/functional/lazy.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

namespace fn = snicholls::fn;
namespace st = snicholls::fn::stream;

constexpr auto nat = st::iterate([](long n) { return n + 1; }, 0L);
constexpr auto inc = [](long x) { return x + 1; };
constexpr auto sq = [](long x) { return x * x; };
constexpr auto odd = [](long x) { return x % 2 != 0; };

static_assert(st::take<5>(nat) == std::array{0L, 1L, 2L, 3L, 4L});
static_assert(st::take<6>(st::map(fn::identity, nat)) == st::take<6>(nat));                       // map id = id
static_assert(st::take<6>(st::map(inc, st::map(sq, nat))) == st::take<6>(st::map(fn::compose(inc, sq), nat)));   // fusion
static_assert(st::take<4>(st::filter(odd, st::filter([](long x) { return x > 4; }, nat))) ==
              st::take<4>(st::filter([](long x) { return x > 4 && x % 2 != 0; }, nat)));       // filter fusion
static_assert(st::nth(nat, 7) == st::head(st::drop(7, nat)) && st::nth(nat, 0) == st::head(nat));
static_assert(st::head(st::tail(st::tail(nat))) == 2);
static_assert(st::take<7>(st::cycle(std::array{1, 2, 3})) == std::array{1, 2, 3, 1, 2, 3, 1});      // period 3
static_assert(st::take<3>(st::repeat('x')) == std::array{'x', 'x', 'x'});
static_assert(st::take<0>(nat).empty());
static_assert(st::take<5>(st::scanl([](long a, long x) { return a + x; }, 0L, nat)) == std::array{0L, 0L, 1L, 3L, 6L});   // triangular
static_assert(st::nth(st::scanl([](long a, long x) { return a + x; }, 0L, nat), 101) == 100L * 101 / 2);
static_assert(st::take<3>(st::drop_while([](long x) { return x < 10; }, nat)) == std::array{10L, 11L, 12L});
static_assert(st::take_while<10>([](long x) { return x < 4; }, nat).size() == 4);
static_assert(st::take_while<3>([](long) { return true; }, nat).size() == 3);                      // the bound stops it
static_assert(st::take_while<10>([](long) { return false; }, nat).empty());
static_assert(st::take<3>(st::zip(nat, st::repeat('a')))[2] == std::pair{2L, 'a'});
static_assert(st::take<4>(st::zip_with(std::multiplies<>{}, nat, nat)) == st::take<4>(st::map(sq, nat)));

// The primes, by trial division over the infinite naturals.
constexpr bool is_prime(long n)
{
    if (n < 2) return false;
    for (long d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}
static_assert(st::take<10>(st::filter(is_prime, nat)) == std::array{2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L, 29L});
static_assert(st::nth(st::filter(is_prime, nat), 99) == 541);   // the hundredth prime

TEST(FunctionalLazyLaws, AThunkIsForcedAtMostOnceAndThenCached)
{
    int calls = 0;
    const fn::lazy answer{[&] { ++calls; return 42; }};
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(answer.force(), 42);
    EXPECT_EQ(answer.force(), 42);
    EXPECT_EQ(calls, 1);
}

TEST(FunctionalLazyLaws, AStreamComputesOnlyWhatIsTaken)
{
    int steps = 0;
    const auto counted = st::iterate([&](int n) { ++steps; return n + 1; }, 0);
    const auto first = st::take<3>(counted);
    EXPECT_EQ(first[2], 2);
    EXPECT_LE(steps, 3);
}

TEST(FunctionalLazyLaws, AThrowingThunkThrowsWhenForced)
{
    const fn::lazy bad{[]() -> int { throw std::runtime_error("no"); }};
    EXPECT_THROW(static_cast<void>(bad.force()), std::runtime_error);
}

} // namespace
