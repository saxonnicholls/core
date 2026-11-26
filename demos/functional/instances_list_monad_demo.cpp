//
//  instances_list_monad_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/instances.hpp>: std::array as Haskell's list monad, where >>= is "for
//  every choice". Pythagorean triples by nondeterministic search, a knight's reachable squares in two moves, and a
//  scenario grid as an applicative cartesian product. The arrays' lengths are in their types, so every search is a
//  fixed-size, allocation-free computation the compiler runs. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/functional/monad.hpp>
#include <snicholls/core/iterate.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <tuple>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;

//! [triples]
// do { a <- [1..n]; b <- [a..n]; c <- [b..n]; guard (a*a + b*b == c*c); return (a, b, c) }
// A guard that may reject is an optional in the array; cat_maybes keeps the survivors.
constexpr auto upto = snicholls::table<17>([](std::size_t i) { return static_cast<int>(i) + 1; });
using triple = std::tuple<int, int, int>;

constexpr auto triples = fn::cat_maybes(fn::mbind(upto, [](int a) {
    return fn::mbind(upto, [a](int b) {
        return fn::fmap([a, b](int c) { return a <= b && b <= c && a * a + b * b == c * c ? std::optional<triple>({a, b, c}) : std::nullopt; },
                        upto);
    });
}));
//! [triples]

//! [knight]
// The eight knight's moves, and every square two moves away from a corner: [moves] >>= move >>= move.
struct square {
    int file, rank;
    friend constexpr bool operator==(const square&, const square&) = default;
};
constexpr std::array<square, 8> jumps{{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}};
constexpr auto moves(square s)
{
    return fn::fmap([s](square j) {
        const square t{s.file + j.file, s.rank + j.rank};
        return t.file >= 0 && t.file < 8 && t.rank >= 0 && t.rank < 8 ? std::optional<square>(t) : std::nullopt;
    }, jumps);
}
// optional squares: a move off the board stays off.
constexpr auto step = [](std::optional<square> s) {
    return s.has_value() ? moves(*s) : std::array<std::optional<square>, 8>{};
};
constexpr auto two_moves = fn::cat_maybes(fn::mbind(moves(square{0, 0}), step));
constexpr std::size_t distinct(const auto& squares)
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < squares.size(); ++i) {
        bool seen = false;
        for (std::size_t j = 0; j < i; ++j) seen = seen || squares[j] == squares[i];
        n += seen ? 0 : 1;
    }
    return n;
}
//! [knight]
} // namespace

int main()
{
    std::printf("snicholls-core functional/instances list-monad demo\n");

    static_assert(triples.size() == 5);   // (3,4,5) (5,12,13) (6,8,10) (8,15,17) (9,12,15)
    static_assert(triples[0] == triple{3, 4, 5} && triples[4] == triple{9, 12, 15});
    check(triples[1] == triple{5, 12, 13}, "the second triple is 5, 12, 13");

    static_assert(distinct(two_moves) == 10);
    check(two_moves.size() == 12 && distinct(two_moves) == 10, "from a1, two knight's moves: 12 paths to 10 squares");

    //! [grid]
    // A scenario grid: every rate shock against every volatility shock, lift_a2 over arrays (Haskell's liftA2 on lists).
    constexpr std::array rates{-0.01, 0.0, 0.01};
    constexpr std::array vols{0.9, 1.1};
    constexpr auto grid = fn::lift_a2([](double r, double v) { return std::pair{r, v}; }, rates, vols);
    static_assert(grid.size() == 6 && grid[0] == std::pair{-0.01, 0.9} && grid[5] == std::pair{0.01, 1.1});
    //! [grid]
    check(grid[3] == std::pair{0.0, 1.1}, "row-major: rates outermost");
    check(fn::join(std::array{std::array{1, 2}, std::array{3, 4}}) == std::array{1, 2, 3, 4}, "join flattens one level");
    check(fn::ap(std::array{+[](int x) { return x + 1; }, +[](int x) { return x * 2; }}, std::array{10, 20}) == std::array{11, 21, 20, 40},
          "ap: every function to every value");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
