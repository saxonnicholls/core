//
//  list_showcase_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/list.hpp>: Data.List classics on std::array, by the compiler: run-length
//  encoding with group, Pythagorean triples in the list monad, a row of Pascal's triangle with zip_with, matrix
//  multiplication with transpose, the eight queens problem on six squares with permutations, and a Caesar cipher.
//  Exits non-zero on a mismatch.
//

#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/iterate.hpp>

#include <array>
#include <cstddef>
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
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/list showcase\n");

    //! [rle]
    // Run-length encoding: map (\g -> (head g, length g)) . group
    constexpr std::array ticks{'u', 'u', 'u', 'd', 'u', 'u', 'd', 'd'};
    constexpr auto runs = fn::group(ticks);
    static_assert(runs.size() == 4 && runs[0].size() == 3 && runs[0][0] == 'u' && runs[3].size() == 2 && runs[3][0] == 'd');
    //! [rle]
    check(runs[1].size() == 1 && runs[1][0] == 'd', "uuu d uu dd: the second run is one down-tick");

    //! [list_monad]
    // Pythagorean triples with legs up to 12, as a list comprehension:
    //   [(a, b, c) | a <- [1..12], b <- [a..12], let c = ..., a^2 + b^2 == c^2]
    // concat_map is the list monad's bind; filter keeps the triples.
    constexpr auto upto12 = snicholls::table<12>([](std::size_t i) { return static_cast<int>(i) + 1; });
    constexpr auto pairs = fn::concat_map([upto12](int a) { return fn::map([a](int b) { return std::pair{a, b}; }, upto12); }, upto12);
    constexpr auto is_square = [](int n) {
        for (int c = 1; c * c <= n; ++c)
            if (c * c == n) return true;
        return false;
    };
    constexpr auto triples = fn::filter([is_square](std::pair<int, int> p) { return p.first < p.second && is_square(p.first * p.first + p.second * p.second); }, pairs);
    static_assert(triples.size() == 4 && triples[0] == std::pair{3, 4} && triples[3] == std::pair{9, 12});
    //! [list_monad]
    check(triples[1] == std::pair{5, 12} && triples[2] == std::pair{6, 8}, "the four triples with legs to 12");

    //! [pascal]
    // The next row of Pascal's triangle is zipWith (+) (0 : row) (row ++ [0]).
    constexpr auto next_row = [](const std::array<long, 8>& row) {
        std::array<long, 8> shifted{};
        for (std::size_t i = 1; i < 8; ++i) shifted[i] = row[i - 1];
        return fn::zip_with(std::plus<>{}, shifted, row);
    };
    constexpr auto row7 = [next_row] {
        std::array<long, 8> r{1};
        for (int i = 0; i < 7; ++i) r = next_row(r);
        return r;
    }();
    static_assert(row7 == std::array<long, 8>{1, 7, 21, 35, 35, 21, 7, 1});
    //! [pascal]
    check(fn::sum(row7) == 128, "row 7 adds to 2^7");

    //! [matrix]
    // Matrix product: each entry is sum (zipWith (*) row column), the columns from transpose.
    constexpr std::array<std::array<int, 2>, 2> a{{{1, 2}, {3, 4}}};
    constexpr std::array<std::array<int, 2>, 2> b{{{5, 6}, {7, 8}}};
    constexpr auto product = [](const auto& x, const auto& y) {
        const auto cols = fn::transpose(y);
        return fn::map([&cols](const std::array<int, 2>& row) {
            return fn::map([&row](const std::array<int, 2>& col) { return fn::sum(fn::zip_with(std::multiplies<>{}, row, col)); }, cols);
        }, x);
    };
    static_assert(product(a, b) == std::array<std::array<int, 2>, 2>{{{19, 22}, {43, 50}}});
    //! [matrix]
    check(product(b, a)[0][0] == 23, "matrices do not commute: (BA)[0][0] = 23");

    //! [queens]
    // N queens on a 6 x 6 board: a permutation puts one queen per row and column; keep those with no shared diagonal.
    constexpr auto safe = [](const std::array<int, 6>& q) {
        for (std::size_t i = 0; i < 6; ++i)
            for (std::size_t j = i + 1; j < 6; ++j) {
                const int dr = static_cast<int>(j - i), dc = q[j] - q[i];
                if (dc == dr || dc == -dr) return false;
            }
        return true;
    };
    constexpr std::size_t solutions = [safe] {
        std::size_t n = 0;
        for (const auto& q : fn::permutations(std::array{0, 1, 2, 3, 4, 5})) n += safe(q) ? 1 : 0;
        return n;
    }();
    static_assert(solutions == 4);
    //! [queens]
    check(solutions == 4, "six queens: four solutions");

    //! [caesar]
    // A Caesar cipher is a map; decoding is the map of the inverse shift; the round trip is the identity.
    constexpr auto shift = [](int k) {
        return [k](char c) { return c < 'a' || c > 'z' ? c : static_cast<char>('a' + (c - 'a' + k + 26) % 26); };
    };
    constexpr std::array msg{'h', 'e', 'l', 'l', 'o', ' ', 'z'};
    static_assert(fn::map(shift(3), msg) == std::array{'k', 'h', 'o', 'o', 'r', ' ', 'c'});
    static_assert(fn::map(shift(-3), fn::map(shift(3), msg)) == msg);
    //! [caesar]
    check(fn::map(shift(13), fn::map(shift(13), msg)) == msg, "rot13 twice is the identity");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
