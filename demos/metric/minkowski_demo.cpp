//
//  minkowski_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/minkowski.hpp>: euclidean, manhattan, chebyshev, minkowski<P>, squared_euclidean
//  and weighted_minkowski between points of different kinds, each metric's term / combine / expand / contract /
//  reduced, a search radius compared in reduced form, and a weighted metric built with weighted<P>(w). Exits non-zero
//  if a result is not what it documents.
//

#include <snicholls/core/metric/minkowski.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

namespace mt = snicholls::metric;
namespace la = snicholls::linalg;

int main()
{
    int failures = 0;
    const auto check = [&](bool ok, const char* what, double value) {
        std::printf("  %-52s %12.9f  %s\n", what, value, ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    };
    const auto near = [](double x, double y) { return std::abs(x - y) <= 1e-12 * (1 + std::abs(y)); };

    //! [minkowski]
    // The same metric between any two point kinds: an array of doubles, a vector of floats, a fixed_vector, a span
    const std::array<double, 3> a{1, -2, 3};
    const std::vector<float> b{4, 2, -1};
    const auto c = la::fixed_vector<double, 3>{{1, -2, 3}};
    const std::span<const float> bs(b);

    // Differences (-3, -4, 4): values by hand, or python3 where marked
    std::puts("Distances between (1, -2, 3) and (4, 2, -1)");
    check(near(mt::euclidean{}(a, b), 6.4031242374328485), "euclidean = sqrt(41)", mt::euclidean{}(a, b));
    check(mt::manhattan{}(a, bs) == 11.0, "manhattan = 3 + 4 + 4", mt::manhattan{}(a, bs));
    check(mt::chebyshev{}(c, b) == 4.0, "chebyshev = max(3, 4, 4)", mt::chebyshev{}(c, b));
    check(mt::squared_euclidean{}(c, bs) == 41.0, "squared_euclidean = 9 + 16 + 16", mt::squared_euclidean{}(c, bs));
    check(near(mt::minkowski<3>{}(a, b), 5.371685354944832), "minkowski<3> = 155^(1/3) (python3)", mt::minkowski<3>{}(a, b));
    check(near(mt::minkowski<64>{}(a, b), 4.04355714452557), "minkowski<64>, near chebyshev (numpy)", mt::minkowski<64>{}(a, b));

    // A weighted metric holds its weights, one per axis: sqrt(1*9 + 2*16 + 0.5*16) = 7
    const auto w = mt::weighted<2>(std::array<double, 3>{1, 2, 0.5});
    check(w(a, b) == 7.0, "weighted<2>, weights (1, 2, 0.5) = 7", w(a, b));

    // A search compares in the reduced form, which needs no root: is b within radius 6.5 of a?
    const mt::euclidean e;
    const double r2 = mt::contract(e, 6.5);                        // 42.25
    const bool inside = mt::accumulate_within(e, a, b, r2) <= r2;  // 9, 25, 41: never above 42.25
    check(inside && e.reduced(a, b) == 41.0, "b within 6.5 of a, by reduced 41 <= 42.25", e.reduced(a, b));
    //! [minkowski]

    std::puts("Each metric's parts: term, combine, expand, contract, reduced");
    check(e.term(0, -3.0) == 9.0 && e.combine(9.0, 16.0) == 25.0 && e.expand(25.0) == 5.0 && e.contract(5.0) == 25.0, "euclidean: 9, 25, 5, 25", 5.0);
    const mt::manhattan m1;
    check(m1.term(1, -4.0) == 4.0 && m1.combine(3.0, 4.0) == 7.0 && m1.expand(7.0) == 7.0 && m1.contract(7.0) == 7.0 && m1.reduced(a, b) == 11.0,
          "manhattan: |d|, +, identity, identity", 11.0);
    const mt::chebyshev ch;
    check(ch.term(2, -4.0) == 4.0 && ch.combine(4.0, 3.0) == 4.0 && ch.expand(4.0) == 4.0 && ch.contract(4.0) == 4.0 && ch.reduced(a, b) == 4.0,
          "chebyshev: |d|, max, identity, identity", 4.0);
    const mt::squared_euclidean sq;
    check(sq.term(0, -3.0) == 9.0 && sq.combine(9.0, 16.0) == 25.0 && sq.expand(41.0) == 41.0 && sq.contract(41.0) == 41.0 && sq.reduced(a, b) == 41.0,
          "squared_euclidean: d^2, +, identity, identity", 41.0);
    const mt::minkowski<3> m3;
    check(m3.term(0, -2.0) == 8.0 && m3.combine(8.0, 19.0) == 27.0 && near(m3.expand(27.0), 3.0) && m3.contract(3.0) == 27.0 && m3.reduced(a, b) == 155.0,
          "minkowski<3>: |d|^3, +, cube root, cube", 155.0);
    check(w.term(1, -4.0) == 32.0 && w.combine(9.0, 32.0) == 41.0 && w.expand(49.0) == 7.0 && w.contract(7.0) == 49.0 && w.reduced(a, b) == 49.0,
          "weighted<2>: w d^2, +, sqrt, square", 49.0);
    const mt::weighted_minkowski<1, la::fixed_vector<double, 3>> w1{la::fixed_vector<double, 3>{{1, 1, 1}}};
    check(w1(a, b) == m1(a, b), "unit weights give the unweighted metric", w1(a, b));

    std::puts("Capabilities");
    check(mt::true_metric<mt::euclidean> && mt::true_metric<mt::chebyshev> && !mt::true_metric<mt::squared_euclidean>,
          "squared_euclidean alone is not a true metric", 0.0);
    // squared: 0, 1, 2 on a line gives d(0, 2) = 4 > d(0, 1) + d(1, 2) = 2
    const std::array<double, 1> p0{0}, p1{1}, p2{2};
    check(sq(p0, p2) > sq(p0, p1) + sq(p1, p2), "its triangle inequality fails: 4 > 1 + 1", sq(p0, p2));

    std::printf("%s\n", failures == 0 ? "all results as documented" : "SOME RESULTS DIFFER");
    return failures == 0 ? 0 : 1;
}
