//
//  angular_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/angular.hpp>: dot, squared_norm, norm and cosine_similarity of two embeddings,
//  the four direction metrics between them (cosine_distance, angular, angle, inner_product), angular's reduced form,
//  a zero vector, and a nearest-embedding search under each metric. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/metric/angular.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/metric/concepts.hpp>
#include <snicholls/core/metric/minkowski.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace mt = snicholls::metric;
namespace la = snicholls::linalg;

namespace {

// The index of the nearest of `points` to `q` under `m`
template <typename M, typename Q, typename P>
std::size_t nearest(const M& m, const Q& q, const std::vector<P>& points)
{
    std::size_t best = 0;
    for (std::size_t i = 1; i < points.size(); ++i)
        if (m(q, points[i]) < m(q, points[best]))
            best = i;
    return best;
}

} // namespace

int main()
{
    int failures = 0;
    const auto check = [&](bool ok, const char* what, double value) {
        std::printf("  %-52s %12.9f  %s\n", what, value, ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    };
    const auto near = [](double x, double y) { return std::abs(x - y) <= 1e-12 * (1 + std::abs(y)); };

    //! [angular]
    // Two embeddings of different kinds: a float array and a fixed_vector of doubles
    const std::array<float, 3> x{1, 2, 3};
    const auto y = la::fixed_vector<double, 3>{{4, 5, 6}};

    std::puts("Inner products and lengths of (1, 2, 3) and (4, 5, 6)");
    check(mt::dot(x, y) == 32.0, "dot = 4 + 10 + 18", mt::dot(x, y));
    check(mt::squared_norm(x) == 14.0f, "squared_norm(x) = 14", mt::squared_norm(x));
    check(near(mt::norm(y), 8.774964387392123), "norm(y) = sqrt(77) (python3)", mt::norm(y));
    check(near(mt::cosine_similarity(x, y), 0.9746318461970762), "cosine_similarity (numpy)", mt::cosine_similarity(x, y));

    std::puts("The four direction metrics (python3 values)");
    check(near(mt::cosine_distance{}(x, y), 0.025368153802923787), "cosine_distance = 1 - cos", mt::cosine_distance{}(x, y));
    check(near(mt::angular{}(x, y), 0.225247214424169), "angular = sqrt(2 - 2 cos)", mt::angular{}(x, y));
    check(near(mt::angle{}(x, y), 0.2257261285527342), "angle = acos(cos), radians", mt::angle{}(x, y));
    check(mt::inner_product{}(x, y) == -32.0, "inner_product = -dot", mt::inner_product{}(x, y));

    // Nearest embedding to a query: the direction metrics agree; inner_product also rewards length
    const std::vector<std::array<double, 3>> docs{{1, 0, 0}, {0.9, 0.1, 0}, {0, 10, 10}};
    const std::array<double, 3> query{1, 0.2, 0};
    const std::size_t by_cos = nearest(mt::cosine_distance{}, query, docs);
    check(by_cos == 1 && nearest(mt::angular{}, query, docs) == 1 && nearest(mt::angle{}, query, docs) == 1, "cosine, angular, angle: doc 1", double(by_cos));
    check(nearest(mt::inner_product{}, query, docs) == 2, "inner_product: doc 2, the longest", 2.0);
    //! [angular]

    std::puts("angular's reduced form, 2 - 2 cos");
    const mt::angular ang;
    const double r = ang.reduced(x, y);
    check(near(r, 2 - 2 * mt::cosine_similarity(x, y)) && near(ang.expand(r), ang(x, y)) && near(ang.contract(ang(x, y)), r), "reduced, expand, contract", r);
    check(ang.expand(-1e-17) == 0.0, "expand of a value rounded below zero is 0, not NaN", ang.expand(-1e-17));

    std::puts("Edge cases");
    const std::array<double, 3> zero{0, 0, 0};
    check(mt::cosine_similarity(zero, y) == 0.0 && mt::cosine_distance{}(zero, y) == 1.0, "a zero vector: cosine 0, cosine_distance 1", 0.0);
    check(near(ang(zero, y), std::sqrt(2.0)) && near(mt::angle{}(zero, y), 1.5707963267948966), "a zero vector: angular sqrt(2), angle pi/2", ang(zero, y));
    const std::array<double, 3> x2{2, 4, 6};
    check(mt::cosine_similarity(x, x2) == 1.0 && mt::angle{}(x, x2) == 0.0, "parallel: cosine exactly 1 (clamped), angle 0", 1.0);
    // on unit vectors euclidean equals angular
    const std::array<double, 2> u{1, 0}, v{0, 1};
    check(near(mt::euclidean{}(u, v), ang(u, v)), "unit vectors: euclidean = angular = sqrt(2)", ang(u, v));

    std::printf("%s\n", failures == 0 ? "all results as documented" : "SOME RESULTS DIFFER");
    return failures == 0 ? 0 : 1;
}
