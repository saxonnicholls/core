//
//  concepts_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/metric/concepts.hpp>: a point type of the caller's own plugged in by a point_traits
//  specialisation, two metrics of the caller's own (one with only operator(), one with term / combine / expand /
//  contract), and what each capability unlocks: a brute-force search for any distance, early-exit scans for a
//  decomposable one, the reduced form of a reducible one, triangle pruning for a true metric. Exits non-zero if a
//  result is not what it documents.
//

#include <snicholls/core/metric/concepts.hpp>

#include <snicholls/core/linalg/fixed.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <type_traits>
#include <vector>

namespace mt = snicholls::metric;
namespace la = snicholls::linalg;

//! [concepts]
// A point type of the caller's own: a colour. One specialisation makes it a point for every metric and structure.
struct rgb {
    float r, g, b;
};

template <>
struct snicholls::metric::point_traits<rgb> {
    using value_type = float;
    static constexpr std::size_t extent = 3;
    static constexpr std::size_t dimension(const rgb&) noexcept { return 3; }
    static constexpr float at(const rgb& c, std::size_t i) noexcept { return i == 0 ? c.r : i == 1 ? c.g : c.b; }
};

namespace {

// A metric of the caller's own with only operator(): the taxicab distance rounded to whole units
struct rounded_taxicab {
    template <mt::point A, mt::point B>
    constexpr double operator()(const A& a, const B& b) const noexcept
    {
        double s = 0;
        for (std::size_t i = 0; i < mt::dimension(a); ++i) {
            const double d = static_cast<double>(mt::coordinate(a, i)) - static_cast<double>(mt::coordinate(b, i));
            s += d < 0 ? -d : d;
        }
        return std::floor(s + 0.5);
    }
};

// A metric of the caller's own built coordinate by coordinate: Euclidean with green counted twice (the eye is most
// sensitive to it). term / combine make it decomposable; reduced / expand / contract make it reducible.
struct weighted_colour {
    static constexpr bool is_metric = true;
    template <typename T>
    constexpr T term(std::size_t axis, T d) const noexcept { return (axis == 1 ? T(2) : T(1)) * d * d; }
    template <typename T>
    constexpr T combine(T acc, T t) const noexcept { return acc + t; }
    template <typename T>
    T expand(T r) const noexcept { return std::sqrt(r); }
    template <typename T>
    constexpr T contract(T d) const noexcept { return d * d; }
    template <mt::point A, mt::point B>
    constexpr mt::distance_t<A, B> reduced(const A& a, const B& b) const noexcept { return mt::accumulate(*this, a, b); }
    template <mt::point A, mt::point B>
    mt::distance_t<A, B> operator()(const A& a, const B& b) const noexcept { return expand(reduced(a, b)); }
};

// distance_for unlocks the brute-force search, for any metric and any points
template <typename M, typename Q, typename P>
    requires mt::distance_for<M, Q, P>
std::size_t nearest(const M& m, const Q& q, const std::vector<P>& points)
{
    std::size_t best = 0;
    for (std::size_t i = 1; i < points.size(); ++i)
        if (mt::reduced_distance(m, q, points[i]) < mt::reduced_distance(m, q, points[best]))
            best = i;
    return best;
}

// decomposable unlocks the early-exit scan: count the points within a radius, stopping each sum once it is over
template <typename M, typename Q, typename P>
    requires mt::decomposable<M, mt::distance_t<Q, P>>
std::size_t within(const M& m, const Q& q, const std::vector<P>& points, double radius)
{
    std::size_t n = 0;
    const auto bound = mt::contract(m, static_cast<mt::distance_t<Q, P>>(radius));
    for (const auto& p : points)
        n += mt::accumulate_within(m, q, p, bound) <= bound ? 1 : 0;
    return n;
}

// true_metric unlocks pruning by the triangle inequality: d(q, p) >= |d(q, pivot) - d(pivot, p)|
template <typename M, typename Q, typename P>
    requires mt::true_metric<M> && mt::distance_for<M, Q, P>
double lower_bound_via(const M& m, const Q& q, const P& pivot, double pivot_to_p)
{
    return std::abs(static_cast<double>(m(q, pivot)) - pivot_to_p);
}

} // namespace
//! [concepts]

int main()
{
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("  %-66s %s\n", what, ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    };

    std::puts("Points: anything point_traits reads, mixed freely");
    const rgb teal{0, 128, 128};
    const std::array<double, 3> grey{128, 128, 128};
    const float red[3] = {255, 0, 0};
    const std::vector<int> white{255, 255, 255};
    const auto navy = la::fixed_vector<float, 3>{{0, 0, 128}};
    const std::span<const double, 3> grey_view(grey);
    check(mt::point<rgb> && mt::point<decltype(red)> && mt::point<decltype(navy)> && !mt::point<int>, "rgb, float[3], fixed_vector are points; int is not");
    check(std::is_same_v<mt::value_t<rgb>, float> && mt::extent_v<rgb> == 3 && mt::extent_v<std::vector<int>> == std::dynamic_extent,
          "value_t<rgb> is float, extent_v 3; a vector's extent is dynamic");
    check(std::is_same_v<mt::distance_t<rgb>, float> && std::is_same_v<mt::distance_t<rgb, std::array<double, 3>>, double>
              && std::is_same_v<mt::distance_t<std::vector<int>>, double>,
          "distance_t: float for two rgbs, double against double or int");
    check(mt::dimension(teal) == 3 && mt::coordinate(teal, 2) == 128.0f && mt::coordinate(grey_view, 0) == 128.0, "dimension and coordinate");
    check(mt::compatible_points<rgb, std::vector<int>> && !mt::compatible_points<rgb, std::array<double, 4>>, "compatible_points rejects 3 against 4");

    std::puts("Capabilities: what each metric declares");
    check(mt::distance_for<rounded_taxicab, rgb> && !mt::reducible<rounded_taxicab, rgb> && !mt::decomposable<rounded_taxicab, float>
              && !mt::true_metric<rounded_taxicab>,
          "rounded_taxicab: distance_for only");
    check(mt::distance_for<weighted_colour, rgb> && mt::reducible<weighted_colour, rgb> && mt::decomposable<weighted_colour, double>
              && mt::true_metric<weighted_colour>,
          "weighted_colour: distance_for, reducible, decomposable, true_metric");

    std::puts("distance_for: brute-force search with either metric");
    const std::vector<rgb> palette{{255, 0, 0}, {0, 0, 128}, {0, 128, 128}, {255, 255, 255}};
    const rgb sea{10, 120, 140};
    check(nearest(rounded_taxicab{}, sea, palette) == 2 && nearest(weighted_colour{}, sea, palette) == 2, "the nearest colour to (10, 120, 140) is teal");
    // |128 - 0| + 0 + 0 = 128 by hand; a metric with no cheaper form is its own reduced form
    check(rounded_taxicab{}(grey, teal) == 128.0 && mt::reduced_distance(rounded_taxicab{}, grey, teal) == 128.0, "taxicab(grey, teal) = 128, and so is its reduced form");
    check(mt::expand(rounded_taxicab{}, 3.0) == 3.0 && mt::contract(rounded_taxicab{}, 3.0) == 3.0, "expand and contract are the identity without a cheaper form");

    std::puts("reducible: compare in the cheaper form, convert at the edges");
    // (128 - 255)^2 + 2 (128)^2 + 128^2 = 16129 + 32768 + 16384 = 65281, by hand
    check(mt::reduced_distance(weighted_colour{}, grey, red) == 65281.0, "reduced(grey, red) = 65281");
    check(weighted_colour{}(grey, red) == std::sqrt(65281.0) && mt::expand(weighted_colour{}, 65281.0) == std::sqrt(65281.0), "distance = expand(reduced)");
    check(mt::contract(weighted_colour{}, 10.0) == 100.0, "contract(10) = 100, a radius as a reduced radius");

    std::puts("decomposable: term by term, with an early exit");
    // white against navy: 255^2, then 2 * 255^2, then 127^2 = 65025, 195075, 211204, by hand
    check(mt::accumulate(weighted_colour{}, white, navy) == 211204.0, "accumulate(white, navy) = 211204");
    check(mt::accumulate_within(weighted_colour{}, white, navy, 1000.0) == 65025.0, "accumulate_within(.., 1000) stops after one axis at 65025");
    check(mt::accumulate_within(weighted_colour{}, white, navy, 211204.0) == 211204.0, "accumulate_within at the exact bound is exact");
    check(within(weighted_colour{}, sea, palette, 30.0) == 1 && within(weighted_colour{}, sea, palette, 1000.0) == 4, "one colour within 30 of the sea, all four within 1000");

    std::puts("true_metric: prune by the triangle inequality");
    const double grey_to_red = weighted_colour{}(grey, red);
    const double bound = lower_bound_via(weighted_colour{}, teal, grey, grey_to_red);
    check(bound <= weighted_colour{}(teal, red), "|d(teal, grey) - d(grey, red)| <= d(teal, red)");

    std::printf("%s\n", failures == 0 ? "all results as documented" : "SOME RESULTS DIFFER");
    return failures == 0 ? 0 : 1;
}
