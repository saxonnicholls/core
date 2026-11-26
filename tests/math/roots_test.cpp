//
//  roots_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/math/roots.hpp>: bisection, Brent's method and the safeguarded Newton
//  iteration on polynomials with known roots, Kepler's equation and x = cos x, a pair of nearly double
//  roots, functions flat at the root; evaluation counts (Brent against bisection); every status;
//  bracketing outward; float and long double; constant evaluation.
//
//  Reference roots were computed in Python 3 with the decimal module at 50 digits (Newton's iteration
//  with series for sin and cos): Kepler's E for M = 1, e = 0.5 is 1.4987011335178483140579854972562399;
//  the Dottie number is 0.73908513321516064165531208767387340.
//

#include <snicholls/core/math/roots.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace {

namespace m = snicholls::math;
using m::root_status;

constexpr double kepler_e = 1.4987011335178483140579854972562399;
constexpr double dottie = 0.73908513321516064165531208767387340;

// ---- constant evaluation -------------------------------------------------------------------------
constexpr auto sqrt2_brent = m::brent([](double x) { return x * x - 2; }, 0.0, 2.0);
static_assert(sqrt2_brent.status == root_status::converged && m::near(sqrt2_brent.x, 1.4142135623730951, 1e-15));
constexpr auto sqrt2_bisect = m::bisect([](double x) { return x * x - 2; }, 0.0, 2.0);
static_assert(sqrt2_bisect.status == root_status::converged && m::near(sqrt2_bisect.x, 1.4142135623730951, 1e-15));
static_assert(sqrt2_brent.iterations < sqrt2_bisect.iterations);
constexpr auto sqrt2_newton = m::newton_safeguarded([](double x) { return std::pair{x * x - 2, 2 * x}; }, 0.0, 2.0, 1.0);
static_assert(sqrt2_newton.status == root_status::converged && m::near(sqrt2_newton.x, 1.4142135623730951, 1e-15));
constexpr auto kepler_ct = m::brent([](double E) { return E - 0.5 * m::sin(E) - 1.0; }, 0.0, 3.0);
static_assert(m::near(kepler_ct.x, kepler_e, 1e-15));
constexpr auto cube_bracket = m::bracket([](double x) { return x * x * x - 1000; }, 0.0, 1.0);
static_assert(cube_bracket.status == root_status::converged && cube_bracket.lo < 10 && cube_bracket.hi > 10);
static_assert(m::bisect([](double x) { return x; }, 1.0, 2.0).status == root_status::not_bracketed);

// Every finder on f, counting evaluations through the callable itself
template <typename F>
struct counted
{
    F f;
    int* n;
    constexpr double operator()(double x) const { ++*n; return f(x); }
};

TEST(RootsPolynomial, AllThreeFindKnownRoots)
{
    // (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6, one root per bracket
    const auto p = [](double x) { return ((x - 6) * x + 11) * x - 6; };
    const auto dp = [&](double x) { return std::pair{p(x), (3 * x - 12) * x + 11}; };
    for (const auto& [lo, hi, root] : {std::tuple{0.5, 1.5, 1.0}, std::tuple{1.6, 2.7, 2.0}, std::tuple{2.5, 10.0, 3.0}}) {
        const auto a = m::bisect(p, lo, hi), b = m::brent(p, lo, hi), c = m::newton_safeguarded(dp, lo, hi, (lo + hi) / 2);
        for (const auto& r : {a, b, c}) {
            EXPECT_EQ(r.status, root_status::converged);
            EXPECT_NEAR(r.x, root, 4e-15 * root);   // within the rounding noise of the cubic's evaluation
            EXPECT_EQ(r.f, p(r.x));
        }
    }
    // A quintic with an irrational root: x^5 - x - 1 = 0 at 1.1673039782614187 (Python decimal, 50 digits)
    const auto q = [](double x) { return x * x * x * x * x - x - 1; };
    EXPECT_NEAR(m::brent(q, 1.0, 2.0).x, 1.1673039782614187, 4e-16);
    EXPECT_NEAR(m::bisect(q, 2.0, 1.0).x, 1.1673039782614187, 4e-16);   // ends in either order
}

TEST(RootsTranscendental, KeplersEquationAndTheDottieNumber)
{
    const auto kepler = [](double E) { return E - 0.5 * std::sin(E) - 1.0; };
    const auto kepler_d = [](double E) { return std::pair{E - 0.5 * std::sin(E) - 1.0, 1 - 0.5 * std::cos(E)}; };
    EXPECT_NEAR(m::brent(kepler, 0.0, m::pi).x, kepler_e, 4e-16);
    EXPECT_NEAR(m::bisect(kepler, 0.0, m::pi).x, kepler_e, 4e-16);
    EXPECT_NEAR(m::newton_safeguarded(kepler_d, 0.0, m::pi, 1.0).x, kepler_e, 4e-16);
    const auto dottie_f = [](double x) { return x - std::cos(x); };
    EXPECT_NEAR(m::brent(dottie_f, 0.0, 1.0).x, dottie, 2e-16);
    EXPECT_NEAR(m::newton_safeguarded([](double x) { return std::pair{x - std::cos(x), 1 + std::sin(x)}; }, 0.0, 1.0, 0.5).x, dottie, 2e-16);
    // An absolute tolerance stops early, within it
    const auto loose = m::brent(dottie_f, 0.0, 1.0, 1e-6);
    EXPECT_EQ(loose.status, root_status::converged);
    EXPECT_NEAR(loose.x, dottie, 1e-6);
    EXPECT_LE(loose.iterations, m::brent(dottie_f, 0.0, 1.0).iterations);
}

TEST(RootsNearDoubleRoot, SeparatesTwoRootsOneMillionthApart)
{
    // (x - 1)(x - 1 - 1e-6): f is at most 2.5e-13 between them
    const auto f = [](double x) { return (x - 1) * (x - 1 - 1e-6); };
    const auto df = [&](double x) { return std::pair{f(x), 2 * x - 2 - 1e-6}; };
    for (const auto& r : {m::bisect(f, 0.0, 1.0000005), m::brent(f, 0.0, 1.0000005), m::newton_safeguarded(df, 0.0, 1.0000005, 0.5)}) {
        EXPECT_EQ(r.status, root_status::converged);
        EXPECT_NEAR(r.x, 1.0, 4e-16);
    }
    for (const auto& r : {m::bisect(f, 1.0000005, 3.0), m::brent(f, 1.0000005, 3.0), m::newton_safeguarded(df, 1.0000005, 3.0, 2.0)}) {
        EXPECT_EQ(r.status, root_status::converged);
        EXPECT_NEAR(r.x, 1.000001, 4e-16);
    }
    // A double root has no sign change: not bracketed, honestly
    EXPECT_EQ(m::brent([](double x) { return (x - 1) * (x - 1); }, 0.0, 3.0).status, root_status::not_bracketed);
}

TEST(RootsFlat, ConvergeWhereFIsFlatAtTheRoot)
{
    // (x - 1)^5 and x^9 have a root of multiplicity five and nine: Newton's step only shrinks by (m-1)/m, so the
    // safeguard must bisect; Brent must not stall.
    const auto f5 = [](double x) { const double d = x - 1; return d * d * d * d * d; };
    const auto f5d = [&](double x) { const double d = x - 1; return std::pair{f5(x), 5 * d * d * d * d}; };
    for (const auto& r : {m::bisect(f5, 0.0, 3.0), m::brent(f5, 0.0, 3.0), m::newton_safeguarded(f5d, 0.0, 3.0, 2.5)}) {
        EXPECT_EQ(r.status, root_status::converged);
        EXPECT_NEAR(r.x, 1.0, 4e-15);
    }
    const auto f9 = [](double x) { const double x3 = x * x * x; return x3 * x3 * x3; };
    EXPECT_NEAR(m::brent(f9, -1.0, 2.0).x, 0.0, 1e-30);
    // x e^(-1/x^2): every derivative vanishes at 0, and f underflows to 0 below |x| = 0.04
    const auto f_flat = [](double x) { return x == 0 ? 0.0 : x * std::exp(-1 / (x * x)); };
    const auto r = m::brent(f_flat, -1.0, 0.7);
    EXPECT_EQ(r.status, root_status::converged);
    EXPECT_EQ(r.f, 0.0);
    EXPECT_LT(std::abs(r.x), 0.05);
    // The expanded quintic is flat and noisy near 1: the bracket closes on a sign change of the rounding
    const auto noisy = [](double x) { return ((((x - 5) * x + 10) * x - 10) * x + 5) * x - 1; };
    const auto rn = m::brent(noisy, 0.0, 3.0);
    EXPECT_EQ(rn.status, root_status::converged);
    EXPECT_NEAR(rn.x, 1.0, 2e-3);
}

TEST(RootsCounts, BrentBeatsBisectionOnSmoothFunctions)
{
    struct problem { double (*f)(double); double lo, hi; };
    const problem problems[] = {
        {[](double x) { return x * x - 2; }, 0, 2},
        {[](double x) { return std::exp(x) - 10; }, 0, 5},
        {[](double x) { return x - std::cos(x); }, 0, 1},
        {[](double x) { return std::log(x) - 1; }, 1, 10},
        {[](double x) { return x * x * x - 2 * x - 5; }, 2, 3},
    };
    for (const auto& p : problems) {
        int nb = 0, nr = 0;
        const auto b = m::bisect(counted<double (*)(double)>{p.f, &nb}, p.lo, p.hi);
        const auto r = m::brent(counted<double (*)(double)>{p.f, &nr}, p.lo, p.hi);
        EXPECT_EQ(b.iterations + 2, nb);   // iterations count the evaluations after the two endpoints
        EXPECT_EQ(r.iterations + 2, nr);
        EXPECT_LE(r.iterations, 12);
        EXPECT_GE(b.iterations, 45);
        EXPECT_LT(3 * r.iterations, b.iterations) << p.lo;
        EXPECT_NEAR(r.x, b.x, 4e-16 * std::abs(b.x));
    }
}

TEST(RootsStatus, EveryStatusIsReported)
{
    const auto f = [](double x) { return x * x - 2; };
    const auto fd = [](double x) { return std::pair{x * x - 2, 2 * x}; };
    // not bracketed: same sign, or NaN at an end
    EXPECT_EQ(m::bisect(f, 2.0, 3.0).status, root_status::not_bracketed);
    EXPECT_EQ(m::brent(f, 2.0, 3.0).status, root_status::not_bracketed);
    EXPECT_EQ(m::newton_safeguarded(fd, 2.0, 3.0, 2.5).status, root_status::not_bracketed);
    EXPECT_EQ(m::brent([](double x) { return std::sqrt(x); }, -1.0, 1.0).status, root_status::not_bracketed);
    // max_iterations: the budget runs out, the best point so far comes back
    const auto b = m::bisect(f, 0.0, 2.0, 0.0, 5);
    EXPECT_EQ(b.status, root_status::max_iterations);
    EXPECT_EQ(b.iterations, 5);
    EXPECT_NEAR(b.x, 1.4142135623730951, 2.0 / 32);
    const auto r = m::brent(f, 0.0, 2.0, 0.0, 2);
    EXPECT_EQ(r.status, root_status::max_iterations);
    EXPECT_EQ(r.iterations, 2);
    EXPECT_EQ(r.f, f(r.x));
    EXPECT_EQ(m::newton_safeguarded(fd, 0.0, 2.0, 1.0, 0.0, 2).status, root_status::max_iterations);
    // converged at an endpoint, with no iteration
    const auto at_end = m::bisect([](double x) { return x - 1; }, 1.0, 2.0);
    EXPECT_EQ(at_end.status, root_status::converged);
    EXPECT_EQ(at_end.x, 1.0);
    EXPECT_EQ(at_end.iterations, 0);
    EXPECT_EQ(m::brent([](double x) { return x - 2; }, 1.0, 2.0).iterations, 0);
    EXPECT_EQ(m::newton_safeguarded([](double x) { return std::pair{x - 2, 1.0}; }, 1.0, 2.0, 1.5).x, 2.0);
    // A midpoint that is the root exactly
    const auto mid = m::bisect([](double x) { return x - 1; }, 0.0, 2.0);
    EXPECT_EQ(mid.x, 1.0);
    EXPECT_EQ(mid.f, 0.0);
    EXPECT_EQ(mid.iterations, 1);
}

TEST(RootsNewton, FallsBackToBisectionWhenTheStepLeavesTheBracket)
{
    // atan: Newton from 1.5 overshoots to about -1.69 and diverges; the safeguard keeps it inside [-1, 2]
    const auto fd = [](double x) { return std::pair{std::atan(x), 1 / (1 + x * x)}; };
    const auto r = m::newton_safeguarded(fd, -1.0, 2.0, 1.5);
    EXPECT_EQ(r.status, root_status::converged);
    EXPECT_NEAR(r.x, 0.0, 1e-15);
    // A zero derivative at the start bisects instead of dividing by zero
    const auto g = [](double x) { return std::pair{x * x * x - 0.001, 3 * x * x}; };
    const auto s = m::newton_safeguarded(g, -1.0, 1.0, 0.0);
    EXPECT_NEAR(s.x, 0.1, 1e-16);
    // A first guess outside the bracket starts from the midpoint
    EXPECT_NEAR(m::newton_safeguarded(g, -1.0, 1.0, 5.0).x, 0.1, 1e-16);
    // Fewer evaluations than Brent near a simple root, with the derivative free
    const auto kd = [](double E) { return std::pair{E - 0.5 * std::sin(E) - 1.0, 1 - 0.5 * std::cos(E)}; };
    EXPECT_LE(m::newton_safeguarded(kd, 0.0, 3.0, 1.0).iterations, m::brent([](double E) { return E - 0.5 * std::sin(E) - 1.0; }, 0.0, 3.0).iterations);
}

TEST(RootsBracket, ExpandsOutwardUntilTheSignChanges)
{
    const auto f = [](double x) { return std::exp(x) - 1e6; };   // root at log(1e6) = 13.8
    const auto b = m::bracket(f, 0.0, 0.5);
    EXPECT_EQ(b.status, root_status::converged);
    EXPECT_LT(b.lo, std::log(1e6));
    EXPECT_GT(b.hi, std::log(1e6));
    EXPECT_EQ(b.f_lo, f(b.lo));
    EXPECT_EQ(b.f_hi, f(b.hi));
    EXPECT_GT(b.iterations, 0);
    EXPECT_NEAR(m::brent(f, b.lo, b.hi).x, std::log(1e6), 1e-14);
    // Already bracketing: no expansion
    EXPECT_EQ(m::bracket([](double x) { return x; }, 0.1, 1.0).iterations, 0);
    // Leftward
    const auto left = m::bracket([](double x) { return x + 100; }, 0.0, 1.0);
    EXPECT_EQ(left.status, root_status::converged);
    EXPECT_LT(left.lo, -100.0);
    // No root: the budget runs out
    const auto none = m::bracket([](double x) { return x * x + 1; }, 0.0, 1.0, 1.6, 20);
    EXPECT_EQ(none.status, root_status::max_iterations);
    EXPECT_EQ(none.iterations, 20);
    // A zero or non-finite step
    EXPECT_EQ(m::bracket([](double x) { return x - 3; }, 0.0, 0.0).status, root_status::not_bracketed);
    EXPECT_EQ(m::bracket([](double x) { return x - 3; }, 0.0, std::numeric_limits<double>::infinity()).status, root_status::not_bracketed);
    EXPECT_EQ(m::bracket([](double x) { return x - 3; }, 0.0, -1.0).status, root_status::converged);   // the step's sign is ignored
}

TEST(RootsTypes, FloatAndLongDouble)
{
    const auto rf = m::brent([](float x) { return x * x - 2.0f; }, 0.0f, 2.0f);
    static_assert(std::is_same_v<decltype(rf), const m::root_result<float>>);
    EXPECT_EQ(rf.status, root_status::converged);
    EXPECT_NEAR(rf.x, 1.41421356f, 2e-7f);
    EXPECT_NEAR(m::bisect([](float x) { return x * x - 2.0f; }, 0.0f, 2.0f, 1e-3f).x, 1.41421356f, 1e-3f);
    const auto rl = m::brent([](long double x) { return x * x - 2.0L; }, 0.0L, 2.0L);
    EXPECT_LE(std::abs(rl.x - 1.41421356237309504880168872420969808L), 4 * std::numeric_limits<long double>::epsilon());   // a few ulp: quad's roots round in its last bits
    const auto nl = m::newton_safeguarded([](long double x) { return std::pair{x * x - 2.0L, 2 * x}; }, 0.0L, 2.0L, 1.0L);
    EXPECT_LE(std::abs(nl.x - 1.41421356237309504880168872420969808L), 4 * std::numeric_limits<long double>::epsilon());   // a few ulp: quad's roots round in its last bits
    constexpr auto ct = m::brent([](float x) { return x * x * x - 8.0f; }, 0.0f, 5.0f);
    static_assert(ct.x == 2.0f);
}

} // namespace
