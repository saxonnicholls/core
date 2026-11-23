//
//  cones_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/optim/cones.hpp>. Each projection is checked by its optimality conditions, never only by
//  a round trip: P(x) lies in the cone, x - P(x) in the polar cone and orthogonal to P(x) (Moreau's decomposition
//  x = P_K(x) - P_K*(-x)), and P is idempotent. Membership in the exponential and power cones and their duals is
//  checked by the defining inequalities, written out here, not by the header's functions. The PSD projection is
//  compared with the eigen decomposition rebuilt here and with numpy, the simplex and l1 ball with the sort-based
//  O(n log n) projection, the capped simplex with its KKT conditions. Then the product of cones, the Nesterov-Todd
//  scalings (W z = W^-1 s, symmetry, the cone mapped into itself), the errors, and constant evaluation.
//

#include <snicholls/core/optim/cones.hpp>

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

namespace {

namespace op = snicholls::optim;
namespace st = snicholls::stats;
namespace la = snicholls::linalg;

using vec = std::vector<double>;
using v3 = std::array<double, 3>;

double dot(std::span<const double> a, std::span<const double> b)
{
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}
double norm(std::span<const double> a) { return std::sqrt(dot(a, a)); }

double uniform(st::wyrand& g, double lo, double hi) { return lo + (hi - lo) * st::uniform_real(g); }

vec random_vector(st::wyrand& g, std::size_t n, double lo = -2, double hi = 2)
{
    vec v(n);
    for (double& x : v)
        x = uniform(g, lo, hi);
    return v;
}

// Moreau's decomposition for a cone K with dual K*: p = P_K(x) in K, d = P_K(x) - x in K*, <p, d> = 0, and P idempotent.
// in_k and in_dual are the test's own membership checks with an absolute tolerance; the checks are made on the point
// divided by its largest magnitude (at least 1), so that 1e200 does not overflow a product.
template <typename Project, typename InK, typename InDual>
void check_moreau(vec x, Project project, InK in_k, InDual in_dual, double tol, const char* what)
{
    const vec x0 = x;
    project(std::span<double>(x));
    double scale = 1;
    for (const double v : x0)
        scale = std::max(scale, std::abs(v));
    vec p(x.size()), d(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        p[i] = x[i] / scale;
        d[i] = (x[i] - x0[i]) / scale;
    }
    std::ostringstream at;
    at.precision(17);
    at << what;
    for (const double v : x0)
        at << " " << v;
    EXPECT_TRUE(in_k(p, tol)) << at.str();
    EXPECT_TRUE(in_dual(d, tol)) << at.str();
    EXPECT_LE(std::abs(dot(p, d)), tol) << at.str();
    vec again = x;
    project(std::span<double>(again));
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(again[i] / scale, p[i], tol) << at.str();
    }
}

// ---- The test's own membership checks ----

bool soc_member(const vec& x, double tol)
{
    double s = 0;
    for (std::size_t i = 1; i < x.size(); ++i)
        s += x[i] * x[i];
    return std::sqrt(s) <= x[0] + tol;
}

// K_exp: y > 0 and y e^{x/y} <= z, or y = 0, x <= 0, z >= 0. The cone is monotone (decreasing x or increasing z
// stays inside), so a point is within sqrt(2) tol of it when (x - tol, y, z + tol) satisfies the inequality, written
// as x <= y log(z / y). (The inequality itself is ill-conditioned where y is near 0: a component of the polar part
// that is a difference of nearly equal numbers sits in an exponent.)
bool exp_member(const vec& p, double tol)
{
    const double x = p[0] - tol, y = p[1], z = p[2] + tol;
    if (y < -tol || z < 0)
        return false;
    if (y <= 0)
        return x <= 0;
    return x <= y * std::log(z / y);
}
// K_exp*: u < 0 and -u e^{v/u} <= e w (v >= u (1 + log(w / -u))), or u = 0, v >= 0, w >= 0; monotone in v and w
bool exp_dual_member(const vec& d, double tol)
{
    const double u = d[0], v = d[1] + tol, w = d[2] + tol;
    if (u > tol || w < 0)
        return false;
    if (u >= 0)
        return v >= 0;
    return v >= u * (1 + std::log(w / -u));
}

// The power cone and its dual are monotone in their first two elements, so a point is within sqrt(2) tol of the
// cone when the point moved by tol in both satisfies the inequality (the inequality itself is ill-conditioned where
// x or y is near 0 and alpha near 0 or 1: x^0.01 changes by 20% between 1e-72 and 1e-80).
bool pow_member(const vec& p, double a, double tol)
{
    if (p[0] < -tol || p[1] < -tol)
        return false;
    return std::pow(std::max(p[0] + tol, 0.0), a) * std::pow(std::max(p[1] + tol, 0.0), 1 - a) >= std::abs(p[2]) - tol;
}
bool pow_dual_member(const vec& d, double a, double tol)
{
    if (d[0] < -tol || d[1] < -tol)
        return false;
    return std::pow(std::max(d[0] + tol, 0.0) / a, a) * std::pow(std::max(d[1] + tol, 0.0) / (1 - a), 1 - a) >= std::abs(d[2]) - tol;
}

// ---- Zero, free, orthant, box ----

TEST(OptimConesZero, ZeroAndFreeAreEachOthersDualAndDecomposeEveryPoint)
{
    vec x{1.5, -2, 0};
    op::project_free(std::span<double>(x));
    EXPECT_EQ(x, (vec{1.5, -2, 0}));
    EXPECT_TRUE(op::in_free_cone(std::span<const double>(x), 0.0));
    EXPECT_FALSE(op::in_zero_cone(std::span<const double>(x), 1.0));
    op::project_zero(std::span<double>(x));
    EXPECT_EQ(x, (vec{0, 0, 0}));
    EXPECT_TRUE(op::in_zero_cone(std::span<const double>(x), 0.0));
    const vec nan{std::numeric_limits<double>::quiet_NaN()};
    EXPECT_FALSE(op::in_free_cone(std::span<const double>(nan), 0.0));
    EXPECT_EQ(op::dual_of(op::cone::zero), op::cone::free);
    EXPECT_EQ(op::dual_of(op::cone::free), op::cone::zero);
    EXPECT_EQ(op::dual_of(op::cone::psd), op::cone::psd);
    EXPECT_EQ(op::dual_of(op::cone::power), op::cone::dual_power);
    EXPECT_EQ(op::dual_of(op::cone::dual_exponential), op::cone::exponential);
}

TEST(OptimConesNonnegative, ProjectionSatisfiesMoreauOnRandomPoints)
{
    st::wyrand g(1);
    auto in_k = [](const vec& x, double tol) {
        return std::all_of(x.begin(), x.end(), [&](double v) { return v >= -tol; });
    };
    for (int t = 0; t < 50; ++t)
        check_moreau(random_vector(g, 7), [](std::span<double> x) { op::project_nonnegative(x); }, in_k, in_k, 1e-14, "orthant");
    const vec y{-1, 0, 2};
    EXPECT_TRUE(op::in_nonnegative_cone(std::span<const double>(y), 1.0));
    EXPECT_FALSE(op::in_nonnegative_cone(std::span<const double>(y), 0.5));
}

TEST(OptimConesBox, ClampsWithInfiniteBoundsAndReportsMismatchedSpans)
{
    const double inf = std::numeric_limits<double>::infinity();
    vec x{-5, 0.5, 9, -1e300, 1e300};
    const vec l{-1, 0, -inf, -inf, 2}, u{1, 1, 3, inf, inf};
    ASSERT_TRUE(op::project_box(std::span<double>(x), l, u));
    EXPECT_EQ(x, (vec{-1, 0.5, 3, -1e300, 1e300}));
    EXPECT_TRUE(op::in_box(std::span<const double>(x), l, u, 0.0));
    vec y{4, -4};
    op::project_box(std::span<double>(y), -2.0, 2.0);
    EXPECT_EQ(y, (vec{2, -2}));
    const vec shorter{0};
    vec z{1, 2};
    EXPECT_EQ(op::project_box(std::span<double>(z), shorter, shorter).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_EQ(z, (vec{1, 2}));
    EXPECT_FALSE(op::in_box(std::span<const double>(z), shorter, shorter, 0.0));
    const vec outside{1.1};
    EXPECT_TRUE(op::in_box(std::span<const double>(outside), std::span<const double>(vec{0}), std::span<const double>(vec{1}), 0.2));
    EXPECT_FALSE(op::in_box(std::span<const double>(outside), std::span<const double>(vec{0}), std::span<const double>(vec{1}), 0.05));
}

// ---- Second-order cones ----

TEST(OptimConesSecondOrder, ClosedFormInEachRegion)
{
    vec inside{2, 1, -1};
    op::project_second_order(std::span<double>(inside));
    EXPECT_EQ(inside, (vec{2, 1, -1}));
    vec polar{-2, 1, -1};
    op::project_second_order(std::span<double>(polar));
    EXPECT_EQ(polar, (vec{0, 0, 0}));
    vec between{0, 3, 4};   // ((0 + 5)/2) (1, 3/5, 4/5)
    op::project_second_order(std::span<double>(between));
    EXPECT_DOUBLE_EQ(between[0], 2.5);
    EXPECT_DOUBLE_EQ(between[1], 1.5);
    EXPECT_DOUBLE_EQ(between[2], 2.0);
    vec scalar{-3};
    op::project_second_order(std::span<double>(scalar));
    EXPECT_EQ(scalar[0], 0.0);
    vec empty;
    op::project_second_order(std::span<double>(empty));
    EXPECT_TRUE(op::in_second_order_cone(std::span<const double>(empty), 0.0));
}

TEST(OptimConesSecondOrder, ProjectionSatisfiesMoreauOnRandomPoints)
{
    st::wyrand g(2);
    for (int t = 0; t < 200; ++t)
        check_moreau(random_vector(g, 1 + static_cast<std::size_t>(t % 6)), [](std::span<double> x) { op::project_second_order(x); }, soc_member, soc_member, 1e-13, "soc");
}

TEST(OptimConesRotatedSecondOrder, ProjectionSatisfiesMoreauAndTheDefiningInequality)
{
    // 2uv >= |w|^2, u, v >= 0
    auto in_k = [](const vec& x, double tol) {
        double w = 0;
        for (std::size_t i = 2; i < x.size(); ++i)
            w += x[i] * x[i];
        return x[0] >= -tol && x[1] >= -tol && 2 * x[0] * x[1] >= w - tol * (1 + std::abs(x[0]) + std::abs(x[1]));
    };
    st::wyrand g(3);
    for (int t = 0; t < 200; ++t)
        check_moreau(random_vector(g, 2 + static_cast<std::size_t>(t % 5)), [](std::span<double> x) { ASSERT_TRUE(op::project_rotated_second_order(x)); }, in_k, in_k, 1e-13, "rsoc");
    vec inside{1, 2, 1.9};   // 2 * 1 * 2 = 4 >= 3.61: unchanged to the bit
    ASSERT_TRUE(op::project_rotated_second_order(std::span<double>(inside)));
    EXPECT_EQ(inside, (vec{1, 2, 1.9}));
    EXPECT_TRUE(op::in_rotated_second_order_cone(std::span<const double>(inside), 0.0));
    const vec outside{1, 2, 2.1};
    EXPECT_FALSE(op::in_rotated_second_order_cone(std::span<const double>(outside), 0.0));
    vec one{1};
    EXPECT_EQ(op::project_rotated_second_order(std::span<double>(one)).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_FALSE(op::in_rotated_second_order_cone(std::span<const double>(one), 0.0));
}

// ---- PSD ----

TEST(OptimConesPsd, SvecIsScaledLowerTriangleColumnByColumnAndPreservesTheTraceInnerProduct)
{
    const vec a{1, 2, 3, 2, 5, 6, 3, 6, 9}, b{2, -1, 0, -1, 4, 1, 0, 1, -3};
    vec sa(6), sb(6), back(9);
    ASSERT_TRUE(op::svec(std::span<const double>(a), std::span<double>(sa)));
    ASSERT_TRUE(op::svec(std::span<const double>(b), std::span<double>(sb)));
    const double r2 = std::sqrt(2.0);
    EXPECT_EQ(sa, (vec{1, 2 * r2, 3 * r2, 5, 6 * r2, 9}));
    double trace_ab = 0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            trace_ab += a[i * 3 + k] * b[k * 3 + i];
    EXPECT_NEAR(dot(sa, sb), trace_ab, 1e-12);
    ASSERT_TRUE(op::smat(std::span<const double>(sa), std::span<double>(back)));
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(back[i], a[i], 1e-14);
    }
    EXPECT_EQ(op::svec_size(3), 6u);
    EXPECT_EQ(*op::psd_order(10), 4u);
    EXPECT_EQ(op::psd_order(5).unwrap_err(), op::error::dimension_mismatch);
    vec five(5);
    EXPECT_EQ(op::svec(std::span<const double>(a), std::span<double>(five)).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_EQ(op::smat(std::span<const double>(five), std::span<double>(back)).unwrap_err(), op::error::dimension_mismatch);
}

TEST(OptimConesPsd, ProjectionMatchesNumpy)
{
    // numpy: A = [[1,2,3],[2,-1,0],[3,0,2]]; w, V = eigh(A); V @ diag(max(w, 0)) @ V.T
    const vec a{1, 2, 3, 2, -1, 0, 3, 0, 2};
    const vec expected{2.18541403, 0.76325555, 2.27674425, 0.76325555, 0.29029756, 0.75457394, 2.27674425, 0.75457394, 2.44127948};
    vec x(6), ws(op::psd_workspace_size(3)), m(9);
    ASSERT_TRUE(op::svec(std::span<const double>(a), std::span<double>(x)));
    ASSERT_TRUE(op::project_psd(std::span<double>(x), std::span<double>(ws)));
    ASSERT_TRUE(op::smat(std::span<const double>(x), std::span<double>(m)));
    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_NEAR(m[i], expected[i], 1e-8);
    }
}

// The projection rebuilt here: V max(L, 0) V^T from the fixed eigen decomposition, summed over every eigenvalue
template <std::size_t N>
vec psd_reference(const vec& a)
{
    la::fixed_matrix<double, N, N> m{};
    for (std::size_t i = 0; i < N * N; ++i)
        m.elements[i] = a[i];
    const auto e = la::symmetric_eigen(m);
    vec p(N * N, 0.0);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t k = 0; k < N; ++k)
                p[i * N + j] += std::max(e->values[k], 0.0) * e->vectors(i, k) * e->vectors(j, k);
    return p;
}

template <std::size_t N>
void psd_random(st::wyrand& g)
{
    constexpr std::size_t k = N * (N + 1) / 2;
    vec a(N * N);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            a[i * N + j] = a[j * N + i] = uniform(g, -2, 2);
    vec x(k), ws(op::psd_workspace_size(N)), m(N * N);
    ASSERT_TRUE(op::svec(std::span<const double>(a), std::span<double>(x)));
    const vec x0 = x;
    ASSERT_TRUE(op::project_psd(std::span<double>(x), std::span<double>(ws)));
    ASSERT_TRUE(op::smat(std::span<const double>(x), std::span<double>(m)));
    const vec ref = psd_reference<N>(a);
    for (std::size_t i = 0; i < N * N; ++i) {
        EXPECT_NEAR(m[i], ref[i], 1e-11) << N;
    }
    // The fixed form gives the same
    std::array<double, k> f{};
    std::copy(x0.begin(), x0.end(), f.begin());
    ASSERT_TRUE(op::project_psd_fixed<N>(std::span<double, k>(f)));
    for (std::size_t i = 0; i < k; ++i) {
        EXPECT_NEAR(f[i], x[i], 1e-11) << N;
    }
    // Optimality: P in the cone, P - X in the cone, <P, P - X> = 0 (svec inner product is the trace one)
    vec d(k);
    for (std::size_t i = 0; i < k; ++i)
        d[i] = x[i] - x0[i];
    EXPECT_TRUE(op::in_psd_cone(std::span<const double>(x), 1e-11, std::span<double>(ws)));
    EXPECT_TRUE(op::in_psd_cone(std::span<const double>(d), 1e-11, std::span<double>(ws)));
    EXPECT_TRUE((op::in_psd_cone_fixed<N>(std::span<const double, k>(f), 1e-11)));
    EXPECT_NEAR(dot(x, d), 0.0, 1e-10);
    // Idempotent
    vec again = x;
    ASSERT_TRUE(op::project_psd(std::span<double>(again), std::span<double>(ws)));
    for (std::size_t i = 0; i < k; ++i) {
        EXPECT_NEAR(again[i], x[i], 1e-11);
    }
}

TEST(OptimConesPsd, ProjectionMatchesTheEigenDecompositionAndSatisfiesMoreau)
{
    st::wyrand g(4);
    for (int t = 0; t < 10; ++t) {
        psd_random<1>(g);
        psd_random<2>(g);
        psd_random<3>(g);
        psd_random<4>(g);
        psd_random<6>(g);
    }
}

TEST(OptimConesPsd, AlreadyPsdIsUnchangedAndErrorsAreReported)
{
    vec x{2, 0.5, 3};   // [[2, 0.354], [0.354, 3]]: positive definite
    vec ws(op::psd_workspace_size(2));
    ASSERT_TRUE(op::project_psd(std::span<double>(x), std::span<double>(ws)));
    EXPECT_EQ(x, (vec{2, 0.5, 3}));
    vec neg{-1, 0, -2};
    ASSERT_TRUE(op::project_psd(std::span<double>(neg), std::span<double>(ws)));
    EXPECT_NEAR(norm(neg), 0.0, 1e-15);
    EXPECT_FALSE(op::in_psd_cone(std::span<const double>(vec{-1, 0, -2}), 0.5, std::span<double>(ws)));
    vec small(3);
    EXPECT_EQ(op::project_psd(std::span<double>(x), std::span<double>(small)).unwrap_err(), op::error::workspace_too_small);
    vec four(4);
    EXPECT_EQ(op::project_psd(std::span<double>(four), std::span<double>(ws)).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_FALSE(op::in_psd_cone(std::span<const double>(four), 0.0, std::span<double>(ws)));
    vec none;
    EXPECT_TRUE(op::project_psd(std::span<double>(none), std::span<double>(none)));
    EXPECT_TRUE(op::in_psd_cone(std::span<const double>(none), 0.0, std::span<double>(none)));
    std::array<double, 3> nan{1, std::numeric_limits<double>::quiet_NaN(), 1};   // Jacobi reports a NaN off the diagonal
    EXPECT_EQ(op::project_psd_fixed<2>(std::span<double, 3>(nan)).unwrap_err(), op::error::did_not_converge);
}

// ---- Exponential cone ----

void check_exp(const v3& v, double tol = 1e-9)
{
    check_moreau(vec(v.begin(), v.end()), [](std::span<double> x) { op::project_exponential(x.first<3>()); }, exp_member, exp_dual_member, tol, "exp");
    // The dual projection by its own conditions: q in K*, q - v in K, orthogonal
    check_moreau(vec(v.begin(), v.end()), [](std::span<double> x) { op::project_dual_exponential(x.first<3>()); }, exp_dual_member, exp_member, tol, "exp dual");
}

TEST(OptimConesExponential, AnalyticCasesAreExact)
{
    v3 in{1, 1, 3};   // 1 * e^1 = 2.718 <= 3
    op::project_exponential(std::span(in));
    EXPECT_EQ(in, (v3{1, 1, 3}));
    v3 polar{1, 1, -3};   // -v = (-1, -1, 3) in K*: 1 * e^{1} <= 3e
    op::project_exponential(std::span(polar));
    EXPECT_EQ(polar, (v3{0, 0, 0}));
    v3 both_negative{-2, -1, 0.5};   // x <= 0 and y <= 0: (x, 0, max(z, 0))
    op::project_exponential(std::span(both_negative));
    EXPECT_EQ(both_negative, (v3{-2, 0, 0.5}));
    v3 both_negative_z{-2, -1, -0.5};
    op::project_exponential(std::span(both_negative_z));
    EXPECT_EQ(both_negative_z, (v3{-2, 0, 0}));
    v3 zero{0, 0, 0};
    op::project_exponential(std::span(zero));
    EXPECT_EQ(zero, (v3{0, 0, 0}));
}

TEST(OptimConesExponential, MatchesAnIndependentSolver)
{
    // scipy.optimize.minimize (SLSQP, ftol 1e-15) of |p - v|^2 / 2 subject to y log(z / y) >= x
    v3 a{1, 1, 1};
    op::project_exponential(std::span(a));
    EXPECT_NEAR(a[0], 0.42630617, 1e-6);
    EXPECT_NEAR(a[1], 0.75167277, 1e-6);
    EXPECT_NEAR(a[2], 1.32536659, 1e-6);
    v3 b{-1, 2, 0.5};
    op::project_exponential(std::span(b));
    EXPECT_NEAR(b[0], -1.17644628, 1e-6);
    EXPECT_NEAR(b[1], 1.70156004, 1e-6);
    EXPECT_NEAR(b[2], 0.85227396, 1e-6);
}

TEST(OptimConesExponential, MoreauHoldsInAllFourRegionsOnARandomCloud)
{
    st::wyrand g(5);
    int regions[4] = {0, 0, 0, 0};
    for (int t = 0; t < 2000; ++t) {
        const v3 v{uniform(g, -3, 3), uniform(g, -3, 3), uniform(g, -3, 3)};
        const vec x(v.begin(), v.end());
        const bool in = exp_member(x, 0), polar = exp_dual_member(vec{-v[0], -v[1], -v[2]}, 0);
        ++regions[in ? 0 : polar ? 1 : (v[0] <= 0 && v[1] <= 0) ? 2 : 3];
        check_exp(v);
    }
    for (int r : regions) {
        EXPECT_GT(r, 50);
    }
}

TEST(OptimConesExponential, MoreauHoldsForComponentsOfMixedMagnitudes)
{
    // Each component +-10^u, u uniform in [-8, 8]: the regions where rho is huge, tiny, or the polar part cancels
    st::wyrand g(13);
    for (int t = 0; t < 5000; ++t) {
        v3 v{};
        for (double& c : v)
            c = (st::uniform_real(g) < 0.5 ? -1 : 1) * std::pow(10.0, uniform(g, -8, 8));
        check_exp(v);
    }
}

TEST(OptimConesPower, MoreauHoldsForComponentsOfMixedMagnitudes)
{
    st::wyrand g(14);
    for (int t = 0; t < 5000; ++t) {
        const double a = uniform(g, 0.02, 0.98);
        vec v(3);
        for (double& c : v)
            c = (st::uniform_real(g) < 0.5 ? -1 : 1) * std::pow(10.0, uniform(g, -8, 8));
        check_moreau(v, [a](std::span<double> x) { op::project_power(x.first<3>(), a); }, [a](const vec& x, double tol) { return pow_member(x, a, tol); },
                     [a](const vec& x, double tol) { return pow_dual_member(x, a, tol); }, 1e-9, "pow mixed");
    }
}

TEST(OptimConesExponential, FribergsHardCases)
{
    const double e = std::exp(1.0);
    const v3 cases[] = {
        {1, 1, e},                  // on the boundary
        {1, 1, e * (1 - 1e-12)},    // just outside
        {1, 1, e * (1 + 1e-12)},    // just inside
        {-1, -1, -e + 1e-12},       // just outside the polar boundary
        {1e-12, -1, 1},             // x tiny positive, y negative: rho beyond 1e12
        {-1e-12, 1, 1e-12},         // x tiny negative, y positive, z tiny
        {1, -1e-12, 1e-12},         // y tiny negative
        {1, 1e-12, -1},             // y tiny positive, z negative
        {0, 1, 0},                  // x = 0: the interval is (-inf, 0)
        {1, 0, 0},                  // y = 0: the interval is (1, inf)
        {0, 1, -1},
        {5, 1, 1},                  // large rho
        {-30, 1, 1e-9},             // very negative rho
        {1, 2, 1e-300},
        {700, 1, 1},                // e^{x/y} would overflow
        {-1, 1e-8, -1e-8},
        {3, -1, 0.2},
    };
    for (const v3& v : cases)
        check_exp(v);
}

TEST(OptimConesExponential, IsPositivelyHomogeneousAcrossHugeAndTinyMagnitudes)
{
    const v3 base[] = {{1, 1, 1}, {-1, 2, 0.5}, {3, -1, 0.2}, {0.3, -2, -1}, {2, 0.5, -0.1}};
    for (const v3& b : base) {
        v3 unit = b;
        op::project_exponential(std::span(unit));
        for (const double k : {1e-300, 1e-150, 1e-20, 1e20, 1e150, 1e300}) {
            v3 x{b[0] * k, b[1] * k, b[2] * k};
            op::project_exponential(std::span(x));
            for (std::size_t i = 0; i < 3; ++i) {
                EXPECT_NEAR(x[i] / k, unit[i], 1e-12) << k;
            }
        }
    }
}

TEST(OptimConesExponential, MembershipByDistance)
{
    EXPECT_TRUE(op::in_exponential_cone(std::span<const double, 3>(v3{1, 1, 3}), 0.0));
    EXPECT_FALSE(op::in_exponential_cone(std::span<const double, 3>(v3{1, 1, 2}), 0.1));
    EXPECT_TRUE(op::in_exponential_cone(std::span<const double, 3>(v3{1, 1, 2.7}), 0.02));
    EXPECT_TRUE(op::in_dual_exponential_cone(std::span<const double, 3>(v3{-1, 1, 1}), 0.0));   // e^{-1} <= e
    EXPECT_FALSE(op::in_dual_exponential_cone(std::span<const double, 3>(v3{1, 1, 1}), 0.1));
    EXPECT_TRUE(op::in_dual_exponential_cone(std::span<const double, 3>(v3{0, 1, 1}), 0.0));
}

// ---- Power cone ----

TEST(OptimConesPower, MatchesAnIndependentSolver)
{
    // scipy.optimize.minimize (SLSQP, ftol 1e-15) of |p - v|^2 / 2 subject to x^0.3 y^0.7 >= |z|
    v3 a{1, -0.5, 2};
    op::project_power(std::span(a), 0.3);
    EXPECT_NEAR(a[0], 1.22782308, 1e-6);
    EXPECT_NEAR(a[1], 0.59569204, 1e-6);
    EXPECT_NEAR(a[2], 0.74004137, 1e-6);
    v3 b{0.2, 0.3, 1.0};
    op::project_power(std::span(b), 0.3);
    EXPECT_NEAR(b[0], 0.39125013, 1e-6);
    EXPECT_NEAR(b[1], 0.59395438, 1e-6);
    EXPECT_NEAR(b[2], 0.52403888, 1e-6);
}

TEST(OptimConesPower, MoreauHoldsAcrossAlpha)
{
    st::wyrand g(6);
    for (const double a : {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        auto proj = [a](std::span<double> x) { op::project_power(x.first<3>(), a); };
        auto dual = [a](std::span<double> x) { op::project_dual_power(x.first<3>(), a); };
        auto in_k = [a](const vec& x, double tol) { return pow_member(x, a, tol); };
        auto in_d = [a](const vec& x, double tol) { return pow_dual_member(x, a, tol); };
        for (int t = 0; t < 300; ++t) {
            const vec v = random_vector(g, 3, -3, 3);
            check_moreau(v, proj, in_k, in_d, 1e-9, "pow");
            check_moreau(v, dual, in_d, in_k, 1e-9, "pow dual");
        }
        for (const v3& v : {v3{1, 1, 0}, v3{-1, 2, 0}, v3{-1, -1, 1}, v3{1e-12, -1, 1}, v3{1, 1, 1e-12}, v3{0, 0, 1}, v3{1e200, -1e200, 3e200}, v3{1e-200, 2e-200, -5e-200}}) {
            check_moreau(vec(v.begin(), v.end()), proj, in_k, in_d, 1e-9, "pow hard");
        }
    }
}

TEST(OptimConesPower, AnalyticCasesAndMembership)
{
    v3 in{4, 1, 1.9};   // 4^0.5 1^0.5 = 2 >= 1.9
    op::project_power(std::span(in), 0.5);
    EXPECT_EQ(in, (v3{4, 1, 1.9}));
    v3 polar{-1, -1, 0.5};   // (1/0.5)^0.5 (1/0.5)^0.5 = 2 >= 0.5
    op::project_power(std::span(polar), 0.5);
    EXPECT_EQ(polar, (v3{0, 0, 0}));
    v3 flat{-1, 2, 0};
    op::project_power(std::span(flat), 0.5);
    EXPECT_EQ(flat, (v3{0, 2, 0}));
    EXPECT_TRUE(op::in_power_cone(std::span<const double, 3>(v3{4, 1, 2}), 0.5, 0.0));
    EXPECT_FALSE(op::in_power_cone(std::span<const double, 3>(v3{4, 1, 3}), 0.5, 0.1));
    EXPECT_TRUE(op::in_dual_power_cone(std::span<const double, 3>(v3{1, 1, 2}), 0.5, 0.0));
    EXPECT_FALSE(op::in_dual_power_cone(std::span<const double, 3>(v3{1, 1, 3}), 0.5, 0.1));
}

// ---- Simplex and l1 ball ----

// The sort-based projection (Held, Wolfe and Crowder 1974; Duchi et al. 2008): O(n log n)
vec simplex_reference(vec y, double r)
{
    vec u = y;
    std::sort(u.begin(), u.end(), std::greater<>());
    double s = 0, tau = 0;
    for (std::size_t j = 0; j < u.size(); ++j) {
        s += u[j];
        const double t = (s - r) / static_cast<double>(j + 1);
        if (u[j] - t > 0)
            tau = t;
    }
    for (double& v : y)
        v = std::max(v - tau, 0.0);
    return y;
}

TEST(OptimConesSimplex, CondatAndMichelotMatchTheSortBasedProjection)
{
    st::wyrand g(7);
    for (int t = 0; t < 300; ++t) {
        const std::size_t n = 1 + static_cast<std::size_t>(t % 40);
        const double r = uniform(g, 0.01, 5);
        const vec y = random_vector(g, n, -3, 3);
        const vec ref = simplex_reference(y, r);
        vec a = y, b = y, ws(n);
        ASSERT_TRUE(op::project_simplex(std::span<double>(a), r, std::span<double>(ws)));
        ASSERT_TRUE(op::project_simplex(std::span<double>(b), r));
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(a[i], ref[i], 1e-12);
            EXPECT_NEAR(b[i], ref[i], 1e-12);
        }
        EXPECT_TRUE(op::in_simplex(std::span<const double>(a), r, 1e-12));
    }
}

TEST(OptimConesSimplex, EdgeCases)
{
    vec ws(4);
    vec equal{1, 1, 1, 1};
    ASSERT_TRUE(op::project_simplex(std::span<double>(equal), 2.0, std::span<double>(ws)));
    EXPECT_EQ(equal, (vec{0.5, 0.5, 0.5, 0.5}));
    vec zero_total{3, -1};
    ASSERT_TRUE(op::project_simplex(std::span<double>(zero_total), 0.0, std::span<double>(ws)));
    EXPECT_EQ(zero_total, (vec{0, 0}));
    vec one{-7};
    ASSERT_TRUE(op::project_simplex(std::span<double>(one), 2.0));
    EXPECT_EQ(one, (vec{2}));
    vec descending{4, 3, 2, 1}, ascending{1, 2, 3, 4};   // Condat's worst orderings
    ASSERT_TRUE(op::project_simplex(std::span<double>(descending), 1.0, std::span<double>(ws)));
    ASSERT_TRUE(op::project_simplex(std::span<double>(ascending), 1.0, std::span<double>(ws)));
    EXPECT_EQ(descending, (vec{1, 0, 0, 0}));
    EXPECT_EQ(ascending, (vec{0, 0, 0, 1}));
    vec x{1, 2};
    EXPECT_EQ(op::project_simplex(std::span<double>(x), -1.0).unwrap_err(), op::error::infeasible);
    vec empty;
    EXPECT_EQ(op::project_simplex(std::span<double>(empty), 1.0).unwrap_err(), op::error::infeasible);
    EXPECT_TRUE(op::project_simplex(std::span<double>(empty), 0.0));
    vec tiny(1);
    EXPECT_EQ(op::project_simplex(std::span<double>(x), 1.0, std::span<double>(tiny)).unwrap_err(), op::error::workspace_too_small);
    EXPECT_FALSE(op::in_simplex(std::span<const double>(vec{0.5, 0.6}), 1.0, 0.05));
    EXPECT_FALSE(op::in_simplex(std::span<const double>(vec{-0.5, 1.5}), 1.0, 0.05));
}

TEST(OptimConesL1Ball, MatchesTheSortBasedProjectionOfTheMagnitudes)
{
    st::wyrand g(8);
    for (int t = 0; t < 300; ++t) {
        const std::size_t n = 1 + static_cast<std::size_t>(t % 30);
        const double r = uniform(g, 0.01, 3);
        const vec y = random_vector(g, n, -2, 2);
        vec mag(n);
        for (std::size_t i = 0; i < n; ++i)
            mag[i] = std::abs(y[i]);
        double l1 = std::accumulate(mag.begin(), mag.end(), 0.0);
        vec ref = l1 <= r ? mag : simplex_reference(mag, r);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = std::copysign(ref[i], y[i]);
        vec a = y, b = y, ws(n);
        ASSERT_TRUE(op::project_l1_ball(std::span<double>(a), r, std::span<double>(ws)));
        ASSERT_TRUE(op::project_l1_ball(std::span<double>(b), r));
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(a[i], ref[i], 1e-12);
            EXPECT_NEAR(b[i], ref[i], 1e-12);
        }
        EXPECT_TRUE(op::in_l1_ball(std::span<const double>(a), r, 1e-12));
    }
    vec inside{0.1, -0.2};
    ASSERT_TRUE(op::project_l1_ball(std::span<double>(inside), 1.0));
    EXPECT_EQ(inside, (vec{0.1, -0.2}));
    vec to_zero{1, -2}, ws(2);
    ASSERT_TRUE(op::project_l1_ball(std::span<double>(to_zero), 0.0, std::span<double>(ws)));
    EXPECT_EQ(to_zero, (vec{0, 0}));
    EXPECT_EQ(op::project_l1_ball(std::span<double>(to_zero), -1.0).unwrap_err(), op::error::infeasible);
    EXPECT_EQ(op::project_l1_ball(std::span<double>(to_zero), -1.0, std::span<double>(ws)).unwrap_err(), op::error::infeasible);
    vec small(1);
    EXPECT_EQ(op::project_l1_ball(std::span<double>(to_zero), 1.0, std::span<double>(small)).unwrap_err(), op::error::workspace_too_small);
    EXPECT_FALSE(op::in_l1_ball(std::span<const double>(vec{1, -1}), 1.0, 0.5));
}

TEST(OptimConesLinfBall, ClampsToTheRadius)
{
    vec x{-3, 0.5, 2};
    ASSERT_TRUE(op::project_linf_ball(std::span<double>(x), 1.0));
    EXPECT_EQ(x, (vec{-1, 0.5, 1}));
    EXPECT_TRUE(op::in_linf_ball(std::span<const double>(x), 1.0, 0.0));
    EXPECT_FALSE(op::in_linf_ball(std::span<const double>(vec{1.5}), 1.0, 0.1));
    EXPECT_EQ(op::project_linf_ball(std::span<double>(x), -1.0).unwrap_err(), op::error::infeasible);
}

TEST(OptimConesCappedSimplex, SatisfiesItsKktConditions)
{
    st::wyrand g(9);
    for (int t = 0; t < 300; ++t) {
        const std::size_t n = 1 + static_cast<std::size_t>(t % 25);
        const double u = uniform(g, 0.1, 2);
        const double r = uniform(g, 0, static_cast<double>(n) * u);
        const vec y = random_vector(g, n, -3, 3);
        vec x = y;
        ASSERT_TRUE(op::project_capped_simplex(std::span<double>(x), r, u));
        EXPECT_TRUE(op::in_capped_simplex(std::span<const double>(x), r, u, 1e-11));
        // KKT: x = clamp(y - tau, 0, u) for one tau: y - x is tau on the free elements, and y <= tau at 0, y - u >= tau at u
        double tau = std::numeric_limits<double>::quiet_NaN();
        for (std::size_t i = 0; i < n; ++i)
            if (x[i] > 1e-12 && x[i] < u - 1e-12)
                tau = y[i] - x[i];
        if (tau == tau) {
            for (std::size_t i = 0; i < n; ++i) {
                const double expected = std::clamp(y[i] - tau, 0.0, u);
                EXPECT_NEAR(x[i], expected, 1e-10);
            }
        }
    }
    // A large cap is the simplex
    st::wyrand h(10);
    const vec y = random_vector(h, 12);
    vec x = y;
    ASSERT_TRUE(op::project_capped_simplex(std::span<double>(x), 1.0, 100.0));
    const vec ref = simplex_reference(y, 1.0);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(x[i], ref[i], 1e-12);
    }
    vec bad{1, 2};
    EXPECT_EQ(op::project_capped_simplex(std::span<double>(bad), 3.0, 1.0).unwrap_err(), op::error::infeasible);
    EXPECT_EQ(op::project_capped_simplex(std::span<double>(bad), 1.0, 0.0).unwrap_err(), op::error::infeasible);
    EXPECT_EQ(bad, (vec{1, 2}));
    vec full{0.3, -5};
    ASSERT_TRUE(op::project_capped_simplex(std::span<double>(full), 2.0));
    EXPECT_EQ(full, (vec{1, 1}));
    EXPECT_FALSE(op::in_capped_simplex(std::span<const double>(vec{1.5, 0}), 1.5, 1.0, 0.1));
}

// ---- Products ----

TEST(OptimConesProduct, ProjectsEachBlockOntoItsConeAndItsDual)
{
    const std::array<op::cone_kind, 8> cones{{{op::cone::zero, 2, 0},
                                             {op::cone::free, 1, 0},
                                             {op::cone::nonnegative, 3, 0},
                                             {op::cone::second_order, 3, 0},
                                             {op::cone::rotated_second_order, 3, 0},
                                             {op::cone::psd, 3, 0},
                                             {op::cone::exponential, 3, 0},
                                             {op::cone::power, 3, 0.4}}};
    const std::span<const op::cone_kind> k(cones);
    ASSERT_EQ(op::product_dimension(k), 21u);
    ASSERT_EQ(op::product_workspace_size(k), op::psd_workspace_size(2));
    st::wyrand g(11);
    const vec x0 = random_vector(g, 21);
    vec x = x0, ws(op::product_workspace_size(k));
    ASSERT_TRUE(op::project(k, std::span<double>(x), std::span<double>(ws)));
    // Block by block, by the individual functions
    vec y = x0;
    const std::span<double> s(y);
    op::project_zero(s.subspan(0, 2));
    op::project_nonnegative(s.subspan(3, 3));
    op::project_second_order(s.subspan(6, 3));
    ASSERT_TRUE(op::project_rotated_second_order(s.subspan(9, 3)));
    ASSERT_TRUE(op::project_psd(s.subspan(12, 3), std::span<double>(ws)));
    op::project_exponential(s.subspan(15).first<3>());
    op::project_power(s.subspan(18).first<3>(), 0.4);
    EXPECT_EQ(x, y);
    EXPECT_TRUE(op::in_cone(k, std::span<const double>(x), 1e-12, std::span<double>(ws)));
    EXPECT_FALSE(op::in_cone(k, std::span<const double>(x0), 1e-12, std::span<double>(ws)));
    // Moreau over the product: x0 = P_K(x0) - P_K*(-x0)
    vec m = x0;
    for (double& v : m)
        v = -v;
    ASSERT_TRUE(op::project_dual(k, std::span<double>(m), std::span<double>(ws)));
    EXPECT_TRUE(op::in_cone(std::span<const op::cone_kind>(std::array<op::cone_kind, 8>{{{op::cone::free, 2, 0},
                                                                                          {op::cone::zero, 1, 0},
                                                                                          {op::cone::nonnegative, 3, 0},
                                                                                          {op::cone::second_order, 3, 0},
                                                                                          {op::cone::rotated_second_order, 3, 0},
                                                                                          {op::cone::psd, 3, 0},
                                                                                          {op::cone::dual_exponential, 3, 0},
                                                                                          {op::cone::dual_power, 3, 0.4}}}),
                            std::span<const double>(m), 1e-12, std::span<double>(ws)));
    for (std::size_t i = 0; i < 21; ++i) {
        EXPECT_NEAR(x[i] - m[i], x0[i], 1e-9) << i;
    }
    EXPECT_NEAR(dot(x, m), 0.0, 1e-9);
}

TEST(OptimConesProduct, ReportsSizesThatDoNotAddUp)
{
    vec x(4), ws(op::psd_workspace_size(2));
    const std::array<op::cone_kind, 1> exp4{{{op::cone::exponential, 4, 0}}};
    EXPECT_EQ(op::project(std::span<const op::cone_kind>(exp4), std::span<double>(x)).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_EQ(op::project_dual(std::span<const op::cone_kind>(exp4), std::span<double>(x)).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_FALSE(op::in_cone(std::span<const op::cone_kind>(exp4), std::span<const double>(x), 1.0));
    const std::array<op::cone_kind, 1> soc3{{{op::cone::second_order, 3, 0}}};
    EXPECT_EQ(op::project(std::span<const op::cone_kind>(soc3), std::span<double>(x)).unwrap_err(), op::error::dimension_mismatch);
    EXPECT_FALSE(op::in_cone(std::span<const op::cone_kind>(soc3), std::span<const double>(x), 1.0));
    const std::array<op::cone_kind, 1> psd4{{{op::cone::psd, 4, 0}}};
    EXPECT_EQ(op::project(std::span<const op::cone_kind>(psd4), std::span<double>(x), std::span<double>(ws)).unwrap_err(), op::error::dimension_mismatch);
    const std::array<op::cone_kind, 2> psd3{{{op::cone::psd, 3, 0}, {op::cone::zero, 1, 0}}};
    EXPECT_EQ(op::project(std::span<const op::cone_kind>(psd3), std::span<double>(x)).unwrap_err(), op::error::workspace_too_small);
    vec three(3);
    const std::array<op::cone_kind, 1> bad_alpha{{{op::cone::power, 3, 1.5}}};
    EXPECT_EQ(op::project(std::span<const op::cone_kind>(bad_alpha), std::span<double>(three)).unwrap_err(), op::error::dimension_mismatch);
    const std::array<op::cone_kind, 1> rsoc1{{{op::cone::rotated_second_order, 1, 0}}};
    vec one(1);
    EXPECT_EQ(op::project(std::span<const op::cone_kind>(rsoc1), std::span<double>(one)).unwrap_err(), op::error::dimension_mismatch);
}

// ---- Nesterov-Todd ----

TEST(OptimConesNesterovTodd, OrthantScalingMapsZAndSToTheSameLambda)
{
    const vec s{1, 4, 0.25}, z{4, 1, 9};
    vec w(3);
    ASSERT_TRUE(op::nt_scaling_nonnegative(s, z, std::span<double>(w)));
    vec wz = z, wis = s;
    op::nt_apply_nonnegative(w, std::span<double>(wz));
    op::nt_apply_inverse_nonnegative(w, std::span<double>(wis));
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(wz[i], wis[i], 1e-15);
        EXPECT_NEAR(wz[i], std::sqrt(s[i] * z[i]), 1e-15);
    }
    EXPECT_EQ(op::nt_scaling_nonnegative(std::span<const double>(vec{1, 0, 1}), z, std::span<double>(w)).unwrap_err(), op::error::not_interior);
    EXPECT_EQ(op::nt_scaling_nonnegative(std::span<const double>(vec{1}), z, std::span<double>(w)).unwrap_err(), op::error::dimension_mismatch);
}

// A random point strictly inside the second-order cone of dimension n
vec soc_interior(st::wyrand& g, std::size_t n)
{
    vec x = random_vector(g, n);
    double s = 0;
    for (std::size_t i = 1; i < n; ++i)
        s += x[i] * x[i];
    x[0] = std::sqrt(s) + uniform(g, 0.01, 2);
    return x;
}

TEST(OptimConesNesterovTodd, SecondOrderScalingIsSymmetricPositiveDefiniteAndMapsZAndSToTheSameLambda)
{
    st::wyrand g(12);
    constexpr std::size_t n = 4;
    for (int t = 0; t < 100; ++t) {
        const vec s = soc_interior(g, n), z = soc_interior(g, n);
        vec w(n);
        const auto eta = op::nt_scaling_second_order(s, z, std::span<double>(w));
        ASSERT_TRUE(eta);
        // |w|_J = 1 and w0 >= 1
        EXPECT_NEAR(w[0] * w[0] - (w[1] * w[1] + w[2] * w[2] + w[3] * w[3]), 1.0, 1e-10);
        EXPECT_GE(w[0], 1.0);
        vec wz = z, wis = s;
        op::nt_apply_second_order(w, *eta, std::span<double>(wz));
        op::nt_apply_inverse_second_order(w, *eta, std::span<double>(wis));
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(wz[i], wis[i], 1e-10 * (1 + std::abs(wz[i])));
        }
        // lambda is in the cone
        EXPECT_TRUE(soc_member(wz, 1e-12));
        // W as a matrix, column by column: symmetric, positive definite, W W^-1 = I
        la::fixed_matrix<double, n, n> wm{}, wi{};
        for (std::size_t j = 0; j < n; ++j) {
            vec e(n, 0.0), f(n, 0.0);
            e[j] = f[j] = 1;
            op::nt_apply_second_order(w, *eta, std::span<double>(e));
            op::nt_apply_inverse_second_order(w, *eta, std::span<double>(f));
            for (std::size_t i = 0; i < n; ++i) {
                wm(i, j) = e[i];
                wi(i, j) = f[i];
            }
        }
        const auto prod = wm * wi;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) {
                EXPECT_NEAR(wm(i, j), wm(j, i), 1e-12);
                EXPECT_NEAR(prod(i, j), i == j ? 1.0 : 0.0, 1e-10);
            }
        const auto e = la::symmetric_eigen(wm);
        ASSERT_TRUE(e);
        EXPECT_GT(e->values[0], 0.0);
        // W maps the cone into itself
        const vec c = soc_interior(g, n);
        vec wc = c;
        op::nt_apply_second_order(w, *eta, std::span<double>(wc));
        EXPECT_TRUE(soc_member(wc, 1e-12));
    }
}

TEST(OptimConesNesterovTodd, EqualPointsGiveTheIdentity)
{
    const vec s{2, 0.5, -1};
    vec w(3);
    const auto eta = op::nt_scaling_second_order(s, s, std::span<double>(w));
    ASSERT_TRUE(eta);
    EXPECT_NEAR(*eta, 1.0, 1e-15);
    EXPECT_NEAR(w[0], 1.0, 1e-15);
    EXPECT_NEAR(w[1], 0.0, 1e-15);
    vec x{3, 1, 2};
    op::nt_apply_second_order(w, *eta, std::span<double>(x));
    EXPECT_NEAR(x[0], 3, 1e-14);
    EXPECT_NEAR(x[2], 2, 1e-14);
    EXPECT_EQ(op::nt_scaling_second_order(std::span<const double>(vec{1, 1, 0}), s, std::span<double>(w)).unwrap_err(), op::error::not_interior);
    EXPECT_EQ(op::nt_scaling_second_order(std::span<const double>(vec{1, 0}), s, std::span<double>(w)).unwrap_err(), op::error::dimension_mismatch);
}

// ---- Constant evaluation and noexcept ----

constexpr bool compile_time()
{
    std::array<double, 3> soc{0, 3, 4};
    op::project_second_order(std::span<double>(soc));
    std::array<double, 3> ex{1, 1, 1};
    op::project_exponential(std::span(ex));
    std::array<double, 3> pw{1, -0.5, 2};
    op::project_power(std::span(pw), 0.3);
    std::array<double, 4> sx{0.5, 0.2, -1, 0.9}, ws{};
    const bool simplex_ok = op::project_simplex(std::span<double>(sx), 1.0, std::span<double>(ws)).is_ok();
    std::array<double, 3> psd{1, 2 * 1.4142135623730951, 1};   // [[1, 2], [2, 1]]: eigenvalues 3 and -1
    const bool psd_ok = op::project_psd_fixed<2>(std::span<double, 3>(psd)).is_ok();
    std::array<double, 3> psd_rt{1, 2 * 1.4142135623730951, 1};
    std::array<double, 10> pws{};
    const bool psd_rt_ok = op::project_psd(std::span<double>(psd_rt), std::span<double>(pws)).is_ok();
    auto near = [](double a, double b, double tol) { return a - b <= tol && b - a <= tol; };
    return soc[0] == 2.5 && near(ex[0], 0.42630617, 1e-6) && near(pw[2], 0.74004137, 1e-6) && simplex_ok && near(sx[0] + sx[1] + sx[3], 1.0, 1e-15) && psd_ok
           && near(psd[0], 1.5, 1e-12) && near(psd[2], 1.5, 1e-12) && psd_rt_ok && near(psd_rt[1], psd[1], 1e-12);
}
static_assert(compile_time());

static_assert(noexcept(op::project_second_order(std::declval<std::span<double>>())));
static_assert(noexcept(op::project_exponential(std::declval<std::span<double, 3>>())));
static_assert(noexcept(op::project_power(std::declval<std::span<double, 3>>(), 0.5)));
static_assert(noexcept(op::project_psd(std::declval<std::span<double>>(), std::declval<std::span<double>>())));
static_assert(noexcept(op::project(std::declval<std::span<const op::cone_kind>>(), std::declval<std::span<double>>())));
static_assert(noexcept(op::project_simplex(std::declval<std::span<float>>(), 1.0f)));
static_assert(op::dual_of(op::cone::exponential) == op::cone::dual_exponential);

TEST(OptimConesFloat, ProjectionsWorkInFloat)
{
    std::array<float, 3> ex{1, 1, 1};
    op::project_exponential(std::span(ex));
    EXPECT_NEAR(ex[0], 0.42630617f, 1e-5f);
    std::array<float, 3> pw{1, -0.5f, 2};
    op::project_power(std::span(pw), 0.3f);
    EXPECT_NEAR(pw[2], 0.74004137f, 1e-5f);
    std::array<float, 3> big{1e30f, 1e30f, 1e30f};
    op::project_exponential(std::span(big));
    EXPECT_NEAR(big[0] / 1e30f, 0.42630617f, 1e-5f);
}

} // namespace
