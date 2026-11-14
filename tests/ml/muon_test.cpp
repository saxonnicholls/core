//
//  muon_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/muon.hpp>: the iteration against the same arithmetic as plain loops, its singular
//  values in Muon's band, a tall matrix as the transpose of a wide one, the step's momentum, Nesterov and scaling,
//  a zero gradient, the working-space size, and compile-time use.
//

#include <snicholls/core/ml/muon.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace la = snicholls::linalg;

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.61 * double(i) + b) + 0.1 * double(i % 3);
    return v;
}

// Full rank: a fixed pseudo-random stream (wave() above is rank 2 or so as a matrix, and Muon keeps a zero
// singular value at zero)
std::vector<double> noise(std::size_t n, std::uint64_t seed)
{
    std::vector<double> v(n);
    std::uint64_t x = seed;
    for (double& e : v) {
        x ^= x << 13, x ^= x >> 7, x ^= x << 17;
        e = double(x >> 11) * 0x1.0p-53 * 2 - 1;
    }
    return v;
}

static_assert(ml::muon_work_size(3, 5) == 4 * 15 + 2 * 9 && ml::muon_work_size(5, 3) == ml::muon_work_size(3, 5));

constexpr bool compile_time()
{
    // A 1 x 2 matrix: the iteration scales it to a unit row, near norm 1
    const std::array<double, 2> g{3, 4};
    std::array<double, 2> out{};
    std::array<double, ml::muon_work_size(1, 2)> work{};
    ml::muon_orthogonalise(la::reference, std::span<const double>(g), 1, 2, std::span<double>(out), std::span<double>(work));
    const double norm = snicholls::math::sqrt(out[0] * out[0] + out[1] * out[1]);
    return norm > 0.6 && norm < 1.25 && snicholls::math::near(out[1] / out[0], 4.0 / 3.0, 1e-12);
}
static_assert(compile_time());

// The iteration by hand: X = G / |G|, five times X = a X + (b A + c A^2) X with A = X X^T, for rows <= cols
std::vector<double> by_hand(const std::vector<double>& g, std::size_t r, std::size_t s)
{
    double n2 = 0;
    for (double v : g)
        n2 += v * v;
    std::vector<double> x(g);
    for (double& v : x)
        v /= std::sqrt(n2) + 1e-7;
    using ns = ml::muon_newton_schulz<double>;
    for (int step = 0; step < ns::steps; ++step) {
        std::vector<double> a(r * r), b(r * r), nx(r * s);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < r; ++j)
                for (std::size_t k = 0; k < s; ++k)
                    a[i * r + j] += x[i * s + k] * x[j * s + k];
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < r; ++j) {
                double aa = 0;
                for (std::size_t k = 0; k < r; ++k)
                    aa += a[i * r + k] * a[k * r + j];
                b[i * r + j] = ns::b * a[i * r + j] + ns::c * aa;
            }
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < s; ++j) {
                double bx = 0;
                for (std::size_t k = 0; k < r; ++k)
                    bx += b[i * r + k] * x[k * s + j];
                nx[i * s + j] = ns::a * x[i * s + j] + bx;
            }
        x = nx;
    }
    return x;
}

std::vector<double> orthogonalise(const std::vector<double>& g, std::size_t rows, std::size_t cols)
{
    std::vector<double> out(g.size()), work(ml::muon_work_size(rows, cols));
    ml::muon_orthogonalise(la::reference, std::span<const double>(g), rows, cols, std::span<double>(out), std::span<double>(work));
    return out;
}

TEST(MlMuon, OrthogonaliseIsTheNewtonSchulzIteration)
{
    const auto g = wave(3 * 7, 1.0, 0.2);
    const auto got = orthogonalise(g, 3, 7), want = by_hand(g, 3, 7);
    for (std::size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], want[i], 1e-12) << i;
}

TEST(MlMuon, EverySingularValueLandsInMuonsBand)
{
    for (auto [rows, cols] : {std::pair{4u, 9u}, std::pair{9u, 4u}, std::pair{6u, 6u}}) {
        auto u = orthogonalise(noise(rows * cols, 7 + rows), rows, cols);
        // svd wants rows >= cols: take the transpose of a wide result
        const std::size_t m = std::max(rows, cols), n = std::min(rows, cols);
        std::vector<double> a(m * n), s(n), v(n * n);
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
                (rows >= cols ? a[i * n + j] : a[j * n + i]) = u[i * cols + j];
        ASSERT_TRUE(la::svd(la::reference, la::matrix_view<double, la::layout_right>(a.data(), {m, n}), std::span<double>(s),
                            la::matrix_view<double, la::layout_right>(v.data(), {n, n}))
                        .is_ok());
        for (double sv : s) {
            EXPECT_GT(sv, 0.6) << rows << "x" << cols;
            EXPECT_LT(sv, 1.25) << rows << "x" << cols;
        }
    }
}

TEST(MlMuon, ATallMatrixIsTheTransposeOfAWideOne)
{
    const std::size_t rows = 8, cols = 3;
    const auto g = wave(rows * cols, 1.0, 0.4);
    std::vector<double> gt(g.size());
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            gt[j * rows + i] = g[i * cols + j];
    const auto tall = orthogonalise(g, rows, cols), wide = orthogonalise(gt, cols, rows);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            EXPECT_EQ(tall[i * cols + j], wide[j * rows + i]);
}

TEST(MlMuon, AStepUpdatesTheMomentumAndMovesTheWeightsByTheScaledOrthogonalisedUpdate)
{
    const std::size_t rows = 6, cols = 3;
    auto w = wave(rows * cols, 0.5, 0.0);
    const auto w0 = w, g = wave(rows * cols, 1.0, 1.0);
    std::vector<double> momentum = wave(rows * cols, 0.2, 2.0), work(ml::muon_work_size(rows, cols));
    const auto m0 = momentum;
    const ml::muon_config<double> config{.lr = 0.05, .momentum = 0.9, .nesterov = true};
    ml::muon_step(la::reference, std::span<double>(w), std::span<const double>(g), std::span<double>(momentum), rows, cols,
                  config, std::span<double>(work));
    std::vector<double> m1(g.size()), update(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        m1[i] = 0.9 * m0[i] + g[i];
        update[i] = g[i] + 0.9 * m1[i];
    }
    const auto x = orthogonalise(update, rows, cols);
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_EQ(momentum[i], m1[i]);
        EXPECT_NEAR(w[i], w0[i] - 0.05 * std::sqrt(2.0) * x[i], 1e-15);   // sqrt(max(1, 6 / 3))
    }
    // Without Nesterov the momentum itself is orthogonalised; a wide matrix is not scaled up
    auto w2 = w0;
    std::vector<double> mom2(g.size());
    ml::muon_step(la::reference, std::span<double>(w2), std::span<const double>(g), std::span<double>(mom2), cols, rows,
                  ml::muon_config<double>{.lr = 0.05, .momentum = 0.9, .nesterov = false}, std::span<double>(work));
    const auto x2 = orthogonalise(g, cols, rows);
    for (std::size_t i = 0; i < g.size(); ++i)
        EXPECT_NEAR(w2[i], w0[i] - 0.05 * x2[i], 1e-15);
}

TEST(MlMuon, ARankDeficientMatrixKeepsItsZeroSingularValues)
{
    // An outer product u v^T has rank 1: the result is rank 1 too, every row a multiple of v
    const std::vector<double> u{1, -2, 0.5}, v{3, 1, -1, 2};
    std::vector<double> g(12);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            g[i * 4 + j] = u[i] * v[j];
    const auto x = orthogonalise(g, 3, 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_NEAR(x[i * 4 + j] * v[0], x[i * 4] * v[j], 1e-12);
}

TEST(MlMuon, AZeroGradientIsAZeroUpdate)
{
    std::vector<double> w{1, 2, 3, 4, 5, 6}, g(6), m(6), work(ml::muon_work_size(2, 3));
    const auto w0 = w;
    ml::muon_step(la::reference, std::span<double>(w), std::span<const double>(g), std::span<double>(m), 2, 3,
                  ml::muon_config<double>{}, std::span<double>(work));
    EXPECT_EQ(w, w0);
}

} // namespace
