//
//  muon_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/muon.hpp>: the iteration against the same arithmetic as plain loops, its singular
//  values in Muon's band, a tall matrix as the transpose of a wide one, the step's momentum, Nesterov and scaling,
//  a zero gradient, the working-space size, compile-time use, every step on the backend, and the bits of the loops
//  0.12.0 ran on the host.
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
#include <string>
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


// muon_step as 0.12.0 wrote it, its elementwise steps loops on the host: the backend's form must give its bits
template <typename T>
void step_0_12(std::span<T> w, std::span<const T> g, std::span<T> momentum, std::size_t rows, std::size_t cols,
               const ml::muon_config<T>& config)
{
    using right = la::layout_right;
    const std::size_t n = rows * cols;
    std::vector<T> update(n), out(n);
    for (std::size_t i = 0; i < n; ++i) {
        momentum[i] = config.momentum * momentum[i] + g[i];
        update[i] = config.nesterov ? g[i] + config.momentum * momentum[i] : momentum[i];
    }
    const bool tall = rows > cols;
    const std::size_t r = tall ? cols : rows, s = tall ? rows : cols;
    std::vector<T> x(r * s), x2(r * s), a(r * r), bm(r * r);
    T norm2 = T(0);
    for (const T v : update)
        norm2 += v * v;
    const T scale = T(1) / (snicholls::math::sqrt(norm2) + T(1e-7));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            (tall ? x[j * s + i] : x[i * s + j]) = update[i * cols + j] * scale;
    using ns = ml::muon_newton_schulz<T>;
    const la::matrix_view<T, right> x2v(x2.data(), {r, s}), av(a.data(), {r, r}), bv(bm.data(), {r, r});
    const la::matrix_view<const T, right> xc(x.data(), {r, s}), ac(a.data(), {r, r}), bc(bm.data(), {r, r});
    for (int step = 0; step < ns::steps; ++step) {
        ASSERT_TRUE(la::gemm(la::reference, T(1), xc, la::transposed(xc), T(0), av).is_ok());
        for (std::size_t i = 0; i < r * r; ++i)
            bm[i] = ns::b * a[i];
        ASSERT_TRUE(la::gemm(la::reference, ns::c, ac, ac, T(1), bv).is_ok());
        for (std::size_t i = 0; i < r * s; ++i)
            x2[i] = ns::a * x[i];
        ASSERT_TRUE(la::gemm(la::reference, T(1), bc, xc, T(1), x2v).is_ok());
        std::copy(x2.begin(), x2.end(), x.begin());
    }
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            out[i * cols + j] = tall ? x[j * s + i] : x[i * s + j];
    const T step = config.lr * snicholls::math::sqrt(std::max(T(1), T(rows) / T(cols)));
    for (std::size_t i = 0; i < n; ++i)
        w[i] -= step * out[i];
}

template <typename T>
void expect_the_bits_of_0_12()
{
    for (auto [rows, cols] : {std::pair{5u, 9u}, std::pair{9u, 5u}, std::pair{7u, 7u}, std::pair{1u, 6u}, std::pair{6u, 1u}})
        for (const bool nesterov : {true, false}) {
            const std::size_t n = rows * cols;
            std::vector<T> w(n), m(n), w0(n), m0(n), work(ml::muon_work_size(rows, cols));
            const auto init = noise(n, 3 + rows);
            for (std::size_t i = 0; i < n; ++i)
                w[i] = w0[i] = T(init[i]);
            const ml::muon_config<T> config{.lr = T(0.02), .momentum = T(0.95), .nesterov = nesterov};
            for (std::uint64_t t = 0; t < 4; ++t) {   // momentum carried over steps
                const auto gd = noise(n, 100 + t + cols);
                std::vector<T> g(n);
                for (std::size_t i = 0; i < n; ++i)
                    g[i] = T(gd[i]);
                ml::muon_step(la::reference, std::span<T>(w), std::span<const T>(g), std::span<T>(m), rows, cols, config, std::span<T>(work));
                step_0_12(std::span<T>(w0), std::span<const T>(g), std::span<T>(m0), rows, cols, config);
            }
            EXPECT_EQ(w, w0) << rows << "x" << cols << (nesterov ? " nesterov" : "");
            EXPECT_EQ(m, m0) << rows << "x" << cols << (nesterov ? " nesterov" : "");
        }
}

TEST(MlMuon, TheStepOnTheBackendGivesTheBitsOfTheHostLoops)
{
    expect_the_bits_of_0_12<float>();
    expect_the_bits_of_0_12<double>();
}

} // namespace

// A backend that runs every elementwise step itself, as a device library does (found by argument-dependent lookup),
// counting the calls and touching the data only through them
namespace muon_test_device {

struct backend : snicholls::linalg::reference_backend {
    mutable int calls = 0, products = 0;
    template <snicholls::linalg::scalar T>
    void gemm(T alpha, snicholls::linalg::matrix_view<const T> a, snicholls::linalg::matrix_view<const T> b, T beta,
              snicholls::linalg::matrix_view<T> c) const
    {
        ++products;
        snicholls::linalg::reference_backend::gemm(alpha, a, b, beta, c);
    }
};

inline void copy_rows(const backend& b, std::span<const double> from, std::span<double> to)
{
    ++b.calls;
    snicholls::ml::copy_rows(snicholls::linalg::reference, from, to);
}
inline void scale_rows(const backend& b, std::span<double> x, double alpha)
{
    ++b.calls;
    snicholls::ml::scale_rows(snicholls::linalg::reference, x, alpha);
}
inline void add_rows(const backend& b, std::span<const double> a, std::span<double> x)
{
    ++b.calls;
    snicholls::ml::add_rows(snicholls::linalg::reference, a, x);
}
template <typename E>
double sum_squares(const backend& b, std::span<E> x)
{
    ++b.calls;
    return snicholls::ml::sum_squares(snicholls::linalg::reference, x);
}

} // namespace muon_test_device

static_assert(snicholls::linalg::blas_backend<muon_test_device::backend, double>);

namespace {

TEST(MlMuon, EveryStepIsACallOnTheBackend)
{
    const std::size_t rows = 4, cols = 6, n = rows * cols;
    std::vector<double> w(n, 0.5), m(n), w0(n, 0.5), m0(n), work(ml::muon_work_size(rows, cols));
    const auto g = noise(n, 11);
    const muon_test_device::backend device;
    ml::muon_step(device, std::span<double>(w), std::span<const double>(g), std::span<double>(m), rows, cols, ml::muon_config<double>{},
                  std::span<double>(work));
    ml::muon_step(la::reference, std::span<double>(w0), std::span<const double>(g), std::span<double>(m0), rows, cols,
                  ml::muon_config<double>{}, std::span<double>(work));
    // The momentum (3), Nesterov (2), the norm, X = g / |g| (2), two copies per iteration (10), the result (1) and w (2)
    EXPECT_EQ(device.calls, 3 + 2 + 1 + 2 + 2 * ml::muon_newton_schulz<double>::steps + 1 + 2);
    EXPECT_EQ(device.products, 3 * ml::muon_newton_schulz<double>::steps);
    EXPECT_EQ(w, w0);
    EXPECT_EQ(m, m0);
}

} // namespace
