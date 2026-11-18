//
//  softmax_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/softmax.hpp>: softmax, log_softmax and logsumexp against long double,
//  logits that overflow exp, logits near 1e4 against the shifted closed forms, the gradients against central differences, the fused cross-entropy in
//  place, online_softmax split and merged in every order, the edge cases, and compile-time use.
//

#include <snicholls/core/ml/softmax.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

constexpr double inf = std::numeric_limits<double>::infinity();

constexpr bool compile_time()
{
    std::array<double, 3> x{1, 2, 3}, p{};
    ml::softmax(x, p);
    const double lse = ml::logsumexp(x);
    ml::online_softmax<double> o;
    for (double v : x)
        o.add(v);
    return m::near(p[0] + p[1] + p[2], 1.0, 1e-15) && m::near(lse, 3.40760596444438, 1e-13) && m::near(o.log_sum_exp(), lse, 1e-15)
        && m::near(ml::cross_entropy(x, 2), lse - 3, 1e-15);
}
static_assert(compile_time());
static_assert(std::is_trivially_copyable_v<ml::online_softmax<float>> && sizeof(ml::online_softmax<float>) == 8);
static_assert(ml::bits_per_byte(0.0, 0) == inf && m::near(ml::bits_per_byte(0.6931471805599453 * 10, 10), 1.0, 1e-15));

std::vector<double> row(std::size_t n, double spread)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = spread * std::sin(1.3 * static_cast<double>(i) + 0.2);
    return v;
}

TEST(MlSoftmax, AgreesWithLongDouble)
{
    const auto x = row(50, 4.0);
    std::vector<double> p(x.size()), lp(x.size());
    ml::softmax(x, p);
    ml::log_softmax(x, lp);
    long double s = 0;
    for (const double v : x)
        s += std::exp(static_cast<long double>(v));
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(p[i], static_cast<double>(std::exp(static_cast<long double>(x[i])) / s), 1e-16);
        EXPECT_NEAR(lp[i], static_cast<double>(x[i] - std::log(s)), 1e-14);
    }
    EXPECT_NEAR(ml::logsumexp(x), static_cast<double>(std::log(s)), 1e-14);
}

TEST(MlSoftmax, LogitsThatOverflowExpStayFinite)
{
    const std::vector<double> x{1000, 999, 1001, -1000};
    std::vector<double> p(4), lp(4);
    ml::softmax(x, p);
    ml::log_softmax(x, lp);
    EXPECT_NEAR(p[2], 1 / (1 + std::exp(-1.0) + std::exp(-2.0)), 1e-15);
    EXPECT_EQ(p[3], 0.0);
    EXPECT_NEAR(lp[3], -2001 - std::log(1 + std::exp(-1.0) + std::exp(-2.0)), 1e-12);   // log(softmax) would be -inf
    EXPECT_NEAR(ml::logsumexp(x), 1001 + std::log(1 + std::exp(-1.0) + std::exp(-2.0)), 1e-12);
    EXPECT_NEAR(ml::cross_entropy(x, 3), 2001 + std::log(1 + std::exp(-1.0) + std::exp(-2.0)), 1e-12);   // large, finite
}

// Logits near 1e4 with differences of order 1, as a language model's are late in training: each result is a number of
// order 1, so it must not carry the rounding of 1e4. Before 0.14.0 x_i - logsumexp lost about eps 1e4 / 2 (3000 units
// in the last place in float, 2600 in double).
template <typename T>
void expect_large_logits_keep_their_digits()
{
    const T base = T(10000);
    const std::array<T, 4> x{base, base, base - T(1), base - T(3)};
    const double s = 2 + std::exp(-1.0) + std::exp(-3.0);
    const std::array<double, 4> shifted{0, 0, -1, -3};
    const double eps = std::numeric_limits<T>::epsilon();
    std::array<T, 4> lp{}, g{};
    ml::log_softmax(x, lp);
    const T loss = ml::cross_entropy_with_gradient(x, 2, g, T(1));
    EXPECT_LE(std::abs(ml::cross_entropy(x, 0) - std::log(s)) / std::log(s), 4 * eps);
    EXPECT_LE(std::abs(loss - (1 + std::log(s))) / (1 + std::log(s)), 4 * eps);
    for (std::size_t i = 0; i < 4; ++i) {
        const double want_lp = shifted[i] - std::log(s);
        const double want_g = std::exp(shifted[i]) / s - (i == 2 ? 1.0 : 0.0);
        EXPECT_LE(std::abs(lp[i] - want_lp) / std::abs(want_lp), 4 * eps) << i;
        EXPECT_LE(std::abs(g[i] - want_g) / std::abs(want_g), 4 * eps) << i;
    }
}

TEST(MlSoftmax, LargeLogitsKeepTheDigitsOfTheLoss)
{
    expect_large_logits_keep_their_digits<float>();
    expect_large_logits_keep_their_digits<double>();
    // The gradient's softmax is softmax's, to the bit
    const std::vector<float> x{9999.5f, 10000.25f, 9998.0f};
    std::vector<float> p(3), g(3);
    ml::softmax(x, p);
    (void)ml::cross_entropy_with_gradient(x, 0, g, 1.0f);
    EXPECT_EQ(g[0], p[0] - 1.0f);
    EXPECT_EQ(g[1], p[1]);
    EXPECT_EQ(g[2], p[2]);
}

TEST(MlSoftmax, EdgeCases)
{
    const std::vector<double> empty;
    EXPECT_EQ(ml::logsumexp(empty), -inf);
    const std::vector<double> one{3.5};
    std::vector<double> p(1);
    ml::softmax(one, p);
    EXPECT_EQ(p[0], 1.0);
    EXPECT_EQ(ml::cross_entropy(one, 0), 0.0);
    const std::vector<double> masked{-inf, 2.0, -inf};   // a causal mask
    std::vector<double> q(3);
    ml::softmax(masked, q);
    EXPECT_EQ(q[0], 0.0);
    EXPECT_EQ(q[1], 1.0);
    const std::vector<double> all_masked{-inf, -inf};
    EXPECT_EQ(ml::logsumexp(all_masked), -inf);
    std::vector<double> r(2);
    ml::softmax(all_masked, r);
    EXPECT_TRUE(std::isnan(r[0]));   // as in PyTorch
    const std::vector<double> with_nan{1.0, std::nan(""), 2.0};
    EXPECT_TRUE(std::isnan(ml::logsumexp(with_nan)));
    ml::softmax(with_nan, q);
    EXPECT_TRUE(std::isnan(q[0]) && std::isnan(q[2]));
}

TEST(MlSoftmax, InPlaceGivesTheSameResult)
{
    auto x = row(17, 3.0);
    std::vector<double> p(x.size());
    ml::softmax(x, p);
    ml::softmax(x, x);
    EXPECT_EQ(x, p);
    auto y = row(17, 3.0);
    std::vector<double> lp(y.size());
    ml::log_softmax(y, lp);
    ml::log_softmax(y, y);
    EXPECT_EQ(y, lp);
}

// A central difference of the loss f(z) = sum_i w_i softmax(z)_i, against softmax_backward
TEST(MlSoftmax, BackwardMatchesACentralDifference)
{
    const auto z = row(9, 2.0);
    const std::vector<double> w{0.3, -1.2, 0.7, 2.0, -0.4, 0.0, 1.1, -0.9, 0.5};
    std::vector<double> p(z.size()), g(z.size());
    ml::softmax(z, p);
    ml::softmax_backward(p, w, g);
    auto f = [&](std::vector<double> v) {
        std::vector<double> q(v.size());
        ml::softmax(v, q);
        double s = 0;
        for (std::size_t i = 0; i < q.size(); ++i)
            s += w[i] * q[i];
        return s;
    };
    for (std::size_t k = 0; k < z.size(); ++k) {
        auto up = z, down = z;
        up[k] += 1e-6;
        down[k] -= 1e-6;
        EXPECT_NEAR(g[k], (f(up) - f(down)) / 2e-6, 1e-8) << k;
    }
}

TEST(MlSoftmax, FusedCrossEntropyGradientMatchesACentralDifferenceAndWorksInPlace)
{
    const auto z = row(12, 3.0);
    std::vector<double> g(z.size());
    const double loss = ml::cross_entropy_with_gradient(z, 5, g, 0.25);
    EXPECT_DOUBLE_EQ(loss, ml::cross_entropy(z, 5));
    for (std::size_t k = 0; k < z.size(); ++k) {
        auto up = z, down = z;
        up[k] += 1e-6;
        down[k] -= 1e-6;
        EXPECT_NEAR(g[k], 0.25 * (ml::cross_entropy(up, 5) - ml::cross_entropy(down, 5)) / 2e-6, 1e-8) << k;
    }
    double sum = 0;
    for (const double v : g)
        sum += v;
    EXPECT_NEAR(sum, 0.0, 1e-15);   // softmax sums to one, the one-hot to one
    auto in_place = z;
    EXPECT_DOUBLE_EQ(ml::cross_entropy_with_gradient(in_place, 5, in_place, 0.25), loss);
    EXPECT_EQ(in_place, g);
}

TEST(MlOnlineSoftmax, SplitAndMergedInAnyOrderMatchesOnePass)
{
    const auto x = row(1000, 30.0);
    const double want = ml::logsumexp(x);
    ml::online_softmax<double> whole;
    for (const double v : x)
        whole.add(v);
    EXPECT_NEAR(whole.log_sum_exp(), want, 1e-12);
    // Four workers on interleaved blocks, merged forwards and backwards
    std::array<ml::online_softmax<double>, 4> part{};
    for (std::size_t i = 0; i < x.size(); ++i)
        part[(i / 37) % 4].add(x[i]);
    ml::online_softmax<double> fwd, bwd, tree = part[0], right = part[2];
    for (std::size_t k = 0; k < 4; ++k) {
        fwd.merge(part[k]);
        bwd.merge(part[3 - k]);
    }
    tree.merge(part[1]);
    right.merge(part[3]);
    tree.merge(right);
    EXPECT_NEAR(fwd.log_sum_exp(), want, 1e-12);
    EXPECT_NEAR(bwd.log_sum_exp(), want, 1e-12);
    EXPECT_NEAR(tree.log_sum_exp(), want, 1e-12);
    EXPECT_NEAR(whole.probability(x[3]), std::exp(x[3] - want), 1e-15);
}

TEST(MlOnlineSoftmax, EmptyPartialsAndMaskedValuesAddNothing)
{
    ml::online_softmax<double> a, empty;
    EXPECT_EQ(a.log_sum_exp(), -inf);
    a.add(-inf);
    EXPECT_EQ(a.log_sum_exp(), -inf);
    a.add(1.0);
    a.merge(empty);
    EXPECT_EQ(a.log_sum_exp(), 1.0);
    empty.merge(a);
    EXPECT_EQ(empty.log_sum_exp(), 1.0);
    ml::online_softmax<float> big;
    big.add(1e30f);
    big.add(-1e30f);
    big.add(1e30f);
    EXPECT_NEAR(big.log_sum_exp(), 1e30f + std::log(2.0f), 1e24f);   // no overflow
}

TEST(MlSoftmax, BitsPerByte)
{
    EXPECT_NEAR(ml::bits_per_byte(1000 * std::log(2.0), 500), 2.0, 1e-15);
    EXPECT_EQ(ml::bits_per_byte(1.0, 0), inf);
}

TEST(MlSoftmax, FloatRowsComputeInFloat)
{
    const std::array<float, 3> x{0.5f, -1.0f, 2.0f};
    std::array<float, 3> p{};
    ml::softmax(x, p);
    static_assert(std::is_same_v<decltype(ml::logsumexp(x)), float>);
    EXPECT_NEAR(p[0] + p[1] + p[2], 1.0f, 1e-6f);
    EXPECT_NEAR(ml::logsumexp(x), static_cast<float>(std::log(std::exp(0.5) + std::exp(-1.0) + std::exp(2.0))), 1e-6f);
}

TEST(MlSoftmax, LogSoftmaxBackwardMatchesCentralDifferencesAndSoftmaxBackward)
{
    // L = c . log_softmax(x)
    std::vector<double> x{0.2, -1.3, 2.1, 0.7, -0.4};
    const std::vector<double> c{0.5, -1.0, 2.0, 0.25, 1.5};
    auto loss = [&] {
        std::vector<double> lp(x.size());
        ml::log_softmax(x, lp);
        double l = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            l += c[i] * lp[i];
        return l;
    };
    std::vector<double> lp(x.size()), d(x.size());
    ml::log_softmax(x, lp);
    ml::log_softmax_backward(lp, c, d);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double keep = x[i], e = 1e-6;
        x[i] = keep + e;
        const double hi = loss();
        x[i] = keep - e;
        const double lo = loss();
        x[i] = keep;
        EXPECT_NEAR(d[i], (hi - lo) / (2 * e), 1e-8) << i;
    }
    // Through softmax: d/dp of c . log p is c / p, and softmax_backward of that is the same gradient
    std::vector<double> p(x.size()), cp(x.size()), via(x.size());
    ml::softmax(x, p);
    for (std::size_t i = 0; i < x.size(); ++i)
        cp[i] = c[i] / p[i];
    ml::softmax_backward(p, cp, via);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(d[i], via[i], 1e-12);
    // In place
    std::vector<double> inplace = c;
    ml::log_softmax_backward(lp, inplace, inplace);
    EXPECT_EQ(inplace, d);
}

constexpr bool log_softmax_backward_compile_time()
{
    // Two equal logits: log p = -ln 2 each; d = (1, 0) gives (1/2, -1/2)
    const std::array<double, 2> lp{-snicholls::math::log(2.0), -snicholls::math::log(2.0)};
    std::array<double, 2> g{};
    ml::log_softmax_backward(lp, std::array<double, 2>{1, 0}, g);
    return snicholls::math::near(g[0], 0.5, 1e-15) && snicholls::math::near(g[1], -0.5, 1e-15);
}
static_assert(log_softmax_backward_compile_time());

// The fused classifier over a sequence on a backend: each row is cross_entropy_with_gradient's, to the bit
TEST(MlSoftmax, CrossEntropyRowsIsTheFusedClassifierRowByRow)
{
    constexpr std::size_t n = 3, v = 5;
    std::vector<double> logits{0.1, -2, 3, 0.5, 1, 4, 4, -1, 0, 2, -0.3, 0.2, 0.7, 1.1, -5}, expect = logits, losses(n), want(n);
    const std::vector<std::uint32_t> targets{2, 0, 4};
    ml::cross_entropy_rows(snicholls::linalg::reference, std::span<double>(logits), targets, std::span<double>(losses), 0.25);
    for (std::size_t t = 0; t < n; ++t) {
        const auto row = std::span<double>(expect).subspan(t * v, v);
        want[t] = ml::cross_entropy_with_gradient(row, targets[t], row, 0.25);
    }
    EXPECT_EQ(losses, want);
    EXPECT_EQ(logits, expect);
}

// Rows wider than the classes (a padded vocabulary): the classes as an unpadded row's, to the bit; the padding 0
TEST(MlSoftmax, CrossEntropyRowsOverTheFirstClassesZeroesThePadding)
{
    constexpr std::size_t n = 3, classes = 5, v = 8;
    std::vector<double> narrow{0.1, -2, 3, 0.5, 1, 4, 4, -1, 0, 2, -0.3, 0.2, 0.7, 1.1, -5}, wide(n * v, 9.0), losses(n), want(n);
    for (std::size_t t = 0; t < n; ++t)
        std::copy_n(narrow.begin() + std::ptrdiff_t(t * classes), classes, wide.begin() + std::ptrdiff_t(t * v));
    const std::vector<std::uint32_t> targets{2, 0, 4};
    ml::cross_entropy_rows(snicholls::linalg::reference, std::span<double>(narrow), targets, std::span<double>(want), 0.25);
    ml::cross_entropy_rows(snicholls::linalg::reference, std::span<double>(wide), targets, std::span<double>(losses), 0.25, classes);
    EXPECT_EQ(losses, want);
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t c = 0; c < v; ++c)
            EXPECT_EQ(wide[t * v + c], c < classes ? narrow[t * classes + c] : 0.0) << t << " " << c;
}

} // namespace
