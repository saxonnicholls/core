//
//  tensor_ops_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/tensor_ops.hpp>: the timestep embedding against ggml's loop to the bit, with an odd
//  width; arange's count and values; cumsum against the running sum, in place, and its backward by central
//  differences; argsort's stable order both ways and the top-k mask with ties; every triangle of ggml_tri; the
//  triangular solve against ggml's loop to the bit and its backward by central differences; and compile-time use.
//

#include <snicholls/core/ml/tensor_ops.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr bool compile_time()
{
    std::array<double, 4> c{};
    ml::cumsum(std::array<double, 4>{1, 2, 3, 4}, c);
    std::array<double, 3> r{};
    ml::arange(0.5, 2.0, r);
    std::array<std::size_t, 5> idx{};
    ml::argsort(std::array<double, 5>{3, 1, 2, 1, 0}, idx, ml::sort_order::ascending);
    std::array<double, 4> t{};
    ml::tri(ml::tri_kind::lower_diag, 2, 2, std::array<double, 4>{1, 2, 3, 4}, t);
    std::array<double, 2> x{};
    ml::solve_tri(std::array<double, 4>{2, 0, 1, 4}, std::array<double, 2>{4, 10}, x, 2, 1);
    return c == std::array<double, 4>{1, 3, 6, 10} && r == std::array<double, 3>{0.5, 2.5, 4.5} && idx == std::array<std::size_t, 5>{4, 1, 3, 2, 0}
           && t == std::array<double, 4>{1, 0, 3, 4} && x == std::array<double, 2>{2, 2} && ml::arange_count(0.0, 1.0, 0.25) == 4
           && ml::arange_count(0.0, 1.0, 0.3) == 4 && ml::arange_count(1.0, 0.0, 1.0) == 0;
}
static_assert(compile_time());
static_assert(ml::in_triangle(ml::tri_kind::upper, 0, 1) && !ml::in_triangle(ml::tri_kind::upper, 1, 1) && ml::in_triangle(ml::tri_kind::upper_diag, 1, 1));

std::vector<double> noise(std::size_t n, std::uint64_t seed)
{
    std::vector<double> v(n);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ull + 1;
    for (auto& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = static_cast<double>(x >> 11) * 0x1.0p-53 * 2.0 - 1.0;
    }
    return v;
}

TEST(TensorOpsTimestep, IsGgmlsLoopToTheBitCosinesFirst)
{
    // ggml_compute_forward_timestep_embedding_f32: freq = expf(-logf(max_period) * j / half), cos at j, sin at j + half
    const std::vector<float> t{0.0f, 1.0f, 17.5f, 999.0f};
    for (const std::size_t dim : {8u, 7u}) {
        std::vector<float> out(t.size() * dim, 42.0f);
        ml::timestep_embedding(t, dim, out);
        const std::size_t half = dim / 2;
        for (std::size_t i = 0; i < t.size(); ++i) {
            for (std::size_t j = 0; j < half; ++j) {
                const float freq = std::exp(-std::log(10000.0f) * float(j) / float(half));
                const float arg = t[i] * freq;
                EXPECT_EQ(std::bit_cast<std::uint32_t>(out[i * dim + j]), std::bit_cast<std::uint32_t>(std::cos(arg)));
                EXPECT_EQ(std::bit_cast<std::uint32_t>(out[i * dim + j + half]), std::bit_cast<std::uint32_t>(std::sin(arg)));
            }
            if (dim % 2 != 0) {
                EXPECT_EQ(out[i * dim + dim - 1], 0.0f);
            }
        }
        EXPECT_EQ(out[0], 1.0f);   // t = 0: cosines 1, sines 0
        EXPECT_EQ(out[half], 0.0f);
    }
}

TEST(TensorOpsArange, CountsAsGgmlAndFills)
{
    EXPECT_EQ(ml::arange_count(0.0f, 10.0f, 3.0f), 4u);   // 0 3 6 9
    EXPECT_EQ(ml::arange_count(5.0f, -1.0f, -2.0f), 3u);   // 5 3 1
    std::vector<float> v(ml::arange_count(5.0f, -1.0f, -2.0f));
    ml::arange(5.0f, -2.0f, v);
    EXPECT_EQ(v, (std::vector<float>{5, 3, 1}));
}

TEST(TensorOpsCumsum, RunningSumInPlaceAndItsBackward)
{
    const auto x = noise(9, 1);
    std::vector<double> y(x.size()), z = x;
    ml::cumsum(x, y);
    ml::cumsum(z, z);
    double s = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        s = i == 0 ? x[0] : s + x[i];
        EXPECT_EQ(y[i], s);
    }
    EXPECT_EQ(y, z);
    const auto g = noise(9, 2);
    std::vector<double> dx(9);
    ml::cumsum_backward(g, dx);
    for (std::size_t i = 0; i < 9; ++i) {
        double r = 0;
        for (std::size_t j = i; j < 9; ++j)
            r += g[j];
        EXPECT_NEAR(dx[i], r, 1e-14);
    }
    std::vector<double> e;
    ml::cumsum(e, e);
    ml::cumsum_backward(e, e);
    EXPECT_TRUE(e.empty());
}

TEST(TensorOpsArgsort, IsStableBothWays)
{
    const std::vector<float> x{2, -1, 2, 5, -1, 2};
    std::vector<std::size_t> up(6), down(6);
    ml::argsort(x, up, ml::sort_order::ascending);
    ml::argsort(x, down, ml::sort_order::descending);
    EXPECT_EQ(up, (std::vector<std::size_t>{1, 4, 0, 2, 5, 3}));
    EXPECT_EQ(down, (std::vector<std::size_t>{3, 0, 2, 5, 1, 4}));
}

TEST(TensorOpsTopK, MasksAllButTheLargestWithTiesToTheLowerIndex)
{
    const std::vector<float> x{1, 3, 3, 0, 3};
    std::vector<float> y(5);
    std::vector<std::size_t> scratch(5);
    constexpr float ninf = -std::numeric_limits<float>::infinity();
    ml::top_k_mask(x, 2, y, scratch);
    EXPECT_EQ(y, (std::vector<float>{ninf, 3, 3, ninf, ninf}));
    ml::top_k_mask(x, 0, y, scratch);
    EXPECT_EQ(y, std::vector<float>(5, ninf));
    std::vector<float> z = x;
    ml::top_k_mask(z, 5, z, scratch);
    EXPECT_EQ(z, x);
}

TEST(TensorOpsTri, EveryTriangleOfGgmlTri)
{
    // ggml_compute_forward_tri_t keeps column i0 of row i01 when the predicate holds, 0 otherwise; over a batch of two
    const std::vector<double> x = noise(2 * 3 * 4, 3);
    for (const auto kind : {ml::tri_kind::upper_diag, ml::tri_kind::upper, ml::tri_kind::lower_diag, ml::tri_kind::lower}) {
        std::vector<double> y(x.size());
        ml::tri(kind, 3, 4, x, y);
        for (std::size_t b = 0; b < 2; ++b)
            for (std::size_t r = 0; r < 3; ++r)
                for (std::size_t c = 0; c < 4; ++c) {
                    const bool keep = kind == ml::tri_kind::upper_diag ? c >= r : kind == ml::tri_kind::upper ? c > r
                                      : kind == ml::tri_kind::lower_diag ? c <= r : c < r;
                    EXPECT_EQ(y[(b * 3 + r) * 4 + c], keep ? x[(b * 3 + r) * 4 + c] : 0.0);
                }
    }
    std::vector<float> mask(9, 0.0f);
    ml::tri(ml::tri_kind::lower_diag, 3, 3, mask, mask, -std::numeric_limits<float>::infinity());   // a causal mask
    EXPECT_EQ(mask[1], -std::numeric_limits<float>::infinity());
    EXPECT_EQ(mask[3], 0.0f);
}

TEST(TensorOpsSolveTri, IsGgmlsSubstitutionToTheBit)
{
    const std::size_t n = 5, k = 3;
    auto a = noise(n * n, 4);
    for (std::size_t i = 0; i < n; ++i)
        a[i * n + i] += 3.0;   // well conditioned
    std::vector<float> af(a.size());
    std::ranges::transform(a, af.begin(), [](double v) { return float(v); });
    const auto bd = noise(n * k, 5);
    std::vector<float> b(bd.size());
    std::ranges::transform(bd, b.begin(), [](double v) { return float(v); });
    std::vector<float> x(n * k), ref(n * k);
    ml::solve_tri(af, b, x, n, k);
    for (std::size_t c = 0; c < k; ++c)   // ggml_compute_forward_solve_tri_f32
        for (std::size_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (std::size_t t = 0; t < i; ++t) {
                const float term = af[i * n + t] * ref[t * k + c];
                sum += term;
            }
            ref[i * k + c] = (b[i * k + c] - sum) / af[i * n + i];
        }
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(x[i]), std::bit_cast<std::uint32_t>(ref[i]));
    std::vector<float> inplace = b;
    ml::solve_tri(af, inplace, inplace, n, k);
    EXPECT_EQ(inplace, x);
}

TEST(TensorOpsSolveTri, BackwardMatchesCentralDifferences)
{
    const std::size_t n = 4, k = 2;
    auto a = noise(n * n, 6);
    for (std::size_t i = 0; i < n; ++i)
        a[i * n + i] += 2.5;
    auto b = noise(n * k, 7);
    const auto r = noise(n * k, 8);
    const auto loss = [&] {
        std::vector<double> x(n * k);
        ml::solve_tri(a, b, x, n, k);
        double s = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            s += x[i] * r[i];
        return s;
    };
    std::vector<double> x(n * k), da(n * n, 0.0), db(n * k);
    ml::solve_tri(a, b, x, n, k);
    ml::solve_tri_backward(a, x, r, da, db, n, k);
    const double h = 1e-6;
    const auto central = [&](std::vector<double>& v, std::size_t i) {
        const double keep = v[i];
        v[i] = keep + h;
        const double up = loss();
        v[i] = keep - h;
        const double down = loss();
        v[i] = keep;
        return (up - down) / (2 * h);
    };
    for (std::size_t i = 0; i < n * k; ++i)
        EXPECT_NEAR(db[i], central(b, i), 1e-7);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            EXPECT_NEAR(da[i * n + j], j <= i ? central(a, i * n + j) : 0.0, 1e-7);
}

} // namespace
