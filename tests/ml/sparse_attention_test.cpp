//
//  sparse_attention_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/sparse_attention.hpp>: every key selected in order is attend to the bit, forward and
//  backward; a selection is attention over the gathered rows; the backward pass touches only the selected rows and
//  matches central differences, with a sink; the indexer's scores against eq. 1 written out, its backward pass against
//  central differences; the top-k selection (order, ties, fewer keys than k); the KL loss and its gradient; compile-time use.
//

#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/sparse_attention.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace m = snicholls::math;

constexpr bool compile_time()
{
    const std::array<double, 5> scores{0.5, 2.0, -1.0, 2.0, 1.0};
    std::array<std::uint32_t, 3> top{};
    const std::size_t k = ml::select_top_k(scores, std::span<std::uint32_t>(top));
    // Two heads of 1: I = 1 · relu(2 · 3) + 0.5 · relu(-1 · 3) = 6
    const std::array<double, 2> q{2, -1}, w{1, 0.5};
    const std::array<double, 1> key{3};
    std::array<double, 1> s{};
    ml::indexer_scores(q, w, key, s);
    return k == 3 && top[0] == 1 && top[1] == 3 && top[2] == 4 && m::near(s[0], 6.0, 1e-12);
}
static_assert(compile_time());

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.53 * static_cast<double>(i) + b);
    return v;
}

TEST(MlSparseAttention, EveryKeyInOrderIsAttendToTheBit)
{
    constexpr std::size_t d = 6, n = 9;
    std::vector<float> q(d), k(n * d), v(n * d), dout(d);
    for (std::size_t i = 0; i < k.size(); ++i) {
        k[i] = std::sin(0.31f * static_cast<float>(i));
        v[i] = std::cos(0.17f * static_cast<float>(i));
    }
    for (std::size_t i = 0; i < d; ++i)
        q[i] = dout[i] = 0.3f * static_cast<float>(i) - 0.7f;
    std::vector<std::uint32_t> all(n);
    std::iota(all.begin(), all.end(), 0u);
    std::vector<float> a(d), b(d);
    const float l1 = ml::attend(q, k, v, a, 0.4f);
    const float l2 = ml::attend_selected(q, k, v, all, b, 0.4f);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(l1), std::bit_cast<std::uint32_t>(l2));
    EXPECT_EQ(a, b);
    std::vector<float> dq1(d), dq2(d), dk1(n * d), dk2(n * d), dv1(n * d), dv2(n * d);
    ml::attend_backward(q, k, v, a, dout, l1, 0.4f, dq1, dk1, dv1);
    ml::attend_selected_backward(q, k, v, all, b, dout, l2, 0.4f, -std::numeric_limits<float>::infinity(), dq2, dk2, dv2);
    EXPECT_EQ(dq1, dq2);
    EXPECT_EQ(dk1, dk2);
    EXPECT_EQ(dv1, dv2);
}

TEST(MlSparseAttention, ASelectionIsAttentionOverTheGatheredRows)
{
    constexpr std::size_t dk = 4, dv = 2, n = 10;
    const auto q = wave(dk, 1, 0), k = wave(n * dk, 1, 1), v = wave(n * dv, 1, 2);
    const std::vector<std::uint32_t> sel{1, 4, 5, 8};
    std::vector<double> gk, gv;
    for (const auto j : sel) {
        gk.insert(gk.end(), k.begin() + j * dk, k.begin() + (j + 1) * dk);
        gv.insert(gv.end(), v.begin() + j * dv, v.begin() + (j + 1) * dv);
    }
    std::vector<double> a(dv), b(dv);
    EXPECT_EQ(ml::attend_selected(q, k, v, sel, a, 0.5, 0.3), ml::attend_with_sink(q, gk, gv, b, 0.5, 0.3));
    EXPECT_EQ(a, b);
}

TEST(MlSparseAttention, BackwardTouchesOnlySelectedRowsAndMatchesCentralDifferences)
{
    constexpr std::size_t dk = 3, dv = 4, n = 8;
    auto q = wave(dk, 0.9, 0.4), k = wave(n * dk, 1.2, 0.8), v = wave(n * dv, 0.8, 1.7);
    const auto w = wave(dv, 1.0, 0.6);
    const std::vector<std::uint32_t> sel{6, 0, 3};
    double sink = -0.2;
    const auto loss = [&] {
        std::vector<double> out(dv);
        ml::attend_selected(q, k, v, sel, out, 0.8, sink);
        double l = 0;
        for (std::size_t i = 0; i < dv; ++i)
            l += w[i] * out[i];
        return l;
    };
    std::vector<double> out(dv), dq(dk), dkeys(n * dk), dvalues(n * dv);
    const double lse = ml::attend_selected(q, k, v, sel, out, 0.8, sink);
    const double dsink = ml::attend_selected_backward(q, k, v, sel, out, w, lse, 0.8, sink, dq, dkeys, dvalues);
    for (const std::size_t j : {1u, 2u, 4u, 5u, 7u}) {
        for (std::size_t i = 0; i < dk; ++i)
            EXPECT_EQ(dkeys[j * dk + i], 0.0);
        for (std::size_t i = 0; i < dv; ++i)
            EXPECT_EQ(dvalues[j * dv + i], 0.0);
    }
    const double h = 1e-6;
    const auto check = [&](double& x, double analytic) {
        const double x0 = x;
        x = x0 + h;
        const double up = loss();
        x = x0 - h;
        const double down = loss();
        x = x0;
        EXPECT_NEAR(analytic, (up - down) / (2 * h), 1e-8);
    };
    for (std::size_t i = 0; i < dk; ++i)
        check(q[i], dq[i]);
    for (std::size_t i = 0; i < n * dk; ++i)
        check(k[i], dkeys[i]);
    for (std::size_t i = 0; i < n * dv; ++i)
        check(v[i], dvalues[i]);
    check(sink, dsink);
}

TEST(MlSparseAttention, IndexerScoresAreEquationOneAndTheirBackwardMatchesCentralDifferences)
{
    constexpr std::size_t heads = 3, d = 4, n = 7;
    auto q = wave(heads * d, 1, 0.2), w = wave(heads, 1, 1.3), k = wave(n * d, 1, 2.1);
    std::vector<double> s(n);
    ml::indexer_scores(q, w, k, s);
    for (std::size_t t = 0; t < n; ++t) {
        double want = 0;
        for (std::size_t j = 0; j < heads; ++j) {
            double a = 0;
            for (std::size_t i = 0; i < d; ++i)
                a += q[j * d + i] * k[t * d + i];
            want += w[j] * std::max(a, 0.0);
        }
        EXPECT_NEAR(s[t], want, 1e-14);
    }
    const auto g = wave(n, 1, -0.5);
    const auto loss = [&] {
        std::vector<double> z(n);
        ml::indexer_scores(q, w, k, z);
        double l = 0;
        for (std::size_t t = 0; t < n; ++t)
            l += g[t] * z[t];
        return l;
    };
    std::vector<double> dq(q.size()), dw(heads), dk(k.size());
    ml::indexer_scores_backward(q, w, k, g, dq, dw, dk);
    const double h = 1e-7;
    const auto check = [&](double& x, double analytic) {
        const double x0 = x;
        x = x0 + h;
        const double up = loss();
        x = x0 - h;
        const double down = loss();
        x = x0;
        EXPECT_NEAR(analytic, (up - down) / (2 * h), 1e-7);
    };
    for (std::size_t i = 0; i < q.size(); ++i)
        check(q[i], dq[i]);
    for (std::size_t i = 0; i < heads; ++i)
        check(w[i], dw[i]);
    for (std::size_t i = 0; i < k.size(); ++i)
        check(k[i], dk[i]);
}

TEST(MlSparseAttention, SelectTopKIsTheBestInPositionOrderTiesToTheEarlier)
{
    const std::vector<float> s{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    std::array<std::uint32_t, 4> top{};
    EXPECT_EQ(ml::select_top_k(s, std::span<std::uint32_t>(top)), 4u);
    EXPECT_EQ(top, (std::array<std::uint32_t, 4>{4, 5, 7, 8}));   // 9 and 6, then the first two of the three 5s
    std::array<std::uint32_t, 5> five{};
    ml::select_top_k(s, std::span<std::uint32_t>(five));
    EXPECT_EQ(five, (std::array<std::uint32_t, 5>{4, 5, 7, 8, 10}));
    std::array<std::uint32_t, 8> more{};
    EXPECT_EQ(ml::select_top_k(std::span<const float>(s).first(3), std::span<std::uint32_t>(more)), 3u);
    EXPECT_EQ(more[0], 0u);
    EXPECT_EQ(more[2], 2u);
    EXPECT_EQ(ml::select_top_k(std::span<const float>{}, std::span<std::uint32_t>(more)), 0u);
}

TEST(MlSparseAttention, KlLossIsTheDivergenceAndItsGradientMatchesCentralDifferences)
{
    constexpr std::size_t n = 6;
    std::vector<double> p{0.1, 0.0, 0.3, 0.25, 0.15, 0.2};
    auto z = wave(n, 1.5, 0.3);
    std::vector<double> g(n);
    const double loss = ml::indexer_kl_loss(p, z, g);
    double mx = z[0], sum = 0;
    for (const double x : z)
        mx = std::max(mx, x);
    for (const double x : z)
        sum += std::exp(x - mx);
    double want = 0;
    for (std::size_t s = 0; s < n; ++s)
        if (p[s] > 0)
            want += p[s] * std::log(p[s] / (std::exp(z[s] - mx) / sum));
    EXPECT_NEAR(loss, want, 1e-14);
    std::vector<double> scratch(n);
    const double h = 1e-6;
    for (std::size_t s = 0; s < n; ++s) {
        const double z0 = z[s];
        z[s] = z0 + h;
        const double up = ml::indexer_kl_loss(p, z, scratch);
        z[s] = z0 - h;
        const double down = ml::indexer_kl_loss(p, z, scratch);
        z[s] = z0;
        EXPECT_NEAR(g[s], (up - down) / (2 * h), 1e-8);
    }
    // The target equal to softmax(I): no divergence, no gradient
    for (std::size_t s = 0; s < n; ++s)
        p[s] = std::exp(z[s] - mx) / sum;
    EXPECT_NEAR(ml::indexer_kl_loss(p, z, g), 0.0, 1e-14);
    for (const double x : g)
        EXPECT_NEAR(x, 0.0, 1e-15);
    EXPECT_EQ(ml::indexer_kl_loss(std::span<const double>{}, std::span<const double>{}, std::span<double>{}), 0.0);
}

} // namespace
