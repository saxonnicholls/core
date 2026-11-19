//
//  pooling_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/pooling.hpp>: each pooling over packed sequences against a direct loop to the bit
//  (the mean against mean_pool_rows), the backward passes against central differences and an adjoint identity, empty
//  sequences, the reranker head against llama.cpp's RANK graph written out (src/llama-graph.cpp, 39039aa), its absent
//  layers, and compile-time use.
//

#include <snicholls/core/ml/pooling.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.43 * double(i) + b);
    return v;
}

TEST(MlPooling, EachKindIsTheDirectLoop)
{
    constexpr std::size_t w = 3;
    const auto rows = wave(9 * w, 1.0, 0.2);   // sequences of 4, 1 and 4 rows
    const std::array<std::size_t, 4> starts{0, 4, 5, 9};
    std::vector<double> mean(3 * w), cls(3 * w), last(3 * w), rank(3 * w), none(rows.size());
    ml::pool_rows<double>(ml::pooling_kind::mean, rows, w, starts, mean);
    ml::pool_rows<double>(ml::pooling_kind::cls, rows, w, starts, cls);
    ml::pool_rows<double>(ml::pooling_kind::last, rows, w, starts, last);
    ml::pool_rows<double>(ml::pooling_kind::rank, rows, w, starts, rank);
    ml::pool_rows<double>(ml::pooling_kind::none, rows, w, starts, none);
    EXPECT_EQ(none, rows);
    EXPECT_EQ(rank, cls);
    for (std::size_t s = 0; s < 3; ++s)
        for (std::size_t c = 0; c < w; ++c) {
            double sum = 0;
            for (std::size_t r = starts[s]; r < starts[s + 1]; ++r)
                sum += rows[r * w + c];
            EXPECT_EQ(mean[s * w + c], sum / double(starts[s + 1] - starts[s]));
            EXPECT_EQ(cls[s * w + c], rows[starts[s] * w + c]);
            EXPECT_EQ(last[s * w + c], rows[(starts[s + 1] - 1) * w + c]);
        }
    // An empty sequence's mean is zeros
    const std::array<std::size_t, 3> with_empty{0, 0, 9};
    std::vector<double> m2(2 * w, 7.0);
    ml::pool_rows<double>(ml::pooling_kind::mean, rows, w, with_empty, m2);
    EXPECT_EQ(m2[0], 0.0);
}

TEST(MlPooling, BackwardIsTheAdjoint)
{
    constexpr std::size_t w = 2;
    const auto rows = wave(7 * w, 1.0, 0.7), dout = wave(3 * w, 1.0, 1.9);
    const std::array<std::size_t, 4> starts{0, 3, 4, 7};
    for (const auto kind : {ml::pooling_kind::mean, ml::pooling_kind::cls, ml::pooling_kind::last, ml::pooling_kind::rank}) {
        std::vector<double> y(3 * w), drows(rows.size());
        ml::pool_rows<double>(kind, rows, w, starts, y);
        ml::pool_rows_backward<double>(kind, dout, w, starts, drows);
        // pooling is linear: <pool(x), dout> = <x, pool^T(dout)>
        double a = 0, b = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            a += y[i] * dout[i];
        for (std::size_t i = 0; i < rows.size(); ++i)
            b += rows[i] * drows[i];
        EXPECT_NEAR(a, b, 1e-14) << int(kind);
    }
    std::vector<double> d(rows.size(), 1.0);
    ml::pool_rows_backward<double>(ml::pooling_kind::none, rows, w, starts, d);
    EXPECT_EQ(d[3], 1.0 + rows[3]);
}

TEST(MlPoolingRank, TheHeadIsLlamaCppsRankGraph)
{
    constexpr std::size_t width = 4, hidden = 3;
    const auto x = wave(width, 1.0, 0.4), cls = wave(hidden * width, 0.5, 1.0), cls_b = wave(hidden, 0.1, 2.0), out = wave(2 * hidden, 0.7, 0.3),
               out_b = wave(2, 0.2, 1.1), nw = wave(hidden, 1.0, 0.9);
    const ml::rank_head_weights<double, double> w{{cls, hidden, width}, cls_b, ml::rank_activation::tanh, nw, {}, 1e-12,
                                                  {out, 2, hidden}, out_b, true};
    std::array<double, hidden> scratch{};
    std::array<double, 2> scores{};
    ml::rank_head(w, x, scratch, scores);
    // ggml_mul_mat(cls, cur) + cls_b, tanh, layer norm with weight, ggml_mul_mat(cls_out, cur) + cls_out_b, softmax
    std::array<double, hidden> h{};
    for (std::size_t i = 0; i < hidden; ++i) {
        double s = 0;
        for (std::size_t j = 0; j < width; ++j)
            s += cls[i * width + j] * x[j];
        h[i] = std::tanh(s + cls_b[i]);
    }
    double mean = (h[0] + h[1] + h[2]) / 3, var = 0;
    for (const double v : h)
        var += (v - mean) * (v - mean);
    const double rstd = 1 / std::sqrt(var / 3 + 1e-12);
    for (std::size_t i = 0; i < hidden; ++i)
        h[i] = (h[i] - mean) * rstd * nw[i];
    std::array<double, 2> logit{};
    for (std::size_t i = 0; i < 2; ++i) {
        logit[i] = out_b[i];
        for (std::size_t j = 0; j < hidden; ++j)
            logit[i] += out[i * hidden + j] * h[j];
    }
    const double z = std::exp(logit[0]) + std::exp(logit[1]);
    EXPECT_NEAR(scores[0], std::exp(logit[0]) / z, 1e-13);
    EXPECT_NEAR(scores[0] + scores[1], 1.0, 1e-15);
}

TEST(MlPoolingRank, AbsentLayersPassTheRowThrough)
{
    const std::array<double, 3> x{0.5, -1.0, 2.0}, proj{1.0, 1.0, 1.0};
    std::array<double, 1> score{};
    // Jina's single-layer head: only cls_out
    ml::rank_head(ml::rank_head_weights<double, double>{.out = {proj, 1, 3}}, x, std::span<double>{}, score);
    EXPECT_EQ(score[0], 1.5);
    // Only cls, with each activation
    std::array<double, 3> s3{}, scratch{};
    const std::array<double, 9> eye{1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (const auto a : {ml::rank_activation::tanh, ml::rank_activation::gelu, ml::rank_activation::relu, ml::rank_activation::identity}) {
        ml::rank_head(ml::rank_head_weights<double, double>{.cls = {eye, 3, 3}, .activation = a}, x, scratch, s3);
        const double want = a == ml::rank_activation::tanh ? std::tanh(-1.0) : a == ml::rank_activation::gelu ? ml::gelu(-1.0)
                          : a == ml::rank_activation::relu ? 0.0 : -1.0;
        EXPECT_NEAR(s3[1], want, 1e-15) << int(a);   // math::tanh against std::tanh
    }
}

constexpr bool compile_time()
{
    const std::array<double, 6> rows{1, 2, 3, 4, 5, 6};   // three rows of 2: sequences [0, 2) and [2, 3)
    const std::array<std::size_t, 3> starts{0, 2, 3};
    std::array<double, 4> mean{}, last{};
    ml::pool_rows<double>(ml::pooling_kind::mean, rows, 2, starts, mean);
    ml::pool_rows<double>(ml::pooling_kind::last, rows, 2, starts, last);
    return mean[0] == 2 && mean[1] == 3 && mean[2] == 5 && last[0] == 3 && last[3] == 6;
}
static_assert(compile_time());

} // namespace
