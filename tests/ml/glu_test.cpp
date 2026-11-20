//
//  glu_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/glu.hpp>: each gated linear unit against ggml's scalar loop (ggml/src/ggml-cpu/vec.h and
//  ops.cpp, llama.cpp 39039aa) written out here, the clamps of gpt-oss's SwiGLU and of SwiGLU with a clamp, the
//  backward passes against central differences, the rows and the split form with and without swapping, and
//  compile-time use.
//

#include <snicholls/core/ml/glu.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using ld = long double;

constexpr std::array kinds{ml::glu_kind::reglu,      ml::glu_kind::geglu,      ml::glu_kind::geglu_erf,   ml::glu_kind::geglu_quick,
                           ml::glu_kind::swiglu,     ml::glu_kind::swiglu_oai, ml::glu_kind::swiglu_clamp};

// ggml's loops, in double (ggml_vec_reglu_f32, ggml_vec_geglu_f32 without the fp16 table, ggml_vec_geglu_erf_f32,
// ggml_vec_geglu_quick_f32, ggml_vec_swiglu_f32, ggml_compute_forward_swiglu_oai_f32, ggml_compute_forward_swiglu_clamp_f32)
double ggml_glu(ml::glu_kind k, double x, double g, double alpha, double limit)
{
    switch (k) {
    case ml::glu_kind::reglu: return x > 0 ? x * g : 0;
    case ml::glu_kind::geglu: return 0.5 * x * (1 + std::tanh(0.79788456080286535587989211986876 * x * (1 + 0.044715 * x * x))) * g;
    case ml::glu_kind::geglu_erf: return 0.5 * x * (1 + std::erf(x * 0.70710678118654752440084436210484)) * g;
    case ml::glu_kind::geglu_quick: return x * (1 / (1 + std::exp(-1.702 * x))) * g;
    case ml::glu_kind::swiglu: return x / (1 + std::exp(-x)) * g;
    case ml::glu_kind::swiglu_oai: {
        const double a = std::min(x, limit), b = std::clamp(g, -limit, limit);
        return a / (1 + std::exp(alpha * -a)) * (b + 1);
    }
    case ml::glu_kind::swiglu_clamp: {
        const double a = std::min(x, limit), b = std::clamp(g, -limit, limit);
        return a / (1 + std::exp(-a)) * b;
    }
    }
    return 0;
}

TEST(MlGlu, EachKindIsGgmlsLoop)
{
    const ml::glu_parameters<double> p{1.702, 1.5};
    for (const auto k : kinds)
        for (double g = -4.0; g <= 4.0; g += 0.37)
            for (double u : {-2.5, -0.6, 0.0, 0.3, 1.9})
                EXPECT_NEAR(ml::glu(k, g, u, p), ggml_glu(k, g, u, p.alpha, p.limit), 1e-13) << int(k) << " " << g << " " << u;
}

TEST(MlGlu, GptOssClampsTheGateAboveAndTheUpBothSides)
{
    const ml::glu_parameters<double> p{};   // gpt-oss: alpha 1.702, limit 7
    EXPECT_EQ(p.alpha, 1.702);
    EXPECT_EQ(p.limit, 7.0);
    // Past the limits the output stops moving
    EXPECT_EQ(ml::glu(ml::glu_kind::swiglu_oai, 9.0, 1.0, p), ml::glu(ml::glu_kind::swiglu_oai, 7.0, 1.0, p));
    EXPECT_EQ(ml::glu(ml::glu_kind::swiglu_oai, 2.0, 30.0, p), ml::glu(ml::glu_kind::swiglu_oai, 2.0, 7.0, p));
    EXPECT_EQ(ml::glu(ml::glu_kind::swiglu_oai, 2.0, -30.0, p), ml::glu(ml::glu_kind::swiglu_oai, 2.0, -7.0, p));
    // The gate is not clamped from below: a very negative gate gives 0, not the value at -limit
    EXPECT_NEAR(ml::glu(ml::glu_kind::swiglu_oai, -40.0, 1.0, p), 0.0, 1e-25);
    // (u + 1): at u = -1 the output is 0 whatever the gate
    EXPECT_EQ(ml::glu(ml::glu_kind::swiglu_oai, 3.0, -1.0, p), 0.0);
    EXPECT_EQ(ml::glu(ml::glu_kind::swiglu_clamp, 9.0, 8.0, p), ml::silu(7.0) * 7.0);
}

template <typename F>
ld central(F f, ld x)
{
    const ld h = 1e-6L;
    return (f(x + h) - f(x - h)) / (2 * h);
}

TEST(MlGlu, BackwardMatchesCentralDifferences)
{
    const ml::glu_parameters<ld> p{1.702L, 1.5L};
    const ml::glu_parameters<double> pd{1.702, 1.5};
    for (const auto k : kinds)
        for (double g = -3.95; g <= 4.0; g += 0.41)
            for (double u : {-2.45, -0.6, 0.35, 1.9}) {
                const double dh = 0.75;
                const auto d = ml::glu_backward(k, g, u, dh, pd);
                const ld dg = central([&](ld v) { return ml::glu(k, v, ld(u), p); }, ld(g));
                const ld du = central([&](ld v) { return ml::glu(k, ld(g), v, p); }, ld(u));
                EXPECT_NEAR(d.gate, double(ld(dh) * dg), 1e-8) << int(k) << " " << g << " " << u;
                EXPECT_NEAR(d.up, double(ld(dh) * du), 1e-8) << int(k) << " " << g << " " << u;
            }
}

TEST(MlGlu, SwigluIsTheMlpsProductToTheBit)
{
    // The forms mlp_forward and mlp_backward always computed: silu(g) u, dh u silu'(g), dh silu(g)
    for (float g = -3.0f; g <= 3.0f; g += 0.25f)
        for (float u : {-1.5f, 0.25f, 2.0f}) {
            const float dh = 0.3f;
            EXPECT_EQ(ml::glu(ml::glu_kind::swiglu, g, u), ml::silu(g) * u);
            const auto d = ml::glu_backward(ml::glu_kind::swiglu, g, u, dh);
            EXPECT_EQ(d.gate, dh * u * ml::silu_derivative(g));
            EXPECT_EQ(d.up, dh * ml::silu(g));
        }
}

TEST(MlGlu, RowsAndTheSplitFormAreTheElementFunction)
{
    std::vector<float> gate(13), up(13), out(13), x(26), split(13), swapped(13);
    for (std::size_t i = 0; i < 13; ++i) {
        gate[i] = float(i) * 0.5f - 3.0f;
        up[i] = 1.0f - float(i) * 0.2f;
        x[i] = gate[i];
        x[13 + i] = up[i];
    }
    for (const auto k : kinds) {
        ml::glu_rows(k, gate, up, out);
        ml::glu_split(k, x, split);
        ml::glu_split(k, x, swapped, true);
        for (std::size_t i = 0; i < 13; ++i) {
            EXPECT_EQ(out[i], ml::glu(k, gate[i], up[i]));
            EXPECT_EQ(split[i], out[i]);
            EXPECT_EQ(swapped[i], ml::glu(k, up[i], gate[i]));
        }
        // Backward over rows and the split form: the element function's gradients, in the right halves
        std::vector<float> dout(13, 0.5f), dg(13), du(13), dx(26), dxs(26);
        ml::glu_rows_backward(k, gate, up, dout, dg, du);
        ml::glu_split_backward(k, x, dout, dx);
        ml::glu_split_backward(k, x, dout, dxs, true);
        for (std::size_t i = 0; i < 13; ++i) {
            const auto d = ml::glu_backward(k, gate[i], up[i], 0.5f);
            EXPECT_EQ(dg[i], d.gate);
            EXPECT_EQ(du[i], d.up);
            EXPECT_EQ(dx[i], d.gate);
            EXPECT_EQ(dx[13 + i], d.up);
            const auto e = ml::glu_backward(k, up[i], gate[i], 0.5f);
            EXPECT_EQ(dxs[13 + i], e.gate);
            EXPECT_EQ(dxs[i], e.up);
        }
    }
}

TEST(MlGlu, InPlaceOverTheGate)
{
    std::vector<double> g{-1.0, 0.5, 2.0}, u{3.0, -2.0, 0.5}, want(3);
    ml::glu_rows(ml::glu_kind::geglu, g, u, want);
    ml::glu_rows(ml::glu_kind::geglu, g, u, g);
    EXPECT_EQ(g, want);
}

constexpr bool compile_time()
{
    std::array<double, 4> x{1.0, -2.0, 3.0, 0.5}, y{};
    ml::glu_split(ml::glu_kind::reglu, x, std::span<double>(y).first(2));   // relu(1) 3, relu(-2) 0.5
    const auto d = ml::glu_backward(ml::glu_kind::reglu, 1.0, 3.0, 2.0);
    return y[0] == 3.0 && y[1] == 0.0 && d.gate == 6.0 && d.up == 2.0
           && ml::glu(ml::glu_kind::swiglu_clamp, 0.0, 5.0) == 0.0;
}
static_assert(compile_time());

} // namespace
