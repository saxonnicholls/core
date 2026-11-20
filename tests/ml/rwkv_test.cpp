//
//  rwkv_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/rwkv.hpp>: RWKV-6's and RWKV-7's WKV against transcriptions of llama.cpp's
//  GGML_OP_RWKV_WKV6 and GGML_OP_RWKV_WKV7 (commit 39039aa) to the bit; decoding token by token to the bit; both
//  backward passes against central differences in double over two sequences with initial states; token shift
//  forward (llama.cpp's (x_prev - x) · μ + x), in pieces, by step, and its backward pass; sizes at compile time.
//

#include <snicholls/core/ml/rwkv.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include "../support/mixer_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::core_testing::central_difference;
using snicholls::core_testing::uniform;
using snicholls::core_testing::weighted;
constexpr auto& ref = snicholls::linalg::reference;

// 2 heads of 3, chunks of 2
constexpr ml::transformer_config C = [] {
    auto c = ml::tiny_mamba2;
    c.mixer = ml::mixer_kind::rwkv7;
    c.mixer_heads = 2;
    c.mixer_head_dim = 3;
    c.mixer_state = 3;
    c.mixer_groups = 2;
    c.conv_width = 0;
    c.mixer_chunk = 2;
    return c;
}();
constexpr std::size_t H = 2, N = 3, D = H * N, S = H * N * N;
static_assert(ml::valid(C) && ml::wkv_state_size(C) == S);

constexpr float compile_time_shift()
{
    const std::array<float, 2> x{2, 4};
    const std::array<float, 1> mu{0.5f};
    std::array<float, 2> out{};
    ml::token_shift(ref, std::span<const float>(x), std::span<const float>(mu), std::span<float>(out));
    return out[0] + out[1];   // 1 + 3
}
static_assert(compile_time_shift() == 4.0f);

std::vector<float> exps(const std::vector<float>& v)
{
    std::vector<float> e(v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
        e[i] = std::exp(v[i]);
    return e;
}

// ggml_compute_forward_rwkv_wkv6_f32, scalar path: state [i key][j value]
std::vector<float> ggml_wkv6(const std::vector<float>& k, const std::vector<float>& v, const std::vector<float>& r,
                             const std::vector<float>& tf, const std::vector<float>& td, std::vector<float>& state, std::size_t T)
{
    std::vector<float> dst(T * D, 0.0f);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t h = 0; h < H; ++h)
            for (std::size_t i = 0; i < N; ++i) {
                const std::size_t ti = t * D + h * N + i;
                const float k_val = k[ti], r_val = r[ti], time_faaaa_val = tf[h * N + i], time_decay_val = td[ti];
                for (std::size_t j = 0; j < N; ++j) {
                    const std::size_t tj = t * D + h * N + j, s = h * N * N + i * N + j;
                    const float kv_val = v[tj] * k_val;
                    const float prev_state_val = state[s];
                    const float temp_val = kv_val * time_faaaa_val + prev_state_val;
                    dst[tj] += temp_val * r_val;
                    state[s] = prev_state_val * time_decay_val + kv_val;
                }
            }
    return dst;
}

// ggml_compute_forward_rwkv_wkv7_f32, scalar path: state [i value][j key]
std::vector<float> ggml_wkv7(const std::vector<float>& r, const std::vector<float>& w, const std::vector<float>& k, const std::vector<float>& v,
                             const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& state, std::size_t T)
{
    std::vector<float> dst(T * D, 0.0f);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t h = 0; h < H; ++h)
            for (std::size_t i = 0; i < N; ++i) {
                const std::size_t ti = t * D + h * N + i, si = h * N * N + i * N;
                const float v_val = v[ti];
                float sa = 0, result = 0;
                for (std::size_t j = 0; j < N; ++j)
                    sa += a[t * D + h * N + j] * state[si + j];
                for (std::size_t j = 0; j < N; ++j) {
                    const std::size_t tj = t * D + h * N + j;
                    const float kv_val = v_val * k[tj];
                    state[si + j] = state[si + j] * w[tj] + kv_val + sa * b[tj];
                    result += state[si + j] * r[tj];
                }
                dst[ti] = result;
            }
    return dst;
}

TEST(MlRwkv, Wkv6IsLlamaCppsWkv6ToTheBit)
{
    constexpr std::size_t rows = 5;
    const auto r = uniform<float>(rows * D, 1), k = uniform<float>(rows * D, 2), v = uniform<float>(rows * D, 3),
               lw = uniform<float>(rows * D, 4, -2.0, -0.01), u = uniform<float>(D, 5);
    std::vector<float> o(rows * D), s(S, 0.0f), gs(S, 0.0f);
    ml::wkv6_forward<C>(ref, ml::wkv6_inputs<float>{r, k, v, lw, u}, std::span<float>(o), 1, std::span<float>(s));
    EXPECT_EQ(o, ggml_wkv6(k, v, r, u, exps(lw), gs, rows));
    EXPECT_EQ(s, gs);
}

TEST(MlRwkv, Wkv7IsLlamaCppsWkv7ToTheBit)
{
    constexpr std::size_t rows = 5;
    const auto r = uniform<float>(rows * D, 6), lw = uniform<float>(rows * D, 7, -0.6, -0.01), k = uniform<float>(rows * D, 8),
               v = uniform<float>(rows * D, 9), a = uniform<float>(rows * D, 10, -0.5, 0.5), b = uniform<float>(rows * D, 11, -0.5, 0.5);
    std::vector<float> o(rows * D), s(S, 0.0f), gs(S, 0.0f);
    ml::wkv7_forward<C>(ref, ml::wkv7_inputs<float>{r, lw, k, v, a, b}, std::span<float>(o), 1, std::span<float>(s));
    EXPECT_EQ(o, ggml_wkv7(r, exps(lw), k, v, a, b, gs, rows));
    for (std::size_t h = 0; h < H; ++h)   // ggml's state is the transpose
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t p = 0; p < N; ++p)
                EXPECT_EQ(s[h * N * N + n * N + p], gs[h * N * N + p * N + n]);
}

TEST(MlRwkv, DecodingIsTheSequenceForwardToTheBit)
{
    constexpr std::size_t rows = 4;
    const auto r = uniform<float>(rows * D, 12), k = uniform<float>(rows * D, 13), v = uniform<float>(rows * D, 14),
               lw = uniform<float>(rows * D, 15, -1.0, -0.01), u = uniform<float>(D, 16), a = uniform<float>(rows * D, 17, -0.5, 0.5),
               b = uniform<float>(rows * D, 18, -0.5, 0.5);
    std::vector<float> o6(rows * D), o7(rows * D), s6(S, 0.0f), s7(S, 0.0f), t6(S, 0.0f), t7(S, 0.0f), ot(D);
    ml::wkv6_forward<C>(ref, ml::wkv6_inputs<float>{r, k, v, lw, u}, std::span<float>(o6), 1, std::span<float>(s6));
    ml::wkv7_forward<C>(ref, ml::wkv7_inputs<float>{r, lw, k, v, a, b}, std::span<float>(o7), 1, std::span<float>(s7));
    for (std::size_t t = 0; t < rows; ++t) {
        const auto row = [&](const std::vector<float>& x) { return std::span<const float>(x).subspan(t * D, D); };
        ml::wkv6_step<C>(ml::wkv6_inputs<float>{row(r), row(k), row(v), row(lw), u}, std::span<float>(ot), std::span<float>(t6));
        for (std::size_t i = 0; i < D; ++i)
            EXPECT_EQ(ot[i], o6[t * D + i]);
        ml::wkv7_step<C>(ml::wkv7_inputs<float>{row(r), row(lw), row(k), row(v), row(a), row(b)}, std::span<float>(ot), std::span<float>(t7));
        for (std::size_t i = 0; i < D; ++i)
            EXPECT_EQ(ot[i], o7[t * D + i]);
    }
    EXPECT_EQ(t6, s6);
    EXPECT_EQ(t7, s7);
}

TEST(MlRwkv, Wkv6BackwardMatchesCentralDifferences)
{
    constexpr std::size_t n = 5, seqs = 2, rows = n * seqs;
    auto r = uniform<double>(rows * D, 20), k = uniform<double>(rows * D, 21), v = uniform<double>(rows * D, 22),
         lw = uniform<double>(rows * D, 23, -1.5, -0.05), u = uniform<double>(D, 24), init = uniform<double>(seqs * S, 25);
    const auto wo = uniform<double>(rows * D, 26), ws = uniform<double>(seqs * S, 27);
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    const auto loss = [&] {
        std::vector<double> o(rows * D), s = init;
        ml::wkv6_forward<C>(ref, ml::wkv6_inputs<double>{r, k, v, lw, u}, std::span<double>(o), seqs, std::span<double>(s));
        return weighted(o, wo) + weighted(s, ws);
    };
    std::vector<double> o(rows * D), s = init, cps(seqs * m * S);
    ml::wkv6_forward<C>(ref, ml::wkv6_inputs<double>{r, k, v, lw, u}, std::span<double>(o), seqs, std::span<double>(s), std::span<double>(cps));
    std::vector<double> dr(r.size()), dk(k.size()), dv(v.size()), dlw(lw.size()), du(u.size()), ds = ws, work(ml::wkv_backward_work<C, double>());
    ml::wkv6_backward<C>(ref, ml::wkv6_inputs<double>{r, k, v, lw, u}, std::span<const double>(wo), ml::wkv6_gradients<double>{dr, dk, dv, dlw, du},
                         seqs, std::span<const double>(cps), std::span<double>(ds), std::span<double>(work));
    const auto check = [&](std::vector<double>& x, const std::vector<double>& g, const char* what) {
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(g[i], central_difference(x, i, loss), 1e-7) << what << " " << i;
    };
    check(r, dr, "r");
    check(k, dk, "k");
    check(v, dv, "v");
    check(lw, dlw, "lw");
    check(u, du, "u");
    check(init, ds, "initial state");
}

TEST(MlRwkv, Wkv7BackwardMatchesCentralDifferences)
{
    constexpr std::size_t n = 5, seqs = 2, rows = n * seqs;
    auto r = uniform<double>(rows * D, 30), lw = uniform<double>(rows * D, 31, -0.6, -0.01), k = uniform<double>(rows * D, 32),
         v = uniform<double>(rows * D, 33), a = uniform<double>(rows * D, 34, -0.5, 0.5), b = uniform<double>(rows * D, 35, -0.5, 0.5),
         init = uniform<double>(seqs * S, 36);
    const auto wo = uniform<double>(rows * D, 37), ws = uniform<double>(seqs * S, 38);
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    const auto loss = [&] {
        std::vector<double> o(rows * D), s = init;
        ml::wkv7_forward<C>(ref, ml::wkv7_inputs<double>{r, lw, k, v, a, b}, std::span<double>(o), seqs, std::span<double>(s));
        return weighted(o, wo) + weighted(s, ws);
    };
    std::vector<double> o(rows * D), s = init, cps(seqs * m * S);
    ml::wkv7_forward<C>(ref, ml::wkv7_inputs<double>{r, lw, k, v, a, b}, std::span<double>(o), seqs, std::span<double>(s), std::span<double>(cps));
    std::vector<double> dr(r.size()), dlw(lw.size()), dk(k.size()), dv(v.size()), da(a.size()), db(b.size()), ds = ws,
        work(ml::wkv_backward_work<C, double>());
    ml::wkv7_backward<C>(ref, ml::wkv7_inputs<double>{r, lw, k, v, a, b}, std::span<const double>(wo),
                         ml::wkv7_gradients<double>{dr, dlw, dk, dv, da, db}, seqs, std::span<const double>(cps), std::span<double>(ds),
                         std::span<double>(work));
    const auto check = [&](std::vector<double>& x, const std::vector<double>& g, const char* what) {
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(g[i], central_difference(x, i, loss), 1e-7) << what << " " << i;
    };
    check(r, dr, "r");
    check(lw, dlw, "lw");
    check(k, dk, "k");
    check(v, dv, "v");
    check(a, da, "a");
    check(b, db, "b");
    check(init, ds, "initial state");
}

TEST(MlRwkv, TokenShiftInPiecesAndByStepIsTheWholeSequence)
{
    constexpr std::size_t d = 4, n = 6;
    const auto x = uniform<float>(n * d, 40), mu = uniform<float>(d, 41, 0.0, 1.0);
    std::vector<float> whole(n * d), pieces(n * d), state(d, 0.0f), step_state(d, 0.0f), out(d);
    ml::token_shift(ref, std::span<const float>(x), std::span<const float>(mu), std::span<float>(whole));
    ml::token_shift(ref, std::span<const float>(x).first(2 * d), std::span<const float>(mu), std::span<float>(pieces).first(2 * d), 1,
                    std::span<float>(state));
    ml::token_shift(ref, std::span<const float>(x).subspan(2 * d), std::span<const float>(mu), std::span<float>(pieces).subspan(2 * d), 1,
                    std::span<float>(state));
    EXPECT_EQ(pieces, whole);
    for (std::size_t t = 0; t < n; ++t) {
        ml::token_shift_step(std::span<const float>(x).subspan(t * d, d), std::span<const float>(mu), std::span<float>(out),
                             std::span<float>(step_state));
        for (std::size_t c = 0; c < d; ++c)
            EXPECT_EQ(out[c], whole[t * d + c]);
    }
    for (std::size_t c = 0; c < d; ++c)   // llama.cpp's form
        EXPECT_EQ(whole[d + c], (x[c] - x[d + c]) * mu[c] + x[d + c]);
}

TEST(MlRwkv, TokenShiftBackwardMatchesCentralDifferences)
{
    constexpr std::size_t d = 3, n = 4, seqs = 2;
    auto x = uniform<double>(seqs * n * d, 50), mu = uniform<double>(d, 51, 0.0, 1.0), init = uniform<double>(seqs * d, 52);
    const auto wo = uniform<double>(x.size(), 53);
    const auto loss = [&] {
        std::vector<double> out(x.size()), st = init;
        ml::token_shift(ref, std::span<const double>(x), std::span<const double>(mu), std::span<double>(out), seqs, std::span<double>(st));
        return weighted(out, wo);
    };
    std::vector<double> dx(x.size()), dmu(d), dinit(init.size());
    ml::token_shift_backward(ref, std::span<const double>(x), std::span<const double>(mu), std::span<const double>(wo), std::span<double>(dx),
                             std::span<double>(dmu), seqs, std::span<const double>(init), std::span<double>(dinit));
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(dx[i], central_difference(x, i, loss), 1e-8) << "x " << i;
    for (std::size_t i = 0; i < d; ++i)
        EXPECT_NEAR(dmu[i], central_difference(mu, i, loss), 1e-8) << "mu " << i;
    for (std::size_t i = 0; i < init.size(); ++i)
        EXPECT_NEAR(dinit[i], central_difference(init, i, loss), 1e-8) << "initial " << i;
}

} // namespace
