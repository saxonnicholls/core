//
//  delta_rule_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/delta_rule.hpp>: the recurrent form against a transcription of llama.cpp's
//  GGML_OP_GATED_DELTA_NET (commit 39039aa), with a gate per head (Qwen3-Next) and per key channel (Kimi); the chunked
//  WY / UT form against it within the stated rounding (float and double, both gates, a short last chunk, two sequences,
//  key groups shared by two heads, initial states); decoding token by token to the bit; the backward pass against
//  central differences in double from either form's checkpoints, both gates; the L2 normalisation and its backward
//  pass; sizes at compile time.
//

#include <snicholls/core/ml/delta_rule.hpp>

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
using snicholls::core_testing::max_relative_difference;
using snicholls::core_testing::uniform;
using snicholls::core_testing::weighted;
constexpr auto& ref = snicholls::linalg::reference;

// 4 value heads of 3 channels sharing 2 key heads of 4 channels, chunks of 3
constexpr ml::transformer_config C = [] {
    auto c = ml::tiny_gated_deltanet;
    c.mixer_heads = 4;
    c.mixer_head_dim = 3;
    c.mixer_state = 4;
    c.mixer_groups = 2;
    c.mixer_chunk = 3;
    return c;
}();
constexpr std::size_t H = 4, P = 3, N = 4, G = 2, S = H * N * P;
static_assert(ml::delta_rule_state_size(C) == S && ml::delta_rule_chunk_work(C) == 3 * 4 + 9 + 3 * 7 + 3 * 3);

template <typename T>
struct problem
{
    std::vector<T> q, k, v, g, beta;
    problem(std::size_t r, std::uint64_t seed, bool per_channel)
        : q(uniform<T>(r * G * N, seed)), k(uniform<T>(r * G * N, seed + 1)), v(uniform<T>(r * H * P, seed + 2)),
          g(uniform<T>(r * H * (per_channel ? N : 1), seed + 3, -0.8, -0.01)), beta(uniform<T>(r * H, seed + 4, 0.1, 0.9))
    {
        ml::l2_normalise_rows(ref, std::span<const T>(q), std::span<T>(q), N, T(1e-6));
        ml::l2_normalise_rows(ref, std::span<const T>(k), std::span<T>(k), N, T(1e-6));
    }
    ml::delta_rule_inputs<T> in() const { return {q, k, v, g, beta}; }
};

// ggml_compute_forward_gated_delta_net_one_chunk, K = 1, one key head per head: the state stored transposed,
// s_out[j·S_v + i] = S[i][j]; plain loops in place of ggml_vec_*
std::vector<float> ggml_gated_delta_net(const problem<float>& pr, std::vector<float>& state, std::size_t T, bool kda)
{
    constexpr std::size_t Sv = 4, Hv = 4;   // square heads, as the op requires
    std::vector<float> attn(T * Hv * Sv), delta(Sv);
    const float scale = 1.0f / std::sqrt(static_cast<float>(Sv));
    for (std::size_t h = 0; h < Hv; ++h) {
        float* s_out = state.data() + h * Sv * Sv;
        for (std::size_t t = 0; t < T; ++t) {
            const float* q_d = pr.q.data() + (t * Hv + h) * Sv;
            const float* k_d = pr.k.data() + (t * Hv + h) * Sv;
            const float* v_d = pr.v.data() + (t * Hv + h) * Sv;
            const float beta_val = pr.beta[t * Hv + h];
            if (kda) {
                for (std::size_t i = 0; i < Sv; ++i)
                    delta[i] = std::exp(pr.g[(t * Hv + h) * Sv + i]);
                for (std::size_t j = 0; j < Sv; ++j)
                    for (std::size_t i = 0; i < Sv; ++i)
                        s_out[j * Sv + i] *= delta[i];
            } else
                for (std::size_t i = 0; i < Sv * Sv; ++i)
                    s_out[i] *= std::exp(pr.g[t * Hv + h]);
            for (std::size_t j = 0; j < Sv; ++j) {
                float sum = 0.0f;
                for (std::size_t i = 0; i < Sv; ++i)
                    sum += s_out[j * Sv + i] * k_d[i];
                delta[j] = (v_d[j] - sum) * beta_val;
            }
            for (std::size_t j = 0; j < Sv; ++j)
                for (std::size_t i = 0; i < Sv; ++i)
                    s_out[j * Sv + i] += k_d[i] * delta[j];
            for (std::size_t j = 0; j < Sv; ++j) {
                float sum = 0.0f;
                for (std::size_t i = 0; i < Sv; ++i)
                    sum += s_out[j * Sv + i] * q_d[i];
                attn[(t * Hv + h) * Sv + j] = sum * scale;
            }
        }
    }
    return attn;
}

constexpr ml::transformer_config square = [] {
    auto c = C;
    c.mixer_groups = 4;
    c.mixer_head_dim = 4;
    return c;
}();

TEST(MlDeltaRule, TheRecurrentFormIsLlamaCppsGatedDeltaNet)
{
    for (const bool kda : {false, true}) {
        constexpr std::size_t rows = 6, D = 16;
        problem<float> pr(rows, kda ? 1 : 2, kda);   // G = H = 4 here: resize the keys and queries to 4 heads
        pr.q = uniform<float>(rows * D, 3);
        pr.k = uniform<float>(rows * D, 4);
        ml::l2_normalise_rows(ref, std::span<const float>(pr.q), std::span<float>(pr.q), 4, 1e-6f);
        ml::l2_normalise_rows(ref, std::span<const float>(pr.k), std::span<float>(pr.k), 4, 1e-6f);
        pr.v = uniform<float>(rows * D, 5);
        std::vector<float> o(rows * D), s(64, 0.0f), gs(64, 0.0f);
        ml::delta_rule_forward<square>(ref, pr.in(), std::span<float>(o), 1, std::span<float>(s));
        const auto expected = ggml_gated_delta_net(pr, gs, rows, kda);
        EXPECT_EQ(o, expected) << (kda ? "per channel" : "per head");
        for (std::size_t h = 0; h < 4; ++h)
            for (std::size_t n = 0; n < 4; ++n)
                for (std::size_t p = 0; p < 4; ++p)
                    EXPECT_EQ(s[h * 16 + n * 4 + p], gs[h * 16 + p * 4 + n]);
    }
}

template <typename T>
void chunked_is_recurrent(double bound, bool kda)
{
    constexpr std::size_t n = 8, seqs = 2;
    const problem<T> pr(n * seqs, 10, kda);
    const auto init = uniform<T>(seqs * S, 20);
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    std::vector<T> o1(pr.v.size()), o2(pr.v.size()), s1 = init, s2 = init, c1(seqs * m * S), c2(seqs * m * S), work(ml::delta_rule_chunk_work(C));
    ml::delta_rule_forward<C>(ref, pr.in(), std::span<T>(o1), seqs, std::span<T>(s1), std::span<T>(c1));
    ml::delta_rule_forward_chunked<C>(ref, pr.in(), std::span<T>(o2), seqs, std::span<T>(s2), std::span<T>(c2), std::span<T>(work));
    EXPECT_LT(max_relative_difference(o1, o2), bound) << kda;
    EXPECT_LT(max_relative_difference(s1, s2), bound) << kda;
    EXPECT_LT(max_relative_difference(c1, c2), bound) << kda;
}

TEST(MlDeltaRule, TheChunkedWyFormIsTheRecurrentFormWithinRounding)
{
    for (const bool kda : {false, true}) {
        chunked_is_recurrent<double>(1e-13, kda);
        chunked_is_recurrent<float>(3e-6, kda);
    }
}

TEST(MlDeltaRule, DecodingIsTheRecurrentFormToTheBit)
{
    constexpr std::size_t n = 5;
    const problem<float> pr(n, 30, false);
    std::vector<float> o(pr.v.size()), s(S, 0.0f), step(S, 0.0f), ot(H * P);
    ml::delta_rule_forward<C>(ref, pr.in(), std::span<float>(o), 1, std::span<float>(s));
    for (std::size_t t = 0; t < n; ++t) {
        const auto rows = [&](const std::vector<float>& x, std::size_t w) { return std::span<const float>(x).subspan(t * w, w); };
        ml::delta_rule_step<C>(ml::delta_rule_inputs<float>{rows(pr.q, G * N), rows(pr.k, G * N), rows(pr.v, H * P), rows(pr.g, H), rows(pr.beta, H)},
                               std::span<float>(ot), std::span<float>(step));
        for (std::size_t i = 0; i < H * P; ++i)
            EXPECT_EQ(ot[i], o[t * H * P + i]) << t;
    }
    EXPECT_EQ(step, s);
}

void backward_matches_central_differences(bool chunked, bool kda)
{
    constexpr std::size_t n = 5, seqs = 2;
    problem<double> pr(n * seqs, 40, kda);
    auto init = uniform<double>(seqs * S, 41);
    const auto wo = uniform<double>(pr.v.size(), 42), ws = uniform<double>(seqs * S, 43);
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    const auto run = [&](std::vector<double>& o, std::vector<double>& s, std::vector<double>& cps) {
        std::vector<double> work(ml::delta_rule_chunk_work(C));
        if (chunked)
            ml::delta_rule_forward_chunked<C>(ref, pr.in(), std::span<double>(o), seqs, std::span<double>(s), std::span<double>(cps),
                                              std::span<double>(work));
        else
            ml::delta_rule_forward<C>(ref, pr.in(), std::span<double>(o), seqs, std::span<double>(s), std::span<double>(cps));
    };
    const auto loss = [&] {
        std::vector<double> o(pr.v.size()), s = init, cps;
        run(o, s, cps);
        return weighted(o, wo) + weighted(s, ws);
    };
    std::vector<double> o(pr.v.size()), s = init, cps(seqs * m * S);
    run(o, s, cps);
    std::vector<double> dq(pr.q.size()), dk(pr.k.size()), dv(pr.v.size()), dg(pr.g.size()), db(pr.beta.size()), ds = ws,
        work(ml::delta_rule_backward_work<C, double>());
    ml::delta_rule_backward<C>(ref, pr.in(), std::span<const double>(wo), ml::delta_rule_gradients<double>{dq, dk, dv, dg, db}, seqs,
                               std::span<const double>(cps), std::span<double>(ds), std::span<double>(work));
    const auto check = [&](std::vector<double>& x, const std::vector<double>& g, const char* what) {
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(g[i], central_difference(x, i, loss), 1e-7) << what << " " << i << (kda ? " per channel" : " per head");
    };
    check(pr.q, dq, "q");
    check(pr.k, dk, "k");
    check(pr.v, dv, "v");
    check(pr.g, dg, "g");
    check(pr.beta, db, "beta");
    check(init, ds, "initial state");
}

TEST(MlDeltaRule, TheBackwardPassMatchesCentralDifferences)
{
    for (const bool kda : {false, true}) {
        backward_matches_central_differences(false, kda);
        backward_matches_central_differences(true, kda);
    }
}

TEST(MlDeltaRule, TheL2NormalisationAndItsBackwardPass)
{
    auto x = uniform<double>(12, 50);
    const auto w = uniform<double>(12, 51);
    std::vector<double> y(12), dx(12);
    ml::l2_normalise_rows(ref, std::span<const double>(x), std::span<double>(y), 4, 1e-6);
    for (std::size_t r = 0; r < 3; ++r)
        EXPECT_NEAR(y[4 * r] * y[4 * r] + y[4 * r + 1] * y[4 * r + 1] + y[4 * r + 2] * y[4 * r + 2] + y[4 * r + 3] * y[4 * r + 3], 1.0, 1e-5);
    ml::l2_normalise_rows_backward(ref, std::span<const double>(x), std::span<const double>(w), std::span<double>(dx), 4, 1e-6);
    const auto loss = [&] {
        std::vector<double> out(12);
        ml::l2_normalise_rows(ref, std::span<const double>(x), std::span<double>(out), 4, 1e-6);
        return weighted(out, w);
    };
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(dx[i], central_difference(x, i, loss), 1e-8);
}

constexpr double compile_time_delta()
{
    // One head of one channel: S = e^0 · 0 + 1 · (2 - 0) · 0.5 = 1, o = 1 · 1 · scale (1)
    constexpr ml::transformer_config c1 = [] {
        auto c = ml::tiny_gated_deltanet;
        c.mixer_heads = c.mixer_head_dim = c.mixer_state = c.mixer_groups = 1;
        c.hybrid_period = c.hybrid_attention_offset = 0;
        return c;
    }();
    const std::array<double, 1> q{1}, k{1}, v{2}, g{0}, beta{0.5};
    std::array<double, 1> o{}, s{};
    ml::delta_rule_step<c1>(ml::delta_rule_inputs<double>{q, k, v, g, beta}, std::span<double>(o), std::span<double>(s));
    return o[0];
}
static_assert(compile_time_delta() == 1.0);

} // namespace
