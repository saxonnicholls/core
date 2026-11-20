//
//  gla_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/gla.hpp>: the recurrent form against a transcription of llama.cpp's
//  GGML_OP_GATED_LINEAR_ATTN (commit 39039aa) to the bit; the chunked form against it within the stated rounding
//  (float and double, a short last chunk, two sequences, key groups shared by two heads, initial states); decoding token
//  by token to the bit; the backward pass against central differences in double from either form's checkpoints;
//  sizes at compile time.
//

#include <snicholls/core/ml/gla.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include "../support/mixer_fixture.hpp"

#include <gtest/gtest.h>

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

// 4 heads of 3 value channels, 2 key groups of 4 key channels, chunks of 3
constexpr ml::transformer_config C = [] {
    auto c = ml::tiny_mamba2;
    c.mixer = ml::mixer_kind::gla;
    c.mixer_heads = 4;
    c.mixer_head_dim = 3;
    c.mixer_state = 4;
    c.mixer_groups = 2;
    c.mixer_chunk = 3;
    return c;
}();
constexpr std::size_t H = 4, P = 3, N = 4, G = 2, S = H * N * P;
static_assert(ml::gla_state_size(C) == S && ml::gla_chunk_work(C) == 3 * 4);
static_assert(ml::mixer_scale_v<C, double> == 0.5);

template <typename T>
struct problem
{
    std::vector<T> q, k, v, gk;
    problem(std::size_t r, std::uint64_t seed)
        : q(uniform<T>(r * G * N, seed)), k(uniform<T>(r * G * N, seed + 1)), v(uniform<T>(r * H * P, seed + 2)),
          gk(uniform<T>(r * H * N, seed + 3, -1.0, -0.01))
    {
    }
    ml::gla_inputs<T> in() const { return {q, k, v, gk}; }
};

// ggml_compute_forward_gla_f32, scalar path, one key head per head: state [i][j], temp = prev · g + v · k, dst += temp · q · scale
std::vector<float> ggml_gla(const std::vector<float>& k, const std::vector<float>& v, const std::vector<float>& q,
                            const std::vector<float>& g, std::vector<float>& state, std::size_t T, std::size_t heads, std::size_t hs, float scale)
{
    const std::size_t ch = heads * hs;
    std::vector<float> dst(T * ch, 0.0f);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t h = 0; h < heads; ++h)
            for (std::size_t i = 0; i < hs; ++i) {
                const std::size_t ti = t * ch + h * hs + i;
                const float k_val = k[ti], q_val = q[ti] * scale, g_val = g[ti];
                for (std::size_t j = 0; j < hs; ++j) {
                    const std::size_t tj = t * ch + h * hs + j, s = h * hs * hs + i * hs + j;
                    const float kv_val = v[tj] * k_val;
                    const float temp_val = state[s] * g_val + kv_val;
                    dst[tj] += temp_val * q_val;
                    state[s] = temp_val;
                }
            }
    return dst;
}

TEST(MlGla, TheRecurrentFormIsLlamaCppsGatedLinearAttentionToTheBit)
{
    constexpr ml::transformer_config square = [] {   // ggml's op: one key head per head, square heads
        auto c = C;
        c.mixer_groups = 4;
        c.mixer_head_dim = 4;
        return c;
    }();
    constexpr std::size_t rows = 6, D = 4 * 4;
    const auto q = uniform<float>(rows * D, 1), k = uniform<float>(rows * D, 2), v = uniform<float>(rows * D, 3),
               gk = uniform<float>(rows * D, 4, -1.0, -0.01);
    std::vector<float> g(gk.size());
    for (std::size_t i = 0; i < g.size(); ++i)
        g[i] = std::exp(gk[i]);
    std::vector<float> o(rows * D), s(4 * 4 * 4, 0.0f), gs(4 * 4 * 4, 0.0f);
    ml::gla_forward<square>(ref, ml::gla_inputs<float>{q, k, v, gk}, std::span<float>(o), 1, std::span<float>(s));
    EXPECT_EQ(o, ggml_gla(k, v, q, g, gs, rows, 4, 4, 0.5f));
    EXPECT_EQ(s, gs);
}

template <typename T>
void chunked_is_recurrent(double bound)
{
    constexpr std::size_t n = 8, seqs = 2;
    const problem<T> pr(n * seqs, 10);
    const auto init = uniform<T>(seqs * S, 20);
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    std::vector<T> o1(pr.v.size()), o2(pr.v.size()), s1 = init, s2 = init, c1(seqs * m * S), c2(seqs * m * S), work(ml::gla_chunk_work(C));
    ml::gla_forward<C>(ref, pr.in(), std::span<T>(o1), seqs, std::span<T>(s1), std::span<T>(c1));
    ml::gla_forward_chunked<C>(ref, pr.in(), std::span<T>(o2), seqs, std::span<T>(s2), std::span<T>(c2), std::span<T>(work));
    EXPECT_LT(max_relative_difference(o1, o2), bound);
    EXPECT_LT(max_relative_difference(s1, s2), bound);
    EXPECT_LT(max_relative_difference(c1, c2), bound);
}

TEST(MlGla, TheChunkedFormIsTheRecurrentFormWithinRounding)
{
    chunked_is_recurrent<double>(1e-13);
    chunked_is_recurrent<float>(3e-6);
}

TEST(MlGla, DecodingIsTheRecurrentFormToTheBit)
{
    constexpr std::size_t n = 5;
    const problem<float> pr(n, 30);
    std::vector<float> o(pr.v.size()), s(S, 0.0f), step(S, 0.0f), ot(H * P);
    ml::gla_forward<C>(ref, pr.in(), std::span<float>(o), 1, std::span<float>(s));
    for (std::size_t t = 0; t < n; ++t) {
        const ml::gla_inputs<float> row{std::span<const float>(pr.q).subspan(t * G * N, G * N), std::span<const float>(pr.k).subspan(t * G * N, G * N),
                                        std::span<const float>(pr.v).subspan(t * H * P, H * P), std::span<const float>(pr.gk).subspan(t * H * N, H * N)};
        ml::gla_step<C>(row, std::span<float>(ot), std::span<float>(step));
        for (std::size_t i = 0; i < H * P; ++i)
            EXPECT_EQ(ot[i], o[t * H * P + i]) << t;
    }
    EXPECT_EQ(step, s);
}

void backward_matches_central_differences(bool chunked)
{
    constexpr std::size_t n = 5, seqs = 2;
    problem<double> pr(n * seqs, 40);
    auto init = uniform<double>(seqs * S, 41);
    const auto wo = uniform<double>(pr.v.size(), 42), ws = uniform<double>(seqs * S, 43);
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    const auto run = [&](std::vector<double>& o, std::vector<double>& s, std::vector<double>& cps) {
        std::vector<double> work(ml::gla_chunk_work(C));
        if (chunked)
            ml::gla_forward_chunked<C>(ref, pr.in(), std::span<double>(o), seqs, std::span<double>(s), std::span<double>(cps), std::span<double>(work));
        else
            ml::gla_forward<C>(ref, pr.in(), std::span<double>(o), seqs, std::span<double>(s), std::span<double>(cps));
    };
    const auto loss = [&] {
        std::vector<double> o(pr.v.size()), s = init, cps;
        run(o, s, cps);
        return weighted(o, wo) + weighted(s, ws);
    };
    std::vector<double> o(pr.v.size()), s = init, cps(seqs * m * S);
    run(o, s, cps);
    std::vector<double> dq(pr.q.size()), dk(pr.k.size()), dv(pr.v.size()), dg(pr.gk.size()), ds = ws, work(ml::gla_backward_work<C, double>());
    ml::gla_backward<C>(ref, pr.in(), std::span<const double>(wo), ml::gla_gradients<double>{dq, dk, dv, dg}, seqs, std::span<const double>(cps),
                        std::span<double>(ds), std::span<double>(work));
    const auto check = [&](std::vector<double>& v, const std::vector<double>& g, const char* what) {
        for (std::size_t i = 0; i < v.size(); ++i)
            EXPECT_NEAR(g[i], central_difference(v, i, loss), 1e-7) << what << " " << i;
    };
    check(pr.q, dq, "q");
    check(pr.k, dk, "k");
    check(pr.v, dv, "v");
    check(pr.gk, dg, "gk");
    check(init, ds, "initial state");
}

TEST(MlGla, TheBackwardPassMatchesCentralDifferences)
{
    backward_matches_central_differences(false);
    backward_matches_central_differences(true);
}

} // namespace
