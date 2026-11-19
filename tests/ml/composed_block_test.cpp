//
//  composed_block_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/composed_block.hpp>: latent attention with one expert chosen with weight 1 is the
//  latent block's dense path to the bit; latent attention with a dense MLP is the latent block, and standard attention
//  with a mixture of experts the MoE block, to the bit; decoding through a latent cache against the training pass within
//  rounding; the backward pass of DeepSeek-V3's block (node-limited routing, a selection bias, both auxiliary losses,
//  two sequences) against central differences in double; the losses by sequence; the buffers' sizes and layouts;
//  DeepSeek-V3's parameter and KV-cache counts against the paper's.
//

#include <snicholls/core/ml/composed_block.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/latent_block.hpp>
#include <snicholls/core/ml/moe_block.hpp>
#include <snicholls/core/stats/random.hpp>

#include "../support/latent_attention_fixture.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
using u32 = std::uint32_t;
using snicholls::core_testing::block_problem;
using snicholls::core_testing::positions;
using snicholls::core_testing::table;
constexpr auto& ref = snicholls::linalg::reference;

template <typename T>
std::vector<T> noise(std::size_t n, std::uint64_t seed, double scale = 1.0)
{
    auto g = stats::wyrand::stream(seed, 0);
    std::vector<T> v(n);
    for (auto& e : v)
        e = static_cast<T>(scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

// DeepSeek-V3's block in miniature: tiny_mla's attention, 4 routed experts in 2 groups (1 kept), top 2, one shared,
// sigmoid scoring scaled by 2.5, both auxiliary losses
constexpr ml::transformer_config v3_like = [] {
    auto c = ml::tiny_mla;
    c.experts = 4;
    c.top_k = 2;
    c.shared_experts = 1;
    c.expert_d_ff = 6;
    c.scoring = ml::moe_scoring::sigmoid;
    c.route_scale = 2.5;
    c.balance_loss_weight = 0.02;
    c.z_loss_weight = 0.01;
    c.expert_groups = 2;
    c.group_top_k = 1;
    return c;
}();
// One expert of tiny_mla's d_ff, chosen with weight exactly 1
constexpr ml::transformer_config one_expert = [] {
    auto c = ml::tiny_mla;
    c.experts = 1;
    c.top_k = 1;
    c.expert_d_ff = c.d_ff;
    return c;
}();
static_assert(ml::valid(v3_like) && ml::valid(one_expert));

// DeepSeek-V3's block's buffers, laid out by the header's own functions
template <ml::transformer_config C, typename T>
struct problem
{
    std::size_t rows, sequences;
    std::vector<T> p, bias, g, xs, values, work;
    std::vector<u32> indices;

    problem(std::size_t n, std::size_t seqs, std::uint64_t seed = 1, bool selection_bias = false)
        : rows(n * seqs), sequences(seqs), p(noise<T>(ml::latent_moe_block_parameter_count(C), seed, 0.4)),
          bias(selection_bias ? noise<T>(C.experts, seed + 1, 0.2) : std::vector<T>{}), g(p.size()),
          xs(noise<T>(rows * C.d_model, seed + 2)), values(ml::latent_moe_block_activation_values(C, rows)),
          work(ml::latent_moe_block_gradient_work(C, rows)), indices(ml::latent_moe_block_activation_indices(C, rows))
    {
        const auto w = weights();
        for (const auto norm : {w.norm1_weight, w.norm2_weight, w.attention.q_norm, w.attention.kv_norm})   // norm weights near 1
            for (std::size_t i = 0; i < norm.size(); ++i)
                p[std::size_t(norm.data() - p.data()) + i] += T(1);
    }
    [[nodiscard]] ml::latent_moe_block_weights<T> weights() const
    {
        return ml::latent_moe_block_weights_in<C>(std::span<const T>(p), std::span<const T>(bias));
    }
    [[nodiscard]] ml::latent_moe_block_activations<T> act()
    {
        return ml::latent_moe_block_activations_in<C>(std::span<T>(values), std::span<u32>(indices), rows);
    }
    [[nodiscard]] ml::latent_moe_block_gradients<T> grads() { return ml::latent_moe_block_gradients_in<C>(std::span<T>(g), std::span<T>(work), rows); }
    // The block over every row, from a copy of xs
    ml::moe_losses<T> forward(std::vector<T>& out)
    {
        out = xs;
        return ml::block_forward_sequence<C>(ref, weights(), std::span<T>(out), table<C, T>(), act(), sequences);
    }
};

template <typename T>
std::vector<T> as_vector(std::span<T> s)
{
    return {s.begin(), s.end()};
}

TEST(MlComposedBlock, LatentAttentionWithOneExpertOfWeightOneIsTheLatentBlockToTheBit)
{
    constexpr auto C = one_expert;
    constexpr std::size_t n = 5, seqs = 2, rows = n * seqs, d = C.d_model, f = C.d_ff;
    problem<C, float> b(n, seqs);
    std::vector<float> ys;
    const auto losses = b.forward(ys);
    EXPECT_EQ(losses.balance, 0.0f);
    EXPECT_EQ(losses.z, 0.0f);
    auto dx = noise<float>(rows * d, 9), dense_dx = dx;
    ml::block_backward<C>(ref, b.weights(), table<C, float>(), b.act(), std::span<float>(dx), b.grads(), seqs);

    // The latent block with the expert's MLP, through the same backend, into fresh buffers
    constexpr auto D = ml::tiny_mla;
    const auto w = b.weights();
    const ml::mlp_weights<ml::dense_weights<float>, float> mlp{w.mlp.experts.up.expert(0), w.mlp.experts.gate.expert(0),
                                                               w.mlp.experts.down.expert(0)};
    const ml::latent_block_weights<ml::dense_weights<float>, float> lw{w.norm1_weight, {}, w.attention, w.norm2_weight, {}, mlp};
    std::vector<float> x0(rows * d), h1(rows * d), x1(rows * d), h2(rows * d), up(rows * f), gate(rows * f), hidden(rows * f),
        avalues(ml::latent_attention_activation_values(D, rows)), awork(ml::latent_attention_gradient_work(D, rows)),
        ga(ml::latent_attention_parameter_count(D)), gn1(d), gn2(d), gup(f * d), ggate(f * d), gdown(d * f), mlp_scratch(2 * rows * f),
        scratch(rows * d);
    const ml::latent_block_activations<float> act{x0, h1, x1, h2, ml::latent_attention_activations_in<D>(std::span<float>(avalues), rows),
                                                  up, gate, hidden};
    ml::mlp_gradients<float> mg{};
    mg.up = gup;
    mg.gate = ggate;
    mg.down = gdown;
    mg.scratch = mlp_scratch;
    const ml::latent_block_gradients<float> grads{gn1, {}, gn2, {}, ml::latent_attention_gradients_in<D>(std::span<float>(ga), std::span<float>(awork), rows),
                                                  mg, scratch};
    auto dense_ys = b.xs;
    ml::block_forward_sequence<D>(ref, lw, std::span<float>(dense_ys), table<D, float>(), act, seqs);
    ml::block_backward<D>(ref, lw, table<D, float>(), act, std::span<float>(dense_dx), grads, seqs);

    EXPECT_EQ(ys, dense_ys);
    EXPECT_EQ(dx, dense_dx);
    const auto gb = b.grads();
    EXPECT_EQ(as_vector(gb.norm1_weight), gn1);
    EXPECT_EQ(as_vector(gb.norm2_weight), gn2);
    EXPECT_EQ(std::vector<float>(b.g.begin() + d, b.g.begin() + std::ptrdiff_t(d + ga.size())), ga);   // the whole attention
    EXPECT_EQ(as_vector(gb.mlp.experts.up), gup);
    EXPECT_EQ(as_vector(gb.mlp.experts.gate), ggate);
    EXPECT_EQ(as_vector(gb.mlp.experts.down), gdown);
    EXPECT_TRUE(std::all_of(gb.mlp.router.begin(), gb.mlp.router.end(), [](float v) { return v == 0.0f; }));   // no routing gradient
}

TEST(MlComposedBlock, LatentAttentionWithADenseMlpIsTheLatentBlockToTheBit)
{
    constexpr auto C = ml::tiny_mla;
    constexpr std::size_t d = C.d_model, f = C.d_ff;
    block_problem<C, double> a(4, 2), b(4, 2);
    const auto want = a.forward();
    auto da = a.dxs;
    ml::block_backward<C>(ref, a.weights(), table<C, double>(), a.act(), std::span<double>(da), a.grads(), 2);

    using W = ml::composed_block_weights<ml::latent_attention_weights<ml::dense_weights<double>, double>,
                                         ml::mlp_weights<ml::dense_weights<double>, double>, double>;
    const W w{b.norm1, {}, b.attn.weights(), b.norm2, {}, {{b.up, f, d}, {b.gate, f, d}, {b.down, d, f}}};
    const ml::composed_block_activations<ml::latent_attention_activations<double>, ml::mlp_activations<double>, double> act{
        b.x0, b.h1, b.x1, b.h2, b.attn.act(), {b.mup, b.mgate, b.mhidden}};
    const auto lg = b.grads();
    const ml::composed_block_gradients<ml::latent_attention_gradients<double>, ml::mlp_gradients<double>, double> grads{
        b.g_norm1, {}, b.g_norm2, {}, lg.attention, lg.mlp, b.scratch};
    auto ys = b.xs, db = b.dxs;
    const auto losses = ml::block_forward_sequence<C>(ref, w, std::span<double>(ys), table<C, double>(), act, 2);
    ml::block_backward<C>(ref, w, table<C, double>(), act, std::span<double>(db), grads, 2);
    EXPECT_EQ(losses.balance + losses.z, 0.0);
    EXPECT_EQ(ys, want);
    EXPECT_EQ(db, da);
    EXPECT_EQ(b.g_norm1, a.g_norm1);
    EXPECT_EQ(b.g_norm2, a.g_norm2);
    EXPECT_EQ(b.g_up, a.g_up);
    EXPECT_EQ(b.g_down, a.g_down);
    EXPECT_EQ(b.attn.g_kv_up, a.attn.g_kv_up);
    EXPECT_EQ(b.attn.g_q_down, a.attn.g_q_down);
}

// Standard attention (GQA, RoPE) with a mixture of experts, as moe_block_test's configuration
constexpr ml::transformer_config standard_moe{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 6, .mlp = ml::mlp_kind::swiglu,
                                              .rope = ml::rope_kind::half, .experts = 4, .top_k = 2, .shared_experts = 1, .expert_d_ff = 6,
                                              .balance_loss_weight = 0.02, .z_loss_weight = 0.01};

// The buffers of a standard-attention, MoE block: weights shared, everything else its own
struct standard_moe_buffers
{
    static constexpr auto C = standard_moe;
    static constexpr std::size_t d = C.d_model, q = C.q_size(), kv = C.kv_size();
    std::size_t rows;
    std::vector<double> aq, ak, av, ay, lse, dq, dk, dv, dy, x0, h1, x1, h2, scratch, mv, mw, gattn, gn1, gn2, gmoe;
    std::vector<u32> mi;
    explicit standard_moe_buffers(std::size_t n)
        : rows(n), aq(n * q), ak(n * kv), av(n * kv), ay(n * q), lse(n * C.n_heads), dq(n * q), dk(n * kv), dv(n * kv), dy(n * q), x0(n * d),
          h1(n * d), x1(n * d), h2(n * d), scratch(n * d), mv(ml::moe_activation_values(C, n)), mw(ml::moe_gradient_work(C, n)),
          gattn(2 * q * d + 2 * kv * d), gn1(d), gn2(d), gmoe(ml::moe_parameter_count(C)), mi(ml::moe_activation_indices(C, n))
    {
    }
    ml::attention_activations<double> attention() { return {aq, ak, av, {}, {}, ay, lse}; }
    ml::attention_gradients<double> attention_grads()
    {
        const std::span<double> g(gattn);
        return {dq, dk, dv, g.first(q * d), g.subspan(q * d, kv * d), g.subspan(q * d + kv * d, kv * d), g.subspan(q * d + 2 * kv * d, d * q),
                {}, {}, {}, {}, {}, {}, dy};
    }
    ml::moe_activations<double> moe() { return ml::moe_activations_in<C>(std::span<double>(mv), std::span<u32>(mi), rows); }
    ml::moe_gradients<double> moe_grads() { return ml::moe_gradients_in<C>(std::span<double>(gmoe), std::span<double>(mw), rows); }
};

TEST(MlComposedBlock, StandardAttentionWithAMixtureOfExpertsIsTheMoeBlockToTheBit)
{
    constexpr auto C = standard_moe;
    constexpr std::size_t n = 6, seqs = 2, d = C.d_model, q = C.q_size(), kv = C.kv_size();
    const auto p = noise<double>(2 * q * d + 2 * kv * d, 3, 0.4), moe = noise<double>(ml::moe_parameter_count(C), 4, 0.4);
    const auto n1 = noise<double>(d, 5, 0.2), n2 = noise<double>(d, 6, 0.2), xs = noise<double>(n * d, 7), dout = noise<double>(n * d, 8);
    const std::span<const double> ps(p);
    const ml::attention_weights<ml::dense_weights<double>, double> attn{
        {ps.first(q * d), q, d}, {ps.subspan(q * d, kv * d), kv, d}, {ps.subspan(q * d + kv * d, kv * d), kv, d}, {ps.subspan(q * d + 2 * kv * d, d * q), d, q}};
    const auto mw = ml::moe_weights_in<C>(std::span<const double>(moe));

    standard_moe_buffers a(n), b(n);
    const ml::moe_block_weights<double> wa{n1, {}, attn, n2, {}, mw};
    static const ml::rope_table<double, 16, 4> rope{};
    auto ya2 = xs, da = dout;
    const ml::moe_block_activations<double> acta{a.x0, a.h1, a.x1, a.h2, a.attention(), a.moe()};
    const auto lossa = ml::block_forward_sequence<C>(ref, wa, std::span<double>(ya2), rope, acta, seqs);
    ml::block_backward<C>(ref, wa, rope, acta, std::span<double>(da), ml::moe_block_gradients<double>{a.gn1, {}, a.gn2, {}, a.attention_grads(), a.moe_grads(), a.scratch}, seqs);

    const ml::composed_block_weights<ml::attention_weights<ml::dense_weights<double>, double>, ml::moe_weights<double>, double> wb{n1, {}, attn, n2, {}, mw};
    const ml::composed_block_activations<ml::attention_activations<double>, ml::moe_activations<double>, double> actb{b.x0, b.h1, b.x1, b.h2,
                                                                                                                       b.attention(), b.moe()};
    auto yb = xs, db = dout;
    const auto lossb = ml::block_forward_sequence<C>(ref, wb, std::span<double>(yb), rope, actb, seqs);
    ml::block_backward<C>(ref, wb, rope, actb, std::span<double>(db),
                          ml::composed_block_gradients<ml::attention_gradients<double>, ml::moe_gradients<double>, double>{
                              b.gn1, {}, b.gn2, {}, b.attention_grads(), b.moe_grads(), b.scratch},
                          seqs);
    EXPECT_EQ(lossa.balance, lossb.balance);
    EXPECT_EQ(lossa.z, lossb.z);
    EXPECT_GT(lossb.balance, 0.0);
    EXPECT_EQ(ya2, yb);
    EXPECT_EQ(da, db);
    EXPECT_EQ(a.gattn, b.gattn);
    EXPECT_EQ(a.gn1, b.gn1);
    EXPECT_EQ(a.gn2, b.gn2);
    EXPECT_EQ(a.gmoe, b.gmoe);
}

TEST(MlComposedBlock, DecodingThroughALatentCacheIsTheTrainingPassWithinRounding)
{
    constexpr auto C = v3_like;
    constexpr std::size_t n = 6, d = C.d_model;
    problem<C, double> b(n, 1, 11, true);
    std::vector<double> trained;
    b.forward(trained);
    std::vector<double> storage(ml::latent_kv_cache_view<C, double>::required_size(1, positions));
    const ml::latent_kv_cache_view<C, double> cache(storage, 1, positions);
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<double> x(b.xs.begin() + std::ptrdiff_t(t * d), b.xs.begin() + std::ptrdiff_t((t + 1) * d));
        ml::block_forward<C>(b.weights(), x, t, cache, 0, table<C, double>());
        for (std::size_t i = 0; i < d; ++i)
            EXPECT_NEAR(x[i], trained[t * d + i], 1e-12 * (1 + std::abs(x[i]))) << "token " << t << " element " << i;
    }
}

TEST(MlComposedBlock, BackwardOfDeepSeekV3sBlockMatchesCentralDifferences)
{
    constexpr auto C = v3_like;
    constexpr std::size_t n = 3, seqs = 2;
    problem<C, double> b(n, seqs, 21, true);
    const auto dout = noise<double>(n * seqs * C.d_model, 22);
    std::vector<double> ys;
    b.forward(ys);
    const auto chosen = as_vector(b.act().mlp.experts);
    auto dx = dout;
    ml::block_backward<C>(ref, b.weights(), table<C, double>(), b.act(), std::span<double>(dx), b.grads(), seqs);
    const auto gx = dx, gp = b.g;
    problem<C, double> m(n, seqs, 21, true);
    const auto loss = [&] {
        std::vector<double> y;
        const auto aux = m.forward(y);
        EXPECT_EQ(as_vector(m.act().mlp.experts), chosen);   // the step is too small to change a choice
        double s = aux.balance + aux.z;
        for (std::size_t i = 0; i < y.size(); ++i)
            s += dout[i] * y[i];
        return s;
    };
    constexpr double h = 1e-6;
    const auto check = [&](std::vector<double>& v, const std::vector<double>& g, const char* what) {
        for (std::size_t j = 0; j < v.size(); ++j) {
            const double v0 = v[j];
            v[j] = v0 + h;
            const double up = loss();
            v[j] = v0 - h;
            const double down = loss();
            v[j] = v0;
            EXPECT_NEAR(g[j], (up - down) / (2 * h), 2e-7 * std::max(1.0, std::abs(g[j]))) << what << ' ' << j;
        }
    };
    check(m.xs, gx, "input");
    check(m.p, gp, "parameter");
}

TEST(MlComposedBlock, TheForwardReturnsTheMixtureOfExpertsLossesOverItsSequences)
{
    constexpr auto C = v3_like;
    problem<C, double> b(4, 2, 31);
    std::vector<double> ys;
    const auto losses = b.forward(ys);
    const auto a = b.act().mlp;
    EXPECT_EQ(losses.balance, ml::moe_balance_loss<C>(ref, std::span<const double>(a.logits), a.experts, std::span<double>{}, 2));
    EXPECT_EQ(losses.z, ml::moe_z_loss<C>(ref, std::span<const double>(a.logits), std::span<double>{}));
    EXPECT_GT(losses.balance, 0.0);
    // Sequence-wise: the mean of each sequence's own loss
    const std::span<const double> logits(a.logits);
    const std::span<const u32> experts(a.experts);
    const double first = ml::moe_balance_loss<C>(ref, logits.first(4 * C.experts), experts.first(4 * C.top_k), std::span<double>{});
    const double second = ml::moe_balance_loss<C>(ref, logits.last(4 * C.experts), experts.last(4 * C.top_k), std::span<double>{});
    EXPECT_NEAR(losses.balance, (first + second) / 2, 1e-15);
}

// tiny_mla's latent attention by hand: q_down 6x16 and its norm, q_up 12x6, kv_down 10x16 and its norm, kv_up 16x8, o 16x8
static_assert(ml::latent_attention_parameter_count(ml::tiny_mla) == 96 + 6 + 72 + 160 + 8 + 128 + 128);
static_assert(ml::latent_attention_activation_values(ml::tiny_mla, 3) == 3 * (12 + 12 + 10 + 8 + 2 + 16 + 8 + 2));
static_assert(ml::latent_attention_gradient_work(ml::tiny_mla, 3) == ml::latent_attention_activation_values(ml::tiny_mla, 3) - 3 * 2);
static_assert(ml::latent_moe_block_parameter_count(v3_like) == 2 * 16 + 598 + ml::moe_parameter_count(v3_like));
static_assert(ml::latent_moe_block_activation_values(v3_like, 5)
              == 4 * 5 * 16 + ml::latent_attention_activation_values(v3_like, 5) + ml::moe_activation_values(v3_like, 5));

TEST(MlComposedBlock, TheLayoutsCoverTheirBuffersEndToEnd)
{
    constexpr auto C = v3_like;
    problem<C, float> b(3, 2);
    const auto w = b.weights();
    EXPECT_EQ(w.norm1_weight.data(), b.p.data());
    EXPECT_TRUE(w.norm1_bias.empty() && w.norm2_bias.empty());
    EXPECT_EQ(w.attention.q_down.data.data(), b.p.data() + C.d_model);
    EXPECT_EQ(w.norm2_weight.data(), w.attention.o.data.data() + C.d_model * C.n_heads * C.v_head_dim);
    EXPECT_EQ(w.mlp.router.data.data(), w.norm2_weight.data() + C.d_model);
    EXPECT_EQ(w.mlp.shared.down.data.data() + w.mlp.shared.down.data.size(), b.p.data() + b.p.size());
    EXPECT_EQ(w.mlp.selection_bias.size(), 0u);
    EXPECT_EQ(w.attention.kv_up.rows, C.n_heads * (C.qk_nope_dim() + C.v_head_dim));
    const auto a = b.act();
    EXPECT_EQ(a.x0.data(), b.values.data());
    EXPECT_EQ(a.attention.cq_raw.data(), a.h2.data() + a.h2.size());
    EXPECT_EQ(a.mlp.logits.data(), a.attention.lse.data() + a.attention.lse.size());
    EXPECT_EQ(a.mlp.shared_y.data() + a.mlp.shared_y.size(), b.values.data() + b.values.size());
    const auto g = b.grads();
    EXPECT_EQ(g.attention.q_down.data(), b.g.data() + C.d_model);
    EXPECT_EQ(g.attention.y.data(), b.work.data());
    EXPECT_EQ(g.scratch.data() + g.scratch.size(), b.work.data() + b.work.size());
    EXPECT_EQ(g.mlp.shared.down_bias.data() + g.mlp.shared.down_bias.size(), b.g.data() + b.g.size());
    // Without query compression q_down is empty and q_up reads d_model
    constexpr auto direct = [] {
        auto c = v3_like;
        c.q_lora_rank = 0;
        return c;
    }();
    std::vector<float> params(ml::latent_attention_parameter_count(direct));
    const auto lw = ml::latent_attention_weights_in<direct>(std::span<const float>(params));
    EXPECT_TRUE(lw.q_down.data.empty() && lw.q_norm.empty());
    EXPECT_EQ(lw.q_up.cols, direct.d_model);
    EXPECT_EQ(lw.q_up.data.data(), params.data());
}

// DeepSeek-V3 (section 4.2, config.json): 61 blocks, the first 3 with a dense MLP of 18,432, a vocabulary of 129,280,
// untied embedding and head, one MTP module. The paper's 671B total and 37B active are the main model's.
constexpr std::size_t v3_vocabulary = 129280, v3_blocks = 61, v3_dense_blocks = 3;
constexpr auto v3 = ml::deepseek_v3;
constexpr std::size_t v3_dense_block = 2 * v3.d_model + ml::latent_attention_parameter_count(v3) + 3 * v3.d_model * v3.d_ff;
constexpr std::size_t v3_embeddings = 2 * v3_vocabulary * v3.d_model + v3.d_model;   // embedding, head, final norm
constexpr std::size_t v3_total = (v3_blocks - v3_dense_blocks) * ml::latent_moe_block_parameter_count(v3) + v3_dense_blocks * v3_dense_block
                                 + v3_embeddings;
constexpr std::size_t v3_idle_experts = (v3.experts - v3.top_k) * 3 * v3.expert_d_ff * v3.d_model;   // per MoE block
constexpr std::size_t v3_active = v3_total - (v3_blocks - v3_dense_blocks) * v3_idle_experts;
constexpr std::size_t v3_mtp_module = 3 * v3.d_model + 2 * v3.d_model * v3.d_model + ml::latent_moe_block_parameter_count(v3);
static_assert(ml::valid(v3) && v3.latent() && v3.moe() && v3.expert_groups == 8 && v3.group_top_k == 4);
static_assert(ml::latent_attention_parameter_count(v3) == 187107328);
static_assert(ml::moe_parameter_count(v3) == 11320164352);
static_assert(ml::latent_moe_block_parameter_count(v3) == 11507286016);
static_assert(v3_total == 671026404352);                        // the paper's 671B
static_assert(v3_active == 37552282624);                        // the paper's 37B: 35.7B without the embedding and head
static_assert(v3_mtp_module == 11610067968);                    // 685B released = 671B + this + its own copy of embedding and head
static_assert(v3_total + v3_mtp_module + 2 * v3_vocabulary * v3.d_model == 684489830400);
static_assert(ml::kv_cache_elements(v3, v3_blocks, 1) == 576 * 61);   // 35,136 numbers a token, against 2 · 128 · 192 · 61 for MHA
static_assert(ml::kv_cache_elements(v3, v3_blocks, 131072) == std::size_t{35136} * 131072);

TEST(MlComposedBlock, DeepSeekV3sCountsAreThePapers)
{
    EXPECT_NEAR(double(v3_total) / 1e9, 671.0, 0.5);
    EXPECT_NEAR(double(v3_active) / 1e9, 37.0, 0.6);
}

} // namespace
