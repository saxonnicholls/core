//
//  composed_block_hybrid_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/composed_block.hpp> with a linear-time mixer as the attention part (TRAINING.md T15):
//  the backward pass of a Mamba-2 block and of a gated-delta-rule block against central differences in double; hybrid
//  stacks laid out by layer_is_attention (Qwen3-Next's three gated-delta-rule layers then attention, and Jamba's
//  pattern with Mamba-2), decoded token by token through a hybrid cache against the batched pass at every position;
//  a small hybrid stack trained to copy the previous token, its loss falling.
//

#include <snicholls/core/ml/composed_block.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/rope.hpp>

#include "../support/mixer_fixture.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::core_testing::central_difference;
using snicholls::core_testing::max_relative_difference;
using snicholls::core_testing::uniform;
using snicholls::core_testing::weighted;
constexpr auto& ref = snicholls::linalg::reference;

constexpr std::size_t positions = 16;
static const ml::rope_table<double, positions, 4> rope{};

// A dense SwiGLU MLP's buffers
template <ml::transformer_config C>
struct mlp_part
{
    static constexpr std::size_t d = C.d_model, f = C.d_ff, count = 3 * f * d;
    std::vector<double> up, gate, hidden, scratch;
    static ml::mlp_weights<ml::dense_weights<double>, double> weights(std::span<const double> p)
    {
        return {{p.first(f * d), f, d}, {p.subspan(f * d, f * d), f, d}, {p.subspan(2 * f * d, d * f), d, f}, {}, {}, {}};
    }
    void size(std::size_t rows)
    {
        up.assign(rows * f, 0.0);
        gate.assign(rows * f, 0.0);
        hidden.assign(rows * f, 0.0);
        scratch.assign(2 * rows * f, 0.0);
    }
    ml::mlp_activations<double> act() { return {up, gate, hidden}; }
    ml::mlp_gradients<double> grads(std::span<double> g)
    {
        return {g.first(f * d), g.subspan(f * d, f * d), g.subspan(2 * f * d, d * f), {}, {}, {}, scratch};
    }
};

// A composed block whose attention part is the configuration's mixer
template <ml::transformer_config C>
struct mixer_block
{
    static constexpr std::size_t d = C.d_model, mixer = ml::mixer_parameter_count(C), count = 2 * d + mixer + mlp_part<C>::count;
    std::vector<double> values, work;
    mlp_part<C> mlp;
    std::vector<double> x0, h1, x1, h2, scratch;
    std::size_t rows = 0, seqs = 1;

    static auto weights(std::span<const double> p)
    {
        return ml::composed_block_weights<decltype(ml::mixer_weights_in<C>(p.subspan(d, mixer))), ml::mlp_weights<ml::dense_weights<double>, double>,
                                          double>{p.first(d), {}, ml::mixer_weights_in<C>(p.subspan(d, mixer)), p.subspan(d + mixer, d), {},
                                                  mlp_part<C>::weights(p.subspan(2 * d + mixer))};
    }
    void size(std::size_t r, std::size_t s)
    {
        rows = r;
        seqs = s;
        values.assign(ml::mixer_activation_values(C, r, s), 0.0);
        work.assign(ml::mixer_gradient_work<C, double>(r, s), 0.0);
        mlp.size(r);
        for (auto* v : {&x0, &h1, &x1, &h2, &scratch})
            v->assign(r * d, 0.0);
    }
    auto act()
    {
        return ml::composed_block_activations<ml::mixer_activations<double>, ml::mlp_activations<double>, double>{
            x0, h1, x1, h2, ml::mixer_activations_in<C>(std::span<double>(values), rows, seqs), mlp.act()};
    }
    auto grads(std::span<double> g)
    {
        return ml::composed_block_gradients<ml::mixer_gradients<double>, ml::mlp_gradients<double>, double>{
            g.first(d), {}, g.subspan(d + mixer, d), {}, ml::mixer_gradients_in<C>(g.subspan(d, mixer), std::span<double>(work)),
            mlp.grads(g.subspan(2 * d + mixer)), scratch};
    }
};

// A composed block with standard attention (RoPE, the configuration's heads)
template <ml::transformer_config C>
struct attention_block
{
    static constexpr std::size_t d = C.d_model, q = C.q_size(), kv = C.kv_size(), attn = 2 * q * d + 2 * kv * d;
    static constexpr std::size_t count = 2 * d + attn + mlp_part<C>::count;
    mlp_part<C> mlp;
    std::vector<double> x0, h1, x1, h2, scratch, aq, ak, av, ay, lse, dq, dk, dv, dy;
    std::size_t rows = 0;

    static auto weights(std::span<const double> p)
    {
        const auto a = p.subspan(d, attn);
        const ml::attention_weights<ml::dense_weights<double>, double> w{{a.first(q * d), q, d}, {a.subspan(q * d, kv * d), kv, d},
                                                                          {a.subspan(q * d + kv * d, kv * d), kv, d},
                                                                          {a.subspan(q * d + 2 * kv * d, d * q), d, q}};
        return ml::composed_block_weights<ml::attention_weights<ml::dense_weights<double>, double>, ml::mlp_weights<ml::dense_weights<double>, double>,
                                          double>{p.first(d), {}, w, p.subspan(d + attn, d), {}, mlp_part<C>::weights(p.subspan(2 * d + attn))};
    }
    void size(std::size_t r, std::size_t)
    {
        rows = r;
        mlp.size(r);
        for (auto* v : {&x0, &h1, &x1, &h2, &scratch})
            v->assign(r * d, 0.0);
        for (auto* v : {&aq, &ay, &dq, &dy})
            v->assign(r * q, 0.0);
        for (auto* v : {&ak, &av, &dk, &dv})
            v->assign(r * kv, 0.0);
        lse.assign(r * C.n_heads, 0.0);
    }
    auto act()
    {
        return ml::composed_block_activations<ml::attention_activations<double>, ml::mlp_activations<double>, double>{
            x0, h1, x1, h2, {aq, ak, av, {}, {}, ay, lse}, mlp.act()};
    }
    auto grads(std::span<double> g)
    {
        const auto a = g.subspan(d, attn);
        return ml::composed_block_gradients<ml::attention_gradients<double>, ml::mlp_gradients<double>, double>{
            g.first(d),
            {},
            g.subspan(d + attn, d),
            {},
            {dq, dk, dv, a.first(q * d), a.subspan(q * d, kv * d), a.subspan(q * d + kv * d, kv * d), a.subspan(q * d + 2 * kv * d, d * q), {}, {}, {},
             {}, {}, {}, dy},
            mlp.grads(g.subspan(2 * d + attn)),
            scratch};
    }
};

// The mixer's own norm weights near 1 as well; attention has none
template <ml::transformer_config C>
void mixer_norm_near_one(const mixer_block<C>&, std::vector<double>& p)
{
    const auto w = mixer_block<C>::weights(std::span<const double>(p)).attention;
    const auto at = static_cast<std::size_t>(w.norm.data() - p.data());
    for (std::size_t j = 0; j < w.norm.size(); ++j)
        p[at + j] += 1.0;
}
template <ml::transformer_config C>
void mixer_norm_near_one(const attention_block<C>&, std::vector<double>&)
{
}

// A stack of L composed blocks, each attention or the mixer as layer_is_attention(C, i) says
template <ml::transformer_config C, std::size_t L>
struct stack
{
    template <std::size_t... I>
    static auto make(std::index_sequence<I...>)
        -> std::tuple<std::conditional_t<ml::layer_is_attention(C, I), attention_block<C>, mixer_block<C>>...>;
    using blocks_t = decltype(make(std::make_index_sequence<L>{}));
    blocks_t blocks;
    std::vector<std::vector<double>> params, grads;

    explicit stack(std::uint64_t seed)
    {
        std::apply(
            [&](auto&... b) {
                (params.push_back(uniform<double>(std::remove_reference_t<decltype(b)>::count, seed++, -0.3, 0.3)), ...);
            },
            blocks);
        for (auto& p : params) {   // the norms' weights near 1
            for (std::size_t i = 0; i < C.d_model; ++i)
                p[i] += 1.0;
        }
        for_each([&](auto& b, std::size_t i) { mixer_norm_near_one(b, params[i]); });
        for (const auto& p : params)
            grads.emplace_back(p.size(), 0.0);
    }
    template <typename F>
    void for_each(F&& f)
    {
        std::size_t i = 0;
        std::apply([&](auto&... b) { (f(b, i++), ...); }, blocks);
    }
    void forward(std::vector<double>& xs, std::size_t seqs)
    {
        const std::size_t rows = xs.size() / C.d_model;
        for_each([&](auto& b, std::size_t i) {
            b.size(rows, seqs);
            ml::block_forward_sequence<C>(ref, b.weights(std::span<const double>(params[i])), std::span<double>(xs), rope, b.act(), seqs);
        });
    }
    void backward(std::vector<double>& dxs, std::size_t seqs)
    {
        for (auto& g : grads)
            std::fill(g.begin(), g.end(), 0.0);
        backward_from<L>(dxs, seqs);
    }
    template <std::size_t I>
    void backward_from(std::vector<double>& dxs, std::size_t seqs)   // the last block first
    {
        if constexpr (I > 0) {
            auto& b = std::get<I - 1>(blocks);
            ml::block_backward<C>(ref, b.weights(std::span<const double>(params[I - 1])), rope, b.act(), std::span<double>(dxs),
                                  b.grads(std::span<double>(grads[I - 1])), seqs);
            backward_from<I - 1>(dxs, seqs);
        }
    }
};

template <ml::transformer_config C>
void one_block_backward_matches_central_differences()
{
    constexpr ml::transformer_config M = [] {
        auto c = C;
        c.hybrid_period = c.hybrid_attention_offset = 0;
        return c;
    }();
    constexpr std::size_t n = 5, seqs = 2;
    stack<M, 1> s(7);
    auto xs = uniform<double>(n * seqs * M.d_model, 8);
    const auto wo = uniform<double>(xs.size(), 9);
    const auto loss = [&] {
        auto x = xs;
        s.forward(x, seqs);
        return weighted(x, wo);
    };
    auto x = xs;
    s.forward(x, seqs);
    auto dx = wo;
    s.backward(dx, seqs);
    for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_NEAR(dx[i], central_difference(xs, i, loss), 1e-7) << "input " << i;
    for (std::size_t i = 0; i < s.params[0].size(); ++i)
        EXPECT_NEAR(s.grads[0][i], central_difference(s.params[0], i, loss), 1e-7) << "parameter " << i;
}

TEST(MlComposedBlockHybrid, AMamba2BlocksBackwardMatchesCentralDifferences)
{
    one_block_backward_matches_central_differences<ml::tiny_mamba2>();
}

TEST(MlComposedBlockHybrid, AGatedDeltaRuleBlocksBackwardMatchesCentralDifferences)
{
    one_block_backward_matches_central_differences<ml::tiny_gated_deltanet>();
}

template <ml::transformer_config C, std::size_t L>
void decoding_is_the_batched_pass()
{
    constexpr std::size_t n = 7, attention_layers = ml::hybrid_attention_index(C, L), mixer_layers = L - attention_layers;
    static_assert(attention_layers > 0 && mixer_layers > 0);
    stack<C, L> s(11);
    const auto xs = uniform<double>(n * C.d_model, 12);
    auto batched = xs;
    s.forward(batched, 1);
    std::vector<double> k(ml::kv_cache_view<C, double>::required_size(attention_layers, n)), v(k.size());
    std::vector<double> states(mixer_layers * ml::mixer_state_values(C), 0.0);
    const ml::hybrid_cache<ml::kv_cache_view<C, double>, double> cache{ml::kv_cache_view<C, double>(k, v, attention_layers, n),
                                                                       {std::span<double>(states), ml::mixer_state_values(C)}};
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<double> x(xs.begin() + static_cast<std::ptrdiff_t>(t * C.d_model), xs.begin() + static_cast<std::ptrdiff_t>((t + 1) * C.d_model));
        s.for_each([&](auto& b, std::size_t i) {
            ml::block_forward<C>(b.weights(std::span<const double>(s.params[i])), x, t, cache, i, rope);
        });
        EXPECT_LT(max_relative_difference(x, std::span<const double>(batched).subspan(t * C.d_model, C.d_model)), 1e-12) << t;
    }
}

TEST(MlComposedBlockHybrid, AQwen3NextStackDecodesAsItsBatchedPass)
{
    decoding_is_the_batched_pass<ml::tiny_gated_deltanet, 4>();   // three gated-delta-rule layers, then attention
}

TEST(MlComposedBlockHybrid, AJambaPatternWithMamba2DecodesAsItsBatchedPass)
{
    constexpr ml::transformer_config jamba_like = [] {   // attention at 4 of every 8 layers (Jamba's period and offset), Mamba-2 elsewhere
        auto c = ml::tiny_mamba2;
        c.hybrid_period = 8;
        c.hybrid_attention_offset = 4;
        return c;
    }();
    decoding_is_the_batched_pass<jamba_like, 6>();
}

TEST(MlComposedBlockHybrid, AHybridStackLearnsToCopyThePreviousToken)
{
    // Qwen3-Next's pattern, four layers; the target at each position is the input one position back, so the stack must
    // carry information along the sequence
    constexpr ml::transformer_config C = ml::tiny_gated_deltanet;
    constexpr std::size_t n = 8, seqs = 2, d = C.d_model;
    stack<C, 4> s(21);
    const auto xs = uniform<double>(n * seqs * d, 22);
    std::vector<double> target(xs.size(), 0.0);
    for (std::size_t q = 0; q < seqs; ++q)
        for (std::size_t t = 1; t < n; ++t)
            for (std::size_t i = 0; i < d; ++i)
                target[(q * n + t) * d + i] = xs[(q * n + t - 1) * d + i];
    const auto step = [&] {
        auto x = xs;
        s.forward(x, seqs);
        double loss = 0;
        std::vector<double> dx(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            dx[i] = (x[i] - target[i]) / static_cast<double>(n * seqs);
            loss += 0.5 * (x[i] - target[i]) * (x[i] - target[i]) / static_cast<double>(n * seqs);
        }
        s.backward(dx, seqs);
        for (std::size_t l = 0; l < s.params.size(); ++l)
            for (std::size_t i = 0; i < s.params[l].size(); ++i)
                s.params[l][i] -= 0.05 * s.grads[l][i];
        return loss;
    };
    const double first = step();
    double last = first;
    for (int i = 0; i < 60; ++i)
        last = step();
    EXPECT_LT(last, 0.5 * first) << "first " << first << ", last " << last;
}

} // namespace
