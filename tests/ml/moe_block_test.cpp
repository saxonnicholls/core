//
//  moe_block_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/moe_block.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
using u32 = std::uint32_t;
constexpr auto& ref = snicholls::linalg::reference;

template <typename T>
std::vector<T> noise(std::size_t n, std::uint64_t seed, double scale = 1.0, double offset = 0.0)
{
    auto g = stats::wyrand::stream(seed, 0);
    std::vector<T> v(n);
    for (auto& e : v)
        e = static_cast<T>(offset + scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

// A block with GQA, RoPE, RMS norm and an MoE MLP: 4 experts, top 2, one shared, both losses
constexpr ml::transformer_config moe_cfg{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 6, .mlp = ml::mlp_kind::swiglu,
                                         .rope = ml::rope_kind::half, .experts = 4, .top_k = 2, .shared_experts = 1, .expert_d_ff = 6,
                                         .balance_loss_weight = 0.02, .z_loss_weight = 0.01};
// One expert, chosen with weight exactly 1: the MoE block is the dense block to the bit
constexpr ml::transformer_config one_expert{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 12, .norm = ml::norm_kind::layer,
                                            .mlp = ml::mlp_kind::gelu_tanh, .bias = true, .experts = 1, .top_k = 1, .expert_d_ff = 12};
constexpr ml::transformer_config dense_of_one = [] {
    auto c = one_expert;
    c.experts = c.top_k = c.expert_d_ff = 0;
    return c;
}();

static const ml::rope_table<double, 16, 4> rope_d{};

template <ml::transformer_config C, typename T>
const auto& rope_for()
{
    static_assert(C.rope == ml::rope_kind::none || std::is_same_v<T, double>, "the tests rotate in double only");
    if constexpr (C.rope == ml::rope_kind::none)
        return ml::no_rope;
    else
        return rope_d;
}

// Every parameter of the block in one buffer, and every buffer of a pass over `rows` rows
template <ml::transformer_config C, typename T>
struct block
{
    static constexpr std::size_t d = C.d_model, q = C.q_size(), kv = C.kv_size();
    static constexpr bool layer = C.norm == ml::norm_kind::layer;
    std::size_t rows;
    // Parameter offsets: norm1 weight and bias, wq wk wv wo, their biases, norm2 weight and bias, the MoE
    std::vector<std::size_t> at;
    std::vector<T> p, g, xs, x0, h1, x1, h2, aq, ak, av, ay, lse, mv, mw, scratch, dq, dk, dv, dy;
    std::vector<u32> mi;

    explicit block(std::size_t n, std::uint64_t seed = 1) : rows(n)
    {
        const std::vector<std::size_t> sizes{d, layer ? d : 0, q * d, kv * d, kv * d, d * q, C.bias ? q : 0, C.bias ? kv : 0, C.bias ? kv : 0,
                                             C.bias ? d : 0, d, layer ? d : 0, ml::moe_parameter_count(C)};
        std::size_t total = 0;
        for (const auto s : sizes) {
            at.push_back(total);
            total += s;
        }
        at.push_back(total);
        p = noise<T>(total, seed, 0.4);
        for (const int norm : {0, 10})
            for (std::size_t i = at[std::size_t(norm)]; i < at[std::size_t(norm) + 1]; ++i)
                p[i] += T(1);   // norm weights near 1
        g.assign(total, T(0));
        xs = noise<T>(n * d, seed + 1);
        for (auto* v : {&x0, &h1, &x1, &h2, &scratch})
            v->assign(n * d, T(0));
        for (auto* v : {&aq, &ay, &dq, &dy})
            v->assign(n * q, T(0));
        for (auto* v : {&ak, &av, &dk, &dv})
            v->assign(n * kv, T(0));
        lse.assign(n * C.n_heads, T(0));
        mv.assign(ml::moe_activation_values(C, n), T(0));
        mw.assign(ml::moe_gradient_work(C, n), T(0));
        mi.assign(ml::moe_activation_indices(C, n), 0);
    }
    std::span<const T> param(std::size_t i) const { return std::span<const T>(p).subspan(at[i], at[i + 1] - at[i]); }
    std::span<T> grad(std::size_t i) { return std::span<T>(g).subspan(at[i], at[i + 1] - at[i]); }
    ml::attention_weights<ml::dense_weights<T>, T> attention() const
    {
        return {{param(2), q, d}, {param(3), kv, d}, {param(4), kv, d}, {param(5), d, q}, param(6), param(7), param(8), param(9)};
    }
    ml::moe_block_weights<T> weights() const
    {
        return {param(0), param(1), attention(), param(10), param(11), ml::moe_weights_in<C>(param(12))};
    }
    ml::attention_activations<T> attention_act() { return {aq, ak, av, {}, {}, ay, lse}; }
    ml::moe_block_activations<T> act()
    {
        return {x0, h1, x1, h2, attention_act(), ml::moe_activations_in<C>(std::span<T>(mv), std::span<u32>(mi), rows)};
    }
    ml::moe_block_gradients<T> grads()
    {
        return {grad(0), grad(1), grad(10), grad(11),
                {dq, dk, dv, grad(2), grad(3), grad(4), grad(5), grad(6), grad(7), grad(8), grad(9), {}, {}, dy},
                ml::moe_gradients_in<C>(grad(12), std::span<T>(mw), rows), scratch};
    }
};

TEST(MlMoeBlock, OneExpertOfWeightOneIsTheDenseBlockToTheBit)
{
    constexpr auto C = one_expert;
    constexpr std::size_t n = 6, seqs = 2, d = C.d_model, f = C.d_ff;
    block<C, float> b(n);
    auto dx = noise<float>(n * d, 9);
    auto dense_xs = b.xs, dense_dx = dx;
    const auto losses = ml::block_forward_sequence<C>(ref, b.weights(), b.xs, rope_for<C, float>(), b.act(), seqs);
    EXPECT_EQ(losses.balance, 0.0f);
    ml::block_backward<C>(ref, b.weights(), rope_for<C, float>(), b.act(), dx, b.grads(), seqs);
    // The dense block with the expert's MLP, through the same backend
    const auto m = ml::moe_weights_in<C>(b.param(12));
    const ml::mlp_weights<ml::dense_weights<float>, float> mlp{m.experts.up.expert(0), {}, m.experts.down.expert(0), m.experts.up_bias,
                                                               {}, m.experts.down_bias};
    const ml::block_weights<ml::dense_weights<float>, float> w{b.param(0), b.param(1), b.attention(), b.param(10), b.param(11), mlp};
    block<C, float> c(n);   // fresh buffers
    std::vector<float> up(n * f), hidden(n * f), mlp_scratch(n * f), gup(f * d), gdown(d * f), gbup(f), gbdown(d);
    const ml::block_activations<float> act{c.x0, c.h1, c.x1, c.h2, c.attention_act(), up, {}, hidden};
    const ml::block_gradients<float> grads{c.grad(0), c.grad(1), c.grad(10), c.grad(11),
                                           {c.dq, c.dk, c.dv, c.grad(2), c.grad(3), c.grad(4), c.grad(5), c.grad(6), c.grad(7), c.grad(8), c.grad(9), {}, {}, c.dy},
                                           {gup, {}, gdown, gbup, {}, gbdown, mlp_scratch},
                                           c.scratch};
    ml::block_forward_sequence<dense_of_one>(ref, w, std::span<float>(dense_xs), rope_for<C, float>(), act, seqs);
    ml::block_backward<dense_of_one>(ref, w, rope_for<C, float>(), act, std::span<float>(dense_dx), grads, seqs);
    EXPECT_EQ(b.xs, dense_xs);
    EXPECT_EQ(dx, dense_dx);
    for (std::size_t i = 0; i < 12; ++i)
        EXPECT_EQ(std::vector<float>(b.grad(i).begin(), b.grad(i).end()), std::vector<float>(c.grad(i).begin(), c.grad(i).end())) << "tensor " << i;
    const auto gm = ml::moe_gradients_in<C>(b.grad(12), std::span<float>(b.mw), n);
    EXPECT_TRUE(std::equal(gup.begin(), gup.end(), gm.experts.up.begin()));
    EXPECT_TRUE(std::equal(gdown.begin(), gdown.end(), gm.experts.down.begin()));
    EXPECT_TRUE(std::equal(gbup.begin(), gbup.end(), gm.experts.up_bias.begin()));
    EXPECT_TRUE(std::all_of(gm.router.begin(), gm.router.end(), [](float v) { return v == 0.0f; }));   // one expert: no routing gradient
}

template <ml::transformer_config C>
void block_backward_is_central_differences(std::size_t n, std::size_t seqs)
{
    block<C, double> b(n, 21);
    const auto dout = noise<double>(n * C.d_model, 22);
    auto run = [&](block<C, double>& m) {
        auto xs = m.xs;
        const auto aux = ml::block_forward_sequence<C>(ref, m.weights(), xs, rope_for<C, double>(), m.act(), seqs);
        double s = aux.balance + aux.z;
        for (std::size_t i = 0; i < xs.size(); ++i)
            s += dout[i] * xs[i];
        return s;
    };
    {
        auto xs = b.xs;
        ml::block_forward_sequence<C>(ref, b.weights(), xs, rope_for<C, double>(), b.act(), seqs);
    }
    auto dx = dout;
    ml::block_backward<C>(ref, b.weights(), rope_for<C, double>(), b.act(), dx, b.grads(), seqs);
    constexpr double h = 1e-6;
    block<C, double> m(n, 21);
    auto check = [&](std::vector<double>& v, std::size_t j) {
        const double v0 = v[j];
        v[j] = v0 + h;
        const double up = run(m);
        v[j] = v0 - h;
        const double down = run(m);
        v[j] = v0;
        return (up - down) / (2 * h);
    };
    for (std::size_t j = 0; j < m.xs.size(); ++j)
        EXPECT_NEAR(dx[j], check(m.xs, j), 1e-6 * std::max(1.0, std::abs(dx[j]))) << "input " << j;
    for (std::size_t j = 0; j < m.p.size(); ++j)
        EXPECT_NEAR(b.g[j], check(m.p, j), 1e-6 * std::max(1.0, std::abs(b.g[j]))) << "parameter " << j;
}

TEST(MlMoeBlock, BackwardMatchesCentralDifferences)
{
    block_backward_is_central_differences<moe_cfg>(6, 2);   // two sequences of 3: attention and the balance loss within each
    block_backward_is_central_differences<one_expert>(4, 1);
}

TEST(MlMoeBlock, TheForwardReturnsTheLossesOfItsSequences)
{
    constexpr auto C = moe_cfg;
    block<C, double> b(8);
    auto xs = b.xs;
    const auto losses = ml::block_forward_sequence<C>(ref, b.weights(), xs, rope_for<C, double>(), b.act(), 2);
    const auto a = b.act().moe;
    EXPECT_EQ(losses.balance, ml::moe_balance_loss<C>(ref, std::span<const double>(a.logits), a.experts, std::span<double>{}, 2));
    EXPECT_EQ(losses.z, ml::moe_z_loss<C>(ref, std::span<const double>(a.logits), std::span<double>{}));
    EXPECT_GT(losses.balance, 0.0);
    EXPECT_GT(losses.z, 0.0);
}

} // namespace
