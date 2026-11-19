//
//  moe_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/moe.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
std::vector<T> noise(std::size_t n, std::uint64_t seed, double scale = 1.0)
{
    auto g = stats::wyrand::stream(seed, 0);
    std::vector<T> v(n);
    for (auto& e : v)
        e = static_cast<T>(scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

// Configurations: the bit-for-bit ones have no biases
constexpr ml::transformer_config mixtral{.d_model = 6, .n_heads = 1, .n_kv_heads = 1, .head_dim = 6, .d_ff = 8, .mlp = ml::mlp_kind::swiglu,
                                         .experts = 4, .top_k = 2, .shared_experts = 1, .expert_d_ff = 8};
constexpr ml::transformer_config deepseek{.d_model = 6, .n_heads = 1, .n_kv_heads = 1, .head_dim = 6, .d_ff = 5, .mlp = ml::mlp_kind::gelu,
                                          .bias = true, .experts = 8, .top_k = 3, .shared_experts = 2, .expert_d_ff = 5,
                                          .scoring = ml::moe_scoring::sigmoid, .route_scale = 2.5, .bias_update_rate = 0.001,
                                          .balance_loss_weight = 0.01, .z_loss_weight = 0.001};
constexpr ml::transformer_config switch_like{.d_model = 5, .n_heads = 1, .n_kv_heads = 1, .head_dim = 5, .d_ff = 6,
                                             .mlp = ml::mlp_kind::relu_squared, .experts = 6, .top_k = 2, .expert_d_ff = 6,
                                             .scoring = ml::moe_scoring::softmax, .balance_loss_weight = 0.05, .z_loss_weight = 0.01};
constexpr ml::transformer_config top1{.d_model = 4, .n_heads = 1, .n_kv_heads = 1, .head_dim = 4, .d_ff = 6, .mlp = ml::mlp_kind::gelu_tanh,
                                      .experts = 5, .top_k = 1, .expert_d_ff = 6};
constexpr ml::transformer_config all_experts{.d_model = 4, .n_heads = 1, .n_kv_heads = 1, .head_dim = 4, .d_ff = 3, .mlp = ml::mlp_kind::swiglu,
                                             .experts = 3, .top_k = 3, .expert_d_ff = 3};

template <ml::transformer_config C>
constexpr ml::transformer_config with_scoring(ml::moe_scoring s, double scale = 1.0)
{
    auto c = C;
    c.scoring = s;
    c.route_scale = scale;
    return c;
}

// One layer's buffers, laid out by the header's own functions
template <ml::transformer_config C, typename T>
struct layer
{
    std::size_t rows;
    std::vector<T> params, bias, xs, ys, values, grads, work, dxs;
    std::vector<u32> indices;

    explicit layer(std::size_t n, std::uint64_t seed = 1, bool with_bias = false)
        : rows(n), params(noise<T>(ml::moe_parameter_count(C), seed, 0.5)), bias(with_bias ? noise<T>(C.experts, seed + 1, 0.3) : std::vector<T>{}),
          xs(noise<T>(n * C.d_model, seed + 2)), ys(n * C.d_model), values(ml::moe_activation_values(C, n)), grads(params.size()),
          work(ml::moe_gradient_work(C, n)), dxs(n * C.d_model), indices(ml::moe_activation_indices(C, n))
    {
    }
    ml::moe_weights<T> weights() const { return ml::moe_weights_in<C>(std::span<const T>(params), std::span<const T>(bias)); }
    ml::moe_activations<T> act() { return ml::moe_activations_in<C>(std::span<T>(values), std::span<u32>(indices), rows); }
    ml::moe_gradients<T> grad() { return ml::moe_gradients_in<C>(std::span<T>(grads), std::span<T>(work), rows); }
    ml::moe_losses<T> forward(std::size_t sequences = 1) { return ml::moe_forward<C>(ref, weights(), xs, ys, act(), sequences); }
    void backward(const std::vector<T>& dys, std::size_t sequences = 1) { ml::moe_backward<C>(ref, weights(), xs, act(), dys, dxs, grad(), sequences); }
};

// ---- Routing

template <ml::transformer_config C>
void softmax_top_k_is_moe_gate()
{
    constexpr std::size_t E = C.experts, K = C.top_k, rows = 9;
    const auto logits = noise<float>(rows * E, 7, 3.0);
    std::vector<u32> ids(rows * K);
    std::vector<float> w(rows * K), dw = noise<float>(rows * K, 8), dz(rows * E);
    ml::moe_route_rows<C>(ref, logits, {}, ids, std::span<float>(w));
    ml::moe_route_rows_backward<C>(ref, logits, ids, dw, std::span<float>(dz));
    for (std::size_t t = 0; t < rows; ++t) {
        const auto z = std::span<const float>(logits).subspan(t * E, E);
        const auto gate = ml::moe_gate<K>(z);
        std::array<float, E> g{};
        ml::moe_gate_backward(gate, std::span<const float>(dw).subspan(t * K, K), g);
        for (std::size_t i = 0; i < K; ++i) {
            EXPECT_EQ(ids[t * K + i], gate.expert[i]);
            EXPECT_EQ(w[t * K + i], gate.weight[i]);
        }
        for (std::size_t e = 0; e < E; ++e)
            EXPECT_EQ(dz[t * E + e], g[e]);
    }
}

TEST(MlMoeRouting, SoftmaxTopKIsMoeGateToTheBit)
{
    softmax_top_k_is_moe_gate<mixtral>();
    softmax_top_k_is_moe_gate<top1>();
    softmax_top_k_is_moe_gate<all_experts>();
}

// Each scoring's weights against its definition in double, the choice by the biased keys
template <ml::transformer_config C>
void weights_are_the_definition()
{
    constexpr std::size_t E = C.experts, K = C.top_k, rows = 7;
    const auto logits = noise<double>(rows * E, 11, 2.0);
    const auto bias = noise<double>(E, 12, 0.5);
    std::vector<u32> ids(rows * K);
    std::vector<double> w(rows * K);
    ml::moe_route_rows<C>(ref, logits, bias, ids, std::span<double>(w));
    for (std::size_t t = 0; t < rows; ++t) {
        std::vector<double> s(E), key(E);
        double total = 0;
        for (std::size_t e = 0; e < E; ++e)
            total += std::exp(logits[t * E + e]);
        for (std::size_t e = 0; e < E; ++e) {
            const double z = logits[t * E + e];
            s[e] = C.scoring == ml::moe_scoring::softmax_top_k ? z : C.scoring == ml::moe_scoring::softmax ? std::exp(z) / total : 1 / (1 + std::exp(-z));
            key[e] = s[e] + bias[e];
        }
        std::vector<std::size_t> order(E);
        for (std::size_t e = 0; e < E; ++e)
            order[e] = e;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return key[a] > key[b]; });
        double sum = 0;
        for (std::size_t i = 0; i < K; ++i)
            sum += C.scoring == ml::moe_scoring::softmax_top_k ? std::exp(s[order[i]]) : s[order[i]];
        for (std::size_t i = 0; i < K; ++i) {
            ASSERT_EQ(ids[t * K + i], order[i]);
            const double raw = C.scoring == ml::moe_scoring::softmax_top_k ? std::exp(s[order[i]]) / sum
                               : C.scoring == ml::moe_scoring::softmax   ? s[order[i]]
                                                                         : s[order[i]] / sum;
            EXPECT_NEAR(w[t * K + i], raw * C.route_scale, 1e-14);
        }
    }
}

TEST(MlMoeRouting, EachScoringIsItsDefinitionAndTheBiasOnlyChooses)
{
    weights_are_the_definition<with_scoring<deepseek>(ml::moe_scoring::softmax_top_k, 1.5)>();
    weights_are_the_definition<with_scoring<deepseek>(ml::moe_scoring::softmax)>();
    weights_are_the_definition<deepseek>();
    weights_are_the_definition<with_scoring<switch_like>(ml::moe_scoring::sigmoid)>();
}

TEST(MlMoeRouting, ABiasChangesTheChoiceButNotTheWeightsFormula)
{
    constexpr auto C = deepseek;   // sigmoid
    const std::vector<double> z{2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0};
    std::vector<double> bias(8, 0.0);
    bias[7] = 10.0;   // expert 7 is chosen first, but weighted by its own sigmoid
    std::vector<u32> ids(3);
    std::vector<double> w(3);
    ml::moe_route_rows<C>(ref, z, bias, ids, std::span<double>(w));
    EXPECT_EQ(ids, (std::vector<u32>{7, 0, 1}));
    const double s7 = 1 / (1 + std::exp(3.0)), s0 = 1 / (1 + std::exp(-2.0)), s1 = 1 / (1 + std::exp(-1.0));
    EXPECT_NEAR(w[0], 2.5 * s7 / (s7 + s0 + s1), 1e-15);
    EXPECT_NEAR(w[1], 2.5 * s0 / (s7 + s0 + s1), 1e-15);
}

TEST(MlMoeRouting, TiesGoToTheLowerExpert)
{
    const std::vector<float> z(4, 0.25f);
    std::vector<u32> ids(2);
    std::vector<float> w(2);
    ml::moe_route_rows<mixtral>(ref, z, {}, ids, std::span<float>(w));
    EXPECT_EQ(ids, (std::vector<u32>{0, 1}));
    EXPECT_EQ(w[0], 0.5f);
    EXPECT_EQ(w[1], 0.5f);
    // With K = E every expert is chosen, in order of the logits, ties in index order
    const std::vector<float> z3{1.0f, 2.0f, 1.0f};
    std::vector<u32> all(3);
    std::vector<float> w3(3);
    ml::moe_route_rows<all_experts>(ref, z3, {}, all, std::span<float>(w3));
    EXPECT_EQ(all, (std::vector<u32>{1, 0, 2}));
    EXPECT_NEAR(w3[0] + w3[1] + w3[2], 1.0f, 1e-6f);
}

template <ml::transformer_config C>
void route_backward_is_central_differences()
{
    constexpr std::size_t E = C.experts, K = C.top_k, rows = 5;
    auto z = noise<double>(rows * E, 21, 2.0);
    const auto bias = noise<double>(E, 22, 0.2);
    const auto dw = noise<double>(rows * K, 23);
    std::vector<u32> ids(rows * K);
    std::vector<double> w(rows * K), dz(rows * E);
    ml::moe_route_rows<C>(ref, z, bias, ids, std::span<double>(w));
    ml::moe_route_rows_backward<C>(ref, z, ids, dw, std::span<double>(dz));
    auto loss = [&] {
        std::vector<u32> i2(rows * K);
        std::vector<double> w2(rows * K);
        ml::moe_route_rows<C>(ref, z, bias, i2, std::span<double>(w2));
        EXPECT_EQ(i2, ids);   // the step is too small to change a choice
        double s = 0;
        for (std::size_t j = 0; j < w2.size(); ++j)
            s += dw[j] * w2[j];
        return s;
    };
    constexpr double h = 1e-6;
    for (std::size_t j = 0; j < z.size(); ++j) {
        const double z0 = z[j];
        z[j] = z0 + h;
        const double up = loss();
        z[j] = z0 - h;
        const double down = loss();
        z[j] = z0;
        EXPECT_NEAR(dz[j], (up - down) / (2 * h), 1e-8) << j;
    }
}

TEST(MlMoeRouting, BackwardOfEachScoringMatchesCentralDifferences)
{
    route_backward_is_central_differences<with_scoring<deepseek>(ml::moe_scoring::softmax_top_k, 1.5)>();
    route_backward_is_central_differences<with_scoring<deepseek>(ml::moe_scoring::softmax, 0.7)>();
    route_backward_is_central_differences<deepseek>();
    route_backward_is_central_differences<top1>();
    route_backward_is_central_differences<all_experts>();
}

// ---- Counting and permutation

TEST(MlMoePermutation, OrdersThePairsByExpertThenRowThenSlot)
{
    constexpr auto C = mixtral;   // E = 4, K = 2
    const std::vector<u32> experts{2, 0, 0, 3, 2, 3, 0, 2};   // rows 0..3, two slots each; expert 1 gets nothing
    std::vector<u32> counts(4), offsets(5), order(8), inverse(8);
    ml::moe_count_rows<C>(ref, experts, counts);
    ml::moe_offsets<C>(ref, counts, offsets);
    ml::moe_permutation<C>(ref, experts, offsets, order, inverse);
    EXPECT_EQ(counts, (std::vector<u32>{3, 0, 3, 2}));
    EXPECT_EQ(offsets, (std::vector<u32>{0, 3, 3, 6, 8}));
    EXPECT_EQ(order, (std::vector<u32>{1, 2, 6, 0, 4, 7, 3, 5}));
    for (std::size_t j = 0; j < 8; ++j)
        EXPECT_EQ(order[inverse[j]], j);
}

TEST(MlMoePermutation, AllRowsToOneExpertAndNoRowsAtAll)
{
    constexpr auto C = top1;   // E = 5, K = 1
    const std::vector<u32> experts(6, 3);
    std::vector<u32> counts(5), offsets(6), order(6), inverse(6);
    ml::moe_count_rows<C>(ref, experts, counts);
    ml::moe_offsets<C>(ref, counts, offsets);
    ml::moe_permutation<C>(ref, experts, offsets, order, inverse);
    EXPECT_EQ(counts, (std::vector<u32>{0, 0, 0, 6, 0}));
    EXPECT_EQ(offsets, (std::vector<u32>{0, 0, 0, 0, 6, 6}));
    EXPECT_EQ(order, (std::vector<u32>{0, 1, 2, 3, 4, 5}));
    // Zero rows: zero counts, zero offsets, nothing permuted
    std::vector<u32> none, c0(5, 9), o0(6, 9);
    ml::moe_count_rows<C>(ref, none, c0);
    ml::moe_offsets<C>(ref, c0, o0);
    ml::moe_permutation<C>(ref, none, o0, std::span<u32>{}, std::span<u32>{});
    EXPECT_EQ(c0, (std::vector<u32>(5, 0)));
    EXPECT_EQ(o0, (std::vector<u32>(6, 0)));
}

TEST(MlMoePermutation, IsConstexpr)
{
    static_assert([] {
        constexpr auto C = mixtral;
        std::array<u32, 4> experts{3, 1, 1, 0}, counts{}, order{}, inverse{};
        std::array<u32, 5> offsets{};
        ml::moe_count_rows<C>(ref, experts, counts);
        ml::moe_offsets<C>(ref, counts, offsets);
        ml::moe_permutation<C>(ref, experts, offsets, order, inverse);
        return order == std::array<u32, 4>{3, 1, 2, 0} && offsets[4] == 4 && inverse[0] == 3;
    }());
}

TEST(MlMoeRouting, RoutesAndCombinesAtCompileTime)
{
    static_assert([] {
        std::array<double, 4> z{0.0, 1.0, 1.0, -1.0};
        std::array<double, 2> w{};
        std::array<u32, 2> ids{};
        ml::moe_route_rows<mixtral>(ref, z, {}, ids, std::span<double>(w));
        return ids[0] == 1 && ids[1] == 2 && w[0] == 0.5 && w[1] == 0.5;
    }());
    static_assert([] {
        std::array<double, 12> yp{1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
        std::array<double, 6> ys{};
        std::array<double, 2> w{0.25, 0.5};
        std::array<u32, 2> inverse{1, 0};
        ml::moe_combine_rows<mixtral>(ref, yp, w, inverse, std::span<double>(ys));
        return ys[0] == 0.25 * 2 + 0.5 * 1 && ys[5] == 1.0;
    }());
}

// ---- Gather and combine

TEST(MlMoeCombine, IsTheSlotOrderSumAndItsBackwardIsExact)
{
    constexpr auto C = mixtral;
    constexpr std::size_t d = C.d_model, K = C.top_k, rows = 4;
    const std::vector<u32> experts{2, 0, 0, 3, 2, 3, 0, 2};
    std::vector<u32> counts(4), offsets(5), order(8), inverse(8);
    ml::moe_count_rows<C>(ref, experts, counts);
    ml::moe_offsets<C>(ref, counts, offsets);
    ml::moe_permutation<C>(ref, experts, offsets, order, inverse);
    const auto xs = noise<float>(rows * d, 31), yp = noise<float>(rows * K * d, 32), w = noise<float>(rows * K, 33), dy = noise<float>(rows * d, 34);
    std::vector<float> xp(rows * K * d), ys(rows * d), ys2(rows * d), dyp(yp.size()), dw(rows * K), dxs(rows * d, 1.0f);
    ml::moe_gather_rows<C>(ref, xs, order, std::span<float>(xp));
    ml::moe_combine_rows<C>(ref, yp, w, inverse, std::span<float>(ys));
    ml::moe_combine_rows<C>(ref, yp, w, inverse, std::span<float>(ys2));
    EXPECT_EQ(ys, ys2);   // deterministic
    ml::moe_combine_rows_backward<C>(ref, dy, yp, w, inverse, dyp, std::span<float>(dw));
    ml::moe_gather_rows_backward<C>(ref, std::span<const float>(xp), inverse, std::span<float>(dxs));
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < d; ++i) {
            const std::size_t p0 = inverse[r * K], p1 = inverse[r * K + 1];
            EXPECT_EQ(xp[p0 * d + i], xs[r * d + i]);
            EXPECT_EQ(xp[p1 * d + i], xs[r * d + i]);
            const float a = w[r * K] * yp[p0 * d + i], b = w[r * K + 1] * yp[p1 * d + i];
            EXPECT_EQ(ys[r * d + i], a + b);
            EXPECT_EQ(dyp[p0 * d + i], w[r * K] * dy[r * d + i]);
            const float g = 1.0f + xp[p0 * d + i];
            EXPECT_EQ(dxs[r * d + i], g + xp[p1 * d + i]);   // added into, slot by slot
        }
    for (std::size_t j = 0; j < rows * K; ++j) {
        float dot = 0;
        for (std::size_t i = 0; i < d; ++i) {
            const float v = yp[inverse[j] * d + i] * dy[(j / K) * d + i];
            dot = dot + v;
        }
        EXPECT_EQ(dw[j], dot);
    }
}

TEST(MlMoeCombine, WithOneSlotIsTheWeightedRow)
{
    constexpr auto C = top1;
    const std::vector<u32> inverse{1, 0};
    const std::vector<float> yp{1, 2, 3, 4, 5, 6, 7, 8}, w{0.5f, 2.0f};
    std::vector<float> ys(8);
    ml::moe_combine_rows<C>(ref, yp, w, inverse, std::span<float>(ys));
    EXPECT_EQ(ys, (std::vector<float>{2.5f, 3, 3.5f, 4, 2, 4, 6, 8}));
}

// ---- The grouped products

TEST(MlMoeGrouped, IsEachExpertsLinearSequenceWithEmptySegmentsSkipped)
{
    constexpr std::size_t E = 3, rows = 4, cols = 5;
    const auto data = noise<double>(E * rows * cols, 41), bias = noise<double>(E * rows, 42);
    const std::vector<u32> offsets{0, 2, 2, 7};
    const auto xs = noise<double>(7 * cols, 43), dys = noise<double>(7 * rows, 44);
    const ml::grouped_weights<double> w{data, E, rows, cols};
    std::vector<double> ys(7 * rows), dxs(7 * cols, 0.5), dw(data.size(), 0.25), db(E * rows);
    ml::grouped_linear(ref, w, offsets, xs, ys, bias);
    ml::grouped_linear_backward(ref, w, offsets, xs, dys, dxs, dw, db);
    std::vector<double> want_y(ys.size()), want_dx(dxs.size(), 0.5), want_dw(dw.size(), 0.25), want_db(db.size());
    for (std::size_t e = 0; e < E; ++e) {
        const std::size_t f = offsets[e], n = offsets[e + 1] - f;
        if (n == 0)
            continue;
        const auto we = w.expert(e);
        ml::linear_sequence(ref, we, std::span<const double>(xs).subspan(f * cols, n * cols), std::span<double>(want_y).subspan(f * rows, n * rows),
                            std::span<const double>(bias).subspan(e * rows, rows));
        ml::linear_sequence_backward(ref, we, std::span<const double>(xs).subspan(f * cols, n * cols),
                                     std::span<const double>(dys).subspan(f * rows, n * rows), std::span<double>(want_dx).subspan(f * cols, n * cols),
                                     std::span<double>(want_dw).subspan(e * rows * cols, rows * cols), std::span<double>(want_db).subspan(e * rows, rows));
    }
    EXPECT_EQ(ys, want_y);
    EXPECT_EQ(dxs, want_dx);
    EXPECT_EQ(dw, want_dw);
    EXPECT_EQ(db, want_db);
    for (std::size_t i = rows * cols; i < 2 * rows * cols; ++i)
        EXPECT_EQ(dw[i], 0.25);   // expert 1 had no rows
    // The slice of a run of experts
    const auto s = w.slice(1, 2);
    EXPECT_EQ(s.experts, 2u);
    EXPECT_EQ(s.expert(0).data.data(), w.expert(1).data.data());
}

// ---- The layer

// The layer token by token, from moe_gate and mlp_forward: the bits of the batched layer for a configuration whose
// order is the same (Mixtral scoring at scale 1, no biases, no selection bias)
template <ml::transformer_config C>
void layer_is_the_per_token_path()
{
    constexpr std::size_t d = C.d_model, E = C.experts, K = C.top_k, rows = 11;
    constexpr auto Ce = ml::moe_expert_config(C);
    layer<C, float> l(rows, 51);
    const auto losses = l.forward();
    EXPECT_EQ(losses.balance, 0.0f);
    EXPECT_EQ(losses.z, 0.0f);
    const auto w = l.weights();
    for (std::size_t r = 0; r < rows; ++r) {
        const auto x = std::span<const float>(l.xs).subspan(r * d, d);
        std::array<float, E> z{};
        ml::linear(w.router, x, z);
        const auto route = ml::moe_gate<K>(z);
        std::array<float, d> y{}, ye{};
        for (std::size_t i = 0; i < K; ++i) {
            const std::size_t e = route.expert[i];
            const ml::mlp_weights<ml::dense_weights<float>, float> we{w.experts.up.expert(e), Ce.mlp == ml::mlp_kind::swiglu ? w.experts.gate.expert(e) : ml::dense_weights<float>{},
                                                                      w.experts.down.expert(e)};
            ml::mlp_forward<Ce>(we, x, ye);
            for (std::size_t j = 0; j < d; ++j) {
                const float v = route.weight[i] * ye[j];
                y[j] = i == 0 ? v : y[j] + v;
            }
        }
        if constexpr (C.shared_experts > 0) {
            std::array<float, d> s{};
            ml::mlp_forward<ml::moe_shared_config(C)>(w.shared, x, s);
            for (std::size_t j = 0; j < d; ++j)
                y[j] = s[j] + y[j];
        }
        for (std::size_t j = 0; j < d; ++j)
            ASSERT_EQ(l.ys[r * d + j], y[j]) << "row " << r << " element " << j;
    }
}

TEST(MlMoeLayer, ForwardIsThePerTokenPathToTheBit)
{
    layer_is_the_per_token_path<mixtral>();
    layer_is_the_per_token_path<top1>();
    layer_is_the_per_token_path<all_experts>();
    layer_is_the_per_token_path<ml::tiny_moe>();
}

// The layer's gradient, every input and every parameter, against central differences in double; the loss is
// sum(dy · y) plus the auxiliary losses the configuration weights
template <ml::transformer_config C>
void layer_backward_is_central_differences(std::size_t rows, std::size_t sequences, bool selection_bias, std::vector<double> forced = {})
{
    layer<C, double> l(rows, 61, selection_bias);
    if (!forced.empty())
        l.bias = forced;
    const auto dy = noise<double>(rows * C.d_model, 62);
    l.forward(sequences);
    const auto chosen = std::vector<u32>(l.act().experts.begin(), l.act().experts.end());
    l.backward(dy, sequences);
    auto loss = [&] {
        layer<C, double> m(rows, 61, selection_bias);
        m.bias = l.bias;
        m.params = l.params;
        m.xs = l.xs;
        const auto aux = m.forward(sequences);
        EXPECT_TRUE(std::equal(chosen.begin(), chosen.end(), m.act().experts.begin()));
        double s = aux.balance + aux.z;
        for (std::size_t i = 0; i < m.ys.size(); ++i)
            s += dy[i] * m.ys[i];
        return s;
    };
    constexpr double h = 1e-6;
    auto check = [&](std::vector<double>& v, const std::vector<double>& g, const char* what) {
        for (std::size_t j = 0; j < v.size(); ++j) {
            const double v0 = v[j];
            v[j] = v0 + h;
            const double up = loss();
            v[j] = v0 - h;
            const double down = loss();
            v[j] = v0;
            EXPECT_NEAR(g[j], (up - down) / (2 * h), 2e-7 * std::max(1.0, std::abs(g[j]))) << what << " " << j;
        }
    };
    const auto gx = l.dxs, gp = l.grads;
    check(l.xs, gx, "input");
    check(l.params, gp, "parameter");
}

TEST(MlMoeLayer, BackwardMatchesCentralDifferences)
{
    layer_backward_is_central_differences<mixtral>(6, 1, false);
    layer_backward_is_central_differences<deepseek>(6, 2, true);       // sigmoid, scale, bias, biases, both losses by sequence
    layer_backward_is_central_differences<switch_like>(6, 3, false);   // softmax over every expert, both losses
    layer_backward_is_central_differences<top1>(4, 1, true);
    layer_backward_is_central_differences<all_experts>(3, 1, false);
}

TEST(MlMoeLayer, AllRowsToOneExpertLeaveTheOthersEmpty)
{
    // A selection bias of 100 on expert 2 sends every row there: the other experts' segments are empty
    const std::vector<double> forced{0, 0, 100, 0, 0};
    layer<top1, double> l(5, 1);
    l.bias = forced;
    l.forward();
    const auto a = l.act();
    EXPECT_TRUE(std::equal(a.counts.begin(), a.counts.end(), std::vector<u32>{0, 0, 5, 0, 0}.begin()));
    l.backward(noise<double>(5 * top1.d_model, 2));
    const auto g = l.grad();
    const std::size_t each = top1.expert_d_ff * top1.d_model;
    for (std::size_t i = 0; i < g.experts.up.size(); ++i)
        EXPECT_EQ(g.experts.up[i] != 0.0, i / each == 2) << i;
    layer_backward_is_central_differences<top1>(5, 1, false, forced);
}

TEST(MlMoeLayer, ZeroRowsIsNothingToDo)
{
    layer<deepseek, float> l(0);
    const auto losses = l.forward();
    l.backward({});
    EXPECT_EQ(losses.balance, 0.0f);
    EXPECT_EQ(losses.z, 0.0f);
    EXPECT_TRUE(std::all_of(l.grads.begin(), l.grads.end(), [](float g) { return g == 0.0f; }));
    EXPECT_EQ(l.act().offsets[deepseek.experts], 0u);
}

TEST(MlMoeLayer, TheInputGradientAddsIntoAndTheWeightGradientsDoNotDependOnIt)
{
    layer<deepseek, double> a(5, 71, true), b(5, 71, true);
    b.dxs.assign(b.dxs.size(), 1.0);
    const auto dy = noise<double>(5 * deepseek.d_model, 72);
    a.forward();
    a.backward(dy);
    b.forward();
    b.backward(dy);
    for (std::size_t i = 0; i < a.dxs.size(); ++i)
        EXPECT_NEAR(b.dxs[i], a.dxs[i] + 1.0, 1e-14);
    EXPECT_EQ(a.grads, b.grads);
}

TEST(MlMoeLayer, BuffersAreSizedByConstexprFunctions)
{
    constexpr auto C = ml::tiny_moe;   // d 8, E 4, K 2, f 16, one shared, SwiGLU, no biases
    static_assert(ml::moe_parameter_count(C) == 4 * 8 + 4 * 3 * 16 * 8 + 3 * 16 * 8);
    static_assert(ml::moe_activation_values(C, 3) == 3 * 4 + 6 + 2 * 6 * 8 + 3 * 6 * 16 + 3 * 3 * 16 + 3 * 8);
    static_assert(ml::moe_activation_indices(C, 3) == 3 * 6 + 2 * 4 + 1);
    static_assert(ml::moe_gradient_work(C, 3) == 2 * 6 * 16 + 2 * 3 * 16 + 3 * 4 + 6 + 2 * 6 * 8);
    static_assert(ml::moe_parameter_count(ml::olmoe_1b_7b) == 64 * 2048 + 64 * 3 * 1024 * 2048);
    static_assert(ml::moe_expert_config(C).d_ff == 16 && ml::moe_expert_config(C).experts == 0 && valid(ml::moe_expert_config(C)));
    static_assert(ml::moe_shared_config(deepseek).d_ff == 10 && valid(ml::moe_shared_config(deepseek)));
    std::vector<float> p(ml::moe_parameter_count(C));
    const auto w = ml::moe_weights_in<C>(std::span<const float>(p));
    EXPECT_EQ(w.router.rows, 4u);
    EXPECT_EQ(w.experts.gate.data.size(), 4u * 16 * 8);
    EXPECT_EQ(w.shared.down.data.data() + w.shared.down.data.size(), p.data() + p.size());
    static_assert(std::is_trivially_copyable_v<ml::moe_weights<float>> && std::is_trivially_copyable_v<ml::moe_activations<float>>
                  && std::is_trivially_copyable_v<ml::moe_gradients<float>> && std::is_trivially_copyable_v<ml::grouped_weights<float>>);
}

TEST(MlMoeLayer, TheExpertRangeOfWeightsAndGradientsIsTheirSlice)
{
    layer<deepseek, float> l(2);
    const auto w = l.weights();
    const auto s = ml::moe_expert_range(w.experts, 2, 3);
    EXPECT_EQ(s.up.experts, 3u);
    EXPECT_EQ(s.down.expert(0).data.data(), w.experts.down.expert(2).data.data());
    EXPECT_EQ(s.up_bias.data(), w.experts.up_bias.data() + 2 * deepseek.expert_d_ff);
    const auto g = l.grad();
    const auto gs = ml::moe_expert_range<deepseek>(g.experts, 2, 3);
    EXPECT_EQ(gs.up.size(), 3u * deepseek.expert_d_ff * deepseek.d_model);
    EXPECT_EQ(gs.down_bias.data(), g.experts.down_bias.data() + 2 * deepseek.d_model);
    EXPECT_TRUE(gs.gate.empty());   // GELU has no gate
}

// ---- Balancing

TEST(MlMoeBalance, TheBiasRuleMovesASkewedRouterTowardUniform)
{
    constexpr auto C = deepseek;   // E 8, K 3, gamma 0.001
    constexpr std::size_t E = C.experts, K = C.top_k, rows = 256;
    // A router that favours the low experts: logits fall with the expert's number, plus noise per token
    auto logits = noise<double>(rows * E, 81, 0.5);
    for (std::size_t t = 0; t < rows; ++t)
        for (std::size_t e = 0; e < E; ++e)
            logits[t * E + e] += 0.3 * (4.0 - double(e));
    std::vector<double> bias(E, 0.0), w(rows * K);
    std::vector<u32> ids(rows * K), counts(E);
    auto load = [&] {
        ml::moe_route_rows<C>(ref, logits, bias, ids, std::span<double>(w));
        ml::moe_count_rows<C>(ref, ids, counts);
        ml::moe_load<E> l;
        l.add(counts);
        return l;
    };
    const auto before = load();
    for (int step = 0; step < 400; ++step) {
        load();
        ml::moe_update_bias<C>(ref, counts, std::span<double>(bias));
    }
    const auto after = load();
    EXPECT_GT(before.max_over_mean(), 1.8);
    EXPECT_LT(after.max_over_mean(), 1.2);
    EXPECT_LT(after.coefficient_of_variation(), before.coefficient_of_variation() / 3);
    EXPECT_GT(bias[E - 1], bias[0]);   // the unloved experts were raised
}

TEST(MlMoeBalance, TheBiasStepIsExactlyGammaBySign)
{
    static_assert(ml::bias_update_rate_v<ml::olmoe_1b_7b, float> == 0.0f);
    static constexpr auto D = [] { auto x = ml::tiny_moe; x.bias_update_rate = 0.25; return x; }();
    const std::vector<u32> counts{3, 1, 2, 2};   // mean 2
    std::vector<float> bias{1.0f, 1.0f, 1.0f, 1.0f};
    ml::moe_update_bias<D>(ref, counts, std::span<float>(bias));
    EXPECT_EQ(bias, (std::vector<float>{0.75f, 1.25f, 1.0f, 1.0f}));
    static_assert([] {
        std::array<u32, 4> n{0, 0, 0, 4};
        std::array<double, 4> b{};
        ml::moe_update_bias<D>(ref, n, std::span<double>(b));
        return b[0] == 0.25 && b[3] == -0.25;
    }());
}

TEST(MlMoeBalance, TheBalanceLossIsAlphaAtUniformRoutingAndSplitsByDenominator)
{
    constexpr auto C = switch_like;   // softmax over 6, K 2, alpha 0.05
    constexpr std::size_t E = C.experts, rows = 6;
    // Equal logits: P_e = 1/6; rows chose (0,1), (2,3), (4,5) twice: f_e = 1/6
    const std::vector<double> z(rows * E, 0.0);
    const std::vector<u32> ids{0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
    EXPECT_NEAR(ml::moe_balance_loss<C>(ref, z, ids, std::span<double>{}), 0.05, 1e-15);
    // Two workers each holding one of two sequences, passing the batch's two: their losses and gradients add to the batch's
    const auto zs = noise<double>(rows * E, 91, 2.0);
    const std::vector<u32> skew{0, 1, 0, 2, 0, 1, 3, 0, 0, 4, 5, 0};
    std::vector<double> whole(rows * E), parts(rows * E);
    const double lw = ml::moe_balance_loss<C>(ref, zs, skew, std::span<double>(whole), 2);
    const auto half = [&](std::size_t s, std::span<double> g) {
        return ml::moe_balance_loss<C>(ref, std::span<const double>(zs).subspan(s * 3 * E, 3 * E), std::span<const u32>(skew).subspan(s * 6, 6),
                                       g.subspan(s * 3 * E, 3 * E), 1, 2);
    };
    const double l0 = half(0, parts), l1 = half(1, parts);
    EXPECT_NEAR(l0 + l1, lw, 1e-15);
    EXPECT_EQ(parts, whole);   // each row's gradient depends only on its sequence and the denominator
}

template <ml::transformer_config C>
void losses_are_central_differences()
{
    constexpr std::size_t E = C.experts, K = C.top_k, rows = 6;
    auto z = noise<double>(rows * E, 101, 2.0);
    std::vector<u32> ids(rows * K);
    std::vector<double> w(rows * K);
    ml::moe_route_rows<C>(ref, z, {}, ids, std::span<double>(w));
    std::vector<double> gb(rows * E, 0.5), gz(rows * E, -0.5);
    ml::moe_balance_loss<C>(ref, z, ids, std::span<double>(gb), 2);
    ml::moe_z_loss<C>(ref, z, std::span<double>(gz));
    constexpr double h = 1e-6;
    for (std::size_t j = 0; j < z.size(); ++j) {
        const double z0 = z[j];
        z[j] = z0 + h;
        const double bu = ml::moe_balance_loss<C>(ref, z, ids, std::span<double>{}, 2), zu = ml::moe_z_loss<C>(ref, z, std::span<double>{});
        z[j] = z0 - h;
        const double bd = ml::moe_balance_loss<C>(ref, z, ids, std::span<double>{}, 2), zd = ml::moe_z_loss<C>(ref, z, std::span<double>{});
        z[j] = z0;
        EXPECT_NEAR(gb[j] - 0.5, (bu - bd) / (2 * h), 1e-9) << j;   // added into
        EXPECT_NEAR(gz[j] + 0.5, (zu - zd) / (2 * h), 1e-9) << j;
    }
}

TEST(MlMoeBalance, TheLossesGradientsMatchCentralDifferences)
{
    losses_are_central_differences<switch_like>();
    losses_are_central_differences<deepseek>();   // the normalised sigmoid
}

TEST(MlMoeBalance, TheZLossIsTheMeanSquaredLogSumExp)
{
    constexpr auto C = switch_like;   // beta 0.01
    std::vector<double> z(12, 0.0);   // two rows of six zeros: lse = log 6
    EXPECT_NEAR(ml::moe_z_loss<C>(ref, z, std::span<double>{}), 0.01 * std::log(6.0) * std::log(6.0), 1e-16);
    EXPECT_EQ(ml::moe_z_loss<C>(ref, std::span<const double>{}, std::span<double>{}), 0.0);
}

TEST(MlMoeLoad, MergesAndReportsTheBalance)
{
    ml::moe_load<4> a, b;
    a.add(std::vector<u32>{4, 4, 4, 4});
    EXPECT_EQ(a.max_over_mean(), 1.0);
    EXPECT_EQ(a.coefficient_of_variation(), 0.0);
    b.add(std::vector<u32>{8, 0, 0, 0});
    a.merge(b);
    EXPECT_EQ(a.total(), 24u);
    EXPECT_EQ(a.max(), 12u);
    EXPECT_EQ(a.min(), 4u);
    EXPECT_EQ(a.mean(), 6.0);
    EXPECT_EQ(a.max_over_mean(), 2.0);
    EXPECT_NEAR(a.coefficient_of_variation(), std::sqrt((36.0 + 3 * 4.0) / 4) / 6, 1e-15);
    const ml::moe_load<4> empty;
    EXPECT_EQ(empty.max_over_mean(), 0.0);
    EXPECT_EQ(empty.coefficient_of_variation(), 0.0);
    static_assert(std::is_trivially_copyable_v<ml::moe_load<64>> && sizeof(ml::moe_load<64>) == 64 * 8);
    static_assert(noexcept(a.merge(b)) && noexcept(a.coefficient_of_variation()));
    static_assert([] {
        ml::moe_load<2> l;
        l.add(std::array<u32, 2>{1, 3});
        return l.max_over_mean() == 1.5 && l.total() == 4;
    }());
}

} // namespace
