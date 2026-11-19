//
//  moe_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/moe.hpp>: routing three tokens in DeepSeek-V3's style and Mixtral's; the permutation,
//  gather, grouped products and combine one by one; the layer trained for a few SGD steps on a regression, its loss
//  falling; and auxiliary-loss-free balancing moving a skewed router toward uniform load. Exits non-zero if a result
//  is not what it documents.
//

#include <snicholls/core/ml/moe.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
using u32 = std::uint32_t;

std::vector<float> noise(std::size_t n, std::uint64_t seed, double scale)
{
    auto g = stats::wyrand::stream(seed, 0);
    std::vector<float> v(n);
    for (auto& e : v)
        e = static_cast<float>(scale * (2.0 * stats::uniform_real(g) - 1.0));
    return v;
}

// DeepSeek-V3's routing on a small scale: 8 experts, each token to 2, one shared expert, sigmoid scores times 2.5, a
// selection bias stepped by 0.01, a small sequence-wise balance loss and ST-MoE's z-loss
constexpr ml::transformer_config ds{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 16, .mlp = ml::mlp_kind::swiglu,
                                    .experts = 8, .top_k = 2, .shared_experts = 1, .expert_d_ff = 16, .scoring = ml::moe_scoring::sigmoid,
                                    .route_scale = 2.5, .bias_update_rate = 0.01, .balance_loss_weight = 0.001, .z_loss_weight = 0.001};
// Switch's softmax over every expert and Mixtral's softmax over the chosen two, on the same shape
constexpr ml::transformer_config sw = [] { auto c = ds; c.scoring = ml::moe_scoring::softmax; c.route_scale = 1; return c; }();
constexpr ml::transformer_config mx = [] { auto c = ds; c.scoring = ml::moe_scoring::softmax_top_k; c.route_scale = 1; return c; }();

} // namespace

int main()
{
    std::printf("snicholls-core moe demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    //! [route]
    // Three tokens' router logits over 8 experts; the bias favours expert 7 for the choice only
    const std::vector<float> logits = noise(3 * 8, 1, 2.0);
    std::vector<float> bias(8, 0.0f);
    bias[7] = 5.0f;
    std::vector<u32> experts(3 * 2);
    std::vector<float> weights(3 * 2);
    ml::moe_route_rows<ds>(backend, logits, bias, experts, std::span<float>(weights));
    // Every token goes to expert 7, weighted by its own sigmoid; the weights of a token sum to route_scale
    //! [route]
    for (std::size_t t = 0; t < 3; ++t)
        std::printf("  token %zu: experts %u, %u, weights %.3f, %.3f\n", t, experts[2 * t], experts[2 * t + 1], static_cast<double>(weights[2 * t]),
                    static_cast<double>(weights[2 * t + 1]));
    for (std::size_t t = 0; t < 3; ++t)
        ok &= experts[2 * t] == 7 && std::abs(weights[2 * t] + weights[2 * t + 1] - 2.5f) < 1e-5f;

    // The other two scorings, and every scoring's backward to the logits
    std::vector<float> wsw(6), wmx(6);
    std::vector<u32> esw(6), emx(6);
    ml::moe_route_rows<sw>(backend, logits, {}, esw, std::span<float>(wsw));
    ml::moe_route_rows<mx>(backend, logits, {}, emx, std::span<float>(wmx));
    ok &= esw == emx;   // the same choice: a softmax keeps the logits' order
    ok &= std::abs(wmx[0] + wmx[1] - 1.0f) < 1e-6f && wsw[0] + wsw[1] < 1.0f;
    const std::vector<float> dw{1, 0, 0, 1, 1, 1};
    std::vector<float> dz_ds(24), dz_sw(24), dz_mx(24);
    ml::moe_route_rows_backward<ds>(backend, logits, experts, dw, std::span<float>(dz_ds));
    ml::moe_route_rows_backward<sw>(backend, logits, esw, dw, std::span<float>(dz_sw));
    ml::moe_route_rows_backward<mx>(backend, logits, emx, dw, std::span<float>(dz_mx));
    // Token 2's weights get equal gradients: under Mixtral's softmax over the chosen, moving both together changes nothing
    ok &= dz_mx[16 + emx[4]] == 0.0f && dz_mx[16 + emx[5]] == 0.0f && dz_ds[7] != 0.0f && dz_sw[0] != 0.0f;

    // The steps of the layer one by one: count, offsets, permutation, gather, the experts, combine
    {
        constexpr std::size_t d = ds.d_model, f = ds.expert_d_ff, n = 6;
        std::vector<u32> counts(8), offsets(9), order(n), inverse(n);
        ml::moe_count_rows<ds>(backend, experts, counts);
        ml::moe_offsets<ds>(backend, counts, offsets);
        ml::moe_permutation<ds>(backend, experts, offsets, order, inverse);
        std::printf("  expert 7 took %u of the %zu choices; permuted order", counts[7], n);
        for (const u32 j : order)
            std::printf(" %u", j);
        std::printf("\n");
        ok &= counts[7] == 3 && offsets[8] == n && order[offsets[7]] == 0;
        const auto xs = noise(3 * d, 2, 1.0), up = noise(8 * f * d, 3, 0.3), gate = noise(8 * f * d, 4, 0.3), down = noise(8 * d * f, 5, 0.3);
        std::vector<float> xp(n * d), yp(n * d), ys(3 * d), su(n * f), sg(n * f), sh(n * f);
        const ml::mlp_weights<ml::grouped_weights<float>, float> w{{up, 8, f, d}, {gate, 8, f, d}, {down, 8, d, f}};
        ml::moe_gather_rows<ds>(backend, xs, order, std::span<float>(xp));
        ml::moe_experts_forward<ds>(backend, w, offsets, xp, yp, {su, sg, sh});
        ml::moe_combine_rows<ds>(backend, yp, weights, inverse, std::span<float>(ys));
        // Backward: combine, experts, gather
        const auto dy = noise(3 * d, 6, 1.0);
        std::vector<float> dyp(n * d), dws(n), dxp(n * d), dxs(3 * d), gu(up.size()), gg(gate.size()), gd(down.size()), scratch(2 * n * f);
        ml::moe_combine_rows_backward<ds>(backend, dy, yp, weights, inverse, dyp, std::span<float>(dws));
        ml::moe_experts_backward<ds>(backend, w, offsets, xp, {su, sg, sh}, dyp, dxp, {gu, gg, gd, {}, {}, {}, scratch});
        ml::moe_gather_rows_backward<ds>(backend, std::span<const float>(dxp), inverse, std::span<float>(dxs));
        // One expert's product is linear_sequence on its segment; a run of experts is a slice
        std::vector<float> y7(3 * f);
        ml::grouped_linear(backend, w.up.slice(7, 1), std::vector<u32>{0, 3}, std::span<const float>(xp).subspan(offsets[7] * d, 3 * d), y7);
        std::vector<float> g7(f * d);
        ml::grouped_linear_backward(backend, w.up.slice(7, 1), std::vector<u32>{0, 3}, std::span<const float>(xp).subspan(offsets[7] * d, 3 * d),
                                    std::span<const float>(y7), {}, g7);
        const auto r = ml::moe_expert_range(w, 6, 2);
        const auto gr = ml::moe_expert_range<ds>(ml::mlp_gradients<float>{gu, gg, gd, {}, {}, {}, scratch}, 6, 2);
        ok &= y7[0] == su[offsets[7] * f] && r.up.experts == 2 && gr.up.size() == 2 * f * d && std::isfinite(dxs[0]) && std::isfinite(g7[0]);
        ok &= ml::moe_expert_config(ds).d_ff == f && ml::moe_shared_config(ds).d_ff == f && w.up.expert(3).rows == f;
    }

    //! [layer]
    // The layer over 32 rows (four sequences of 8) learning to map xs to a fixed target by SGD
    constexpr std::size_t rows = 32, d = ds.d_model;
    std::vector<float> params = noise(ml::moe_parameter_count(ds), 7, 0.3), grads(params.size()), selection(ds.experts, 0.0f);
    std::vector<float> values(ml::moe_activation_values(ds, rows)), work(ml::moe_gradient_work(ds, rows));
    std::vector<u32> indices(ml::moe_activation_indices(ds, rows));
    const auto xs = noise(rows * d, 8, 1.0), target = noise(rows * d, 9, 0.5);
    std::vector<float> ys(rows * d), dys(rows * d), dxs(rows * d);
    const auto act = ml::moe_activations_in<ds>(std::span<float>(values), std::span<u32>(indices), rows);
    const auto g = ml::moe_gradients_in<ds>(std::span<float>(grads), std::span<float>(work), rows);
    auto step = [&] {
        const auto w = ml::moe_weights_in<ds>(std::span<const float>(params), std::span<const float>(selection));
        const auto aux = ml::moe_forward<ds>(backend, w, xs, ys, act, 4);
        float loss = aux.balance + aux.z;
        for (std::size_t i = 0; i < ys.size(); ++i) {
            const float e = ys[i] - target[i];
            loss += 0.5f * e * e / rows;
            dys[i] = e / rows;
        }
        std::fill(grads.begin(), grads.end(), 0.0f);
        ml::moe_backward<ds>(backend, w, xs, act, dys, dxs, g, 4);
        for (std::size_t i = 0; i < params.size(); ++i)
            params[i] -= 0.5f * grads[i];
        ml::moe_update_bias<ds>(backend, act.counts, std::span<float>(selection));   // the load, toward uniform
        return loss;
    };
    const float first = step();
    float last = first;
    for (int i = 0; i < 60; ++i)
        last = step();
    //! [layer]
    std::printf("  layer: loss %.4f -> %.4f after 60 SGD steps\n", static_cast<double>(first), static_cast<double>(last));
    ok &= last < 0.7f * first;

    //! [balance]
    // A router that prefers the low-numbered experts; DeepSeek-V3's bias rule evens the load out over 300 steps
    constexpr std::size_t tokens = 512;
    auto skewed = noise(tokens * 8, 10, 0.5);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t e = 0; e < 8; ++e)
            skewed[t * 8 + e] += 0.4f * (3.5f - float(e));
    std::vector<float> b(8, 0.0f), wts(tokens * 2);
    std::vector<u32> chosen(tokens * 2), load(8);
    ml::moe_load<8> before, after;
    for (int s = 0; s <= 300; ++s) {
        ml::moe_route_rows<ds>(backend, skewed, b, chosen, std::span<float>(wts));
        ml::moe_count_rows<ds>(backend, chosen, load);
        ml::moe_update_bias<ds>(backend, load, std::span<float>(b));
        if (s == 0)
            before.add(load);
        else if (s > 200)
            after.add(load);   // the last 100 steps
    }
    // The auxiliary losses on the last step's routing, and their gradients added into the logits'
    std::vector<float> dlogits(tokens * 8, 0.0f);
    const float balance = ml::moe_balance_loss<ds>(backend, skewed, chosen, std::span<float>(dlogits), 4);
    const float z = ml::moe_z_loss<ds>(backend, skewed, std::span<float>(dlogits));
    //! [balance]
    std::printf("  load max/mean %.2f, cv %.2f at the start; %.2f, %.2f over the last 100 steps (min %llu, max %llu of %llu)\n", before.max_over_mean(),
                before.coefficient_of_variation(), after.max_over_mean(), after.coefficient_of_variation(),
                static_cast<unsigned long long>(after.min()), static_cast<unsigned long long>(after.max()),
                static_cast<unsigned long long>(after.total()));
    std::printf("  balance loss %.5f (alpha at uniform: 0.001), z-loss %.5f, mean load %.1f\n", static_cast<double>(balance),
                static_cast<double>(z), after.mean());
    ml::moe_load<8> merged = before;
    merged.merge(after);
    ok &= before.max_over_mean() > 1.5 && after.max_over_mean() < 1.15 && after.coefficient_of_variation() < before.coefficient_of_variation() / 3;
    ok &= merged.total() == before.total() + after.total() && balance > 0 && z > 0 && std::isfinite(dlogits[0]);

    std::printf(ok ? "moe demo: ok\n" : "moe demo: FAILED\n");
    return ok ? 0 : 1;
}
