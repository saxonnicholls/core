//
//  composed_block_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/composed_block.hpp>: DeepSeek-V3's block in miniature (multi-head latent attention and
//  a mixture of experts with node-limited routing), its buffers sized and laid out by the header's functions, trained
//  for a few SGD steps on a regression over two sequences with its auxiliary losses in the loss, then decoded token by
//  token through a latent cache against the training pass; and the same skeleton composed with a dense MLP. Exits
//  non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/composed_block.hpp>

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

// DeepSeek-V3's block on a small scale: tiny_mla's latent attention; 8 routed experts of 8 in 4 groups, each token's 2
// from its best 2 groups; one shared expert; sigmoid scores times 2.5; the sequence-wise balance loss and the z-loss
constexpr ml::transformer_config v3_small = [] {
    auto c = ml::tiny_mla;
    c.experts = 8;
    c.top_k = 2;
    c.shared_experts = 1;
    c.expert_d_ff = 8;
    c.scoring = ml::moe_scoring::sigmoid;
    c.route_scale = 2.5;
    c.bias_update_rate = 0.01;
    c.balance_loss_weight = 0.001;
    c.z_loss_weight = 0.001;
    c.expert_groups = 4;
    c.group_top_k = 2;
    return c;
}();

} // namespace

int main()
{
    std::printf("snicholls-core composed block demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    //! [composed_block]
    constexpr auto C = v3_small;
    constexpr std::size_t n = 6, sequences = 2, rows = n * sequences, d = C.d_model;
    // Every buffer one allocation, sized by constant expressions and cut into views by the *_in functions
    std::vector<float> params = noise(ml::latent_moe_block_parameter_count(C), 1, 0.3), grads(params.size()), selection(C.experts, 0.0f);
    std::vector<float> values(ml::latent_moe_block_activation_values(C, rows)), work(ml::latent_moe_block_gradient_work(C, rows));
    std::vector<u32> indices(ml::latent_moe_block_activation_indices(C, rows));
    {
        const auto w = ml::latent_moe_block_weights_in<C>(std::span<const float>(params));
        for (const auto norm : {w.norm1_weight, w.norm2_weight, w.attention.q_norm, w.attention.kv_norm})   // norms start at 1
            for (std::size_t i = 0; i < norm.size(); ++i)
                params[std::size_t(norm.data() - params.data()) + i] = 1.0f;
    }
    const auto act = ml::latent_moe_block_activations_in<C>(std::span<float>(values), std::span<u32>(indices), rows);
    const auto g = ml::latent_moe_block_gradients_in<C>(std::span<float>(grads), std::span<float>(work), rows);
    static const ml::rope_table<float, 16, C.qk_rope_dim> rope{ml::rope_base_v<C, float>};

    // Training: two sequences of 6 tokens, the block learning a fixed target; its auxiliary losses join the loss
    const auto input = noise(rows * d, 2, 1.0), target = noise(rows * d, 3, 0.5);
    std::vector<float> xs(rows * d);
    auto step = [&] {
        const auto w = ml::latent_moe_block_weights_in<C>(std::span<const float>(params), std::span<const float>(selection));
        xs = input;
        const auto aux = ml::block_forward_sequence<C>(backend, w, std::span<float>(xs), rope, act, sequences);
        float loss = aux.balance + aux.z;
        std::vector<float> dxs(rows * d);
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const float e = xs[i] - target[i];
            loss += 0.5f * e * e / rows;
            dxs[i] = e / rows;
        }
        std::fill(grads.begin(), grads.end(), 0.0f);
        ml::block_backward<C>(backend, w, rope, act, std::span<float>(dxs), g, sequences);   // dxs: the input's gradient
        for (std::size_t i = 0; i < params.size(); ++i)
            params[i] -= 0.2f * grads[i];
        ml::moe_update_bias<C>(backend, act.mlp.counts, std::span<float>(selection));   // auxiliary-loss-free balancing
        return loss;
    };
    const float first = step();
    float last = first;
    for (int i = 0; i < 40; ++i)
        last = step();
    //! [composed_block]
    std::printf("  DeepSeek-V3's block, %zu parameters: loss %.4f -> %.4f after 40 SGD steps\n", params.size(), static_cast<double>(first),
                static_cast<double>(last));
    ok &= last < 0.8f * first && std::isfinite(last);

    //! [composed_decode]
    // Decoding: one token at a time, the attention absorbed over a latent cache (10 numbers a token), the experts
    // through mlp_forward's overload for moe_weights; the training pass's outputs within rounding
    const auto w = ml::latent_moe_block_weights_in<C>(std::span<const float>(params), std::span<const float>(selection));
    xs = input;
    ml::block_forward_sequence<C>(backend, w, std::span<float>(xs), rope, act, sequences);
    ml::latent_kv_cache<C, float, 1, 16> cache;
    float worst = 0.0f;
    for (std::size_t t = 0; t < n; ++t) {   // the first sequence
        std::vector<float> x(input.begin() + std::ptrdiff_t(t * d), input.begin() + std::ptrdiff_t((t + 1) * d));
        ml::block_forward<C>(w, x, t, cache.view(), 0, rope);
        for (std::size_t i = 0; i < d; ++i)
            worst = std::max(worst, std::abs(x[i] - xs[t * d + i]));
    }
    //! [composed_decode]
    std::printf("  decoding against training: largest difference %.2e\n", static_cast<double>(worst));
    ok &= worst < 1e-5f;

    // The same skeleton with a dense MLP: latent attention and SwiGLU, no auxiliary losses
    {
        constexpr auto D = ml::tiny_mla;
        constexpr std::size_t f = D.d_ff;
        const auto up = noise(f * d, 4, 0.3), gate = noise(f * d, 5, 0.3), down = noise(d * f, 6, 0.3);
        const ml::composed_block_weights<ml::latent_attention_weights<ml::dense_weights<float>, float>, ml::mlp_weights<ml::dense_weights<float>, float>,
                                         float>
            dense{{}, {}, w.attention, {}, {}, {{up, f, d}, {gate, f, d}, {down, d, f}}};
        std::vector<float> x0(rows * d), h1(rows * d), x1(rows * d), h2(rows * d), mu(rows * f), mg(rows * f), mh(rows * f), scratch(rows * d),
            gu(up.size()), gg(gate.size()), gd(down.size()), gs(2 * rows * f), av(ml::latent_attention_activation_values(D, rows)),
            ag(ml::latent_attention_parameter_count(D)), aw(ml::latent_attention_gradient_work(D, rows));
        const ml::composed_block_activations<ml::latent_attention_activations<float>, ml::mlp_activations<float>, float> da{
            x0, h1, x1, h2, ml::latent_attention_activations_in<D>(std::span<float>(av), rows), {mu, mg, mh}};
        ml::mlp_gradients<float> mgr{};
        mgr.up = gu;
        mgr.gate = gg;
        mgr.down = gd;
        mgr.scratch = gs;
        const ml::composed_block_gradients<ml::latent_attention_gradients<float>, ml::mlp_gradients<float>, float> dg{
            {}, {}, {}, {}, ml::latent_attention_gradients_in<D>(std::span<float>(ag), std::span<float>(aw), rows), mgr, scratch};
        auto ys = input;
        const auto none = ml::block_forward_sequence<D>(backend, dense, std::span<float>(ys), rope, da, sequences);
        auto dys = noise(rows * d, 7, 1.0);
        ml::block_backward<D>(backend, dense, rope, da, std::span<float>(dys), dg, sequences);
        std::printf("  with a dense MLP: auxiliary losses %.1f and %.1f, d input[0] = %.4f\n", static_cast<double>(none.balance),
                    static_cast<double>(none.z), static_cast<double>(dys[0]));
        ok &= none.balance == 0.0f && none.z == 0.0f && std::isfinite(dys[0]) && gd[0] != 0.0f;
    }

    std::printf(ok ? "composed block demo: ok\n" : "composed block demo: FAILED\n");
    return ok ? 0 : 1;
}
