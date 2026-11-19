//
//  latent_block_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/latent_block.hpp>: a block with multi-head latent attention and a SwiGLU MLP, trained
//  over a sequence through the reference backend (forward and backward), then decoded token by token through a latent
//  cache, which gives the training pass's residual stream within rounding. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/latent_block.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

std::vector<float> filled(std::size_t n, float phase)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 0.4f * std::sin(0.53f * static_cast<float>(i) + phase);
    return v;
}

} // namespace

int main()
{
    std::printf("snicholls-core latent block demo\n");
    bool ok = true;

    //! [latent_block]
    constexpr auto C = ml::tiny_mla;
    constexpr std::size_t d = C.d_model, f = C.d_ff, h = C.n_heads, dqk = C.head_dim, dr = C.qk_rope_dim, dv = C.v_head_dim;
    constexpr std::size_t dc = C.kv_lora_rank, dq = C.q_lora_rank, kvw = C.qk_nope_dim() + dv, n = 6;
    const auto q_down = filled(dq * d, 0.1f), q_up = filled(h * dqk * dq, 0.2f), kv_down = filled((dc + dr) * d, 0.3f),
               kv_up = filled(h * kvw * dc, 0.4f), o = filled(d * h * dv, 0.5f), up = filled(f * d, 0.6f), gate = filled(f * d, 0.7f),
               down = filled(d * f, 0.8f);
    const ml::latent_block_weights<ml::dense_weights<float>, float> w{
        {}, {}, {{q_down, dq, d}, {}, {q_up, h * dqk, dq}, {kv_down, dc + dr, d}, {}, {kv_up, h * kvw, dc}, {o, d, h * dv}, {}},
        {}, {}, {{up, f, d}, {gate, f, d}, {down, d, f}}};
    static const ml::rope_table<float, 16, dr> rope{ml::rope_base_v<C, float>};

    // Training: the block over the sequence in place, saving its activations, then the backward pass
    std::vector<float> x0(n * d), h1(n * d), x1(n * d), h2(n * d), mup(n * f), mgate(n * f), mhidden(n * f);
    std::vector<float> cq_raw(n * dq), cq(n * dq), q(n * h * dqk), ckv_raw(n * (dc + dr)), ckv(n * dc), kr(n * dr), kv(n * h * kvw),
        y(n * h * dv), lse(n * h);
    const ml::latent_block_activations<float> act{x0, h1, x1, h2, {cq_raw, cq, q, ckv_raw, ckv, kr, kv, y, lse, {}}, mup, mgate, mhidden};
    const auto input = filled(n * d, 1.0f);
    auto xs = input;
    ml::block_forward_sequence<C>(snicholls::linalg::reference, w, std::span<float>(xs), rope, act);

    std::vector<float> g_q_down(q_down.size()), g_q_up(q_up.size()), g_kv_down(kv_down.size()), g_kv_up(kv_up.size()), g_o(o.size()),
        g_up(up.size()), g_gate(gate.size()), g_down(down.size()), g_scratch(2 * n * f), scratch(n * d);
    std::vector<float> w_y(y.size()), w_q(q.size()), w_kv(kv.size()), w_kr(kr.size()), w_ckv(ckv.size()), w_ckv_raw(ckv_raw.size()),
        w_cq(cq.size()), w_cq_raw(cq.size());
    ml::mlp_gradients<float> mg{};
    mg.up = g_up;
    mg.gate = g_gate;
    mg.down = g_down;
    mg.scratch = g_scratch;
    const ml::latent_block_gradients<float> grads{
        {}, {}, {}, {}, {g_q_down, {}, g_q_up, g_kv_down, {}, g_kv_up, g_o, {}, w_y, w_q, w_kv, w_kr, w_ckv, w_ckv_raw, w_cq, w_cq_raw}, mg, scratch};
    auto dxs = filled(n * d, 2.0f);   // the gradient in the block's output, replaced by the gradient in its input
    ml::block_backward<C>(snicholls::linalg::reference, w, rope, act, std::span<float>(dxs), grads);

    // Decoding: the same block one token at a time, its attention absorbed, its cache 10 numbers a token
    ml::latent_kv_cache<C, float, 1, 16> cache;
    float worst = 0.0f;
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<float> x(input.begin() + std::ptrdiff_t(t * d), input.begin() + std::ptrdiff_t((t + 1) * d));
        ml::block_forward<C>(w, x, t, cache.view(), 0, rope);
        for (std::size_t i = 0; i < d; ++i)
            worst = std::fmax(worst, std::fabs(x[i] - xs[t * d + i]));
    }
    //! [latent_block]

    double gn = 0;
    for (const float g : g_down)
        gn += static_cast<double>(g) * static_cast<double>(g);
    std::printf("  backward: |d down| = %.4f, d input[0] = %.4f\n", std::sqrt(gn), static_cast<double>(dxs[0]));
    std::printf("  decoding against training: largest difference %.2e\n", static_cast<double>(worst));
    ok &= worst < 1e-5f && gn > 0 && std::isfinite(dxs[0]);
    std::printf(ok ? "latent block demo: ok\n" : "latent block demo: FAILED\n");
    return ok ? 0 : 1;
}
