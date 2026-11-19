//
//  latent_attention_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/latent_attention.hpp>: multi-head latent attention on a small model: the training
//  passes over a batch of two sequences through the reference backend, then decoding the first sequence token by token
//  in the absorbed form through a latent cache, which gives the training form's outputs within rounding; and the KV
//  cache's size against multi-head attention's. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/latent_attention.hpp>
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
        v[i] = 0.5f * std::sin(0.61f * static_cast<float>(i) + phase);
    return v;
}

} // namespace

int main()
{
    std::printf("snicholls-core latent attention demo\n");
    bool ok = true;

    //! [latent_training]
    // tiny_mla: 16 wide, 2 heads of 4 plain + 2 rotary query-key channels and 4 value channels, latents of 6 and 8
    constexpr auto C = ml::tiny_mla;
    constexpr std::size_t d = C.d_model, h = C.n_heads, dqk = C.head_dim, dr = C.qk_rope_dim, dv = C.v_head_dim;
    constexpr std::size_t dc = C.kv_lora_rank, dq = C.q_lora_rank, kvw = C.qk_nope_dim() + dv, n = 5, sequences = 2, rows = n * sequences;
    const auto q_down = filled(dq * d, 0.1f), q_up = filled(h * dqk * dq, 0.2f), kv_down = filled((dc + dr) * d, 0.3f),
               kv_up = filled(h * kvw * dc, 0.4f), o = filled(d * h * dv, 0.5f);
    const ml::latent_attention_weights<ml::dense_weights<float>, float> w{
        {q_down, dq, d}, {}, {q_up, h * dqk, dq}, {kv_down, dc + dr, d}, {}, {kv_up, h * kvw, dc}, {o, d, h * dv}, {}};
    static const ml::rope_table<float, 16, dr> rope{ml::rope_base_v<C, float>};   // the rotary channels only

    // The activations a backward pass reads, for every row of the batch
    std::vector<float> cq_raw(rows * dq), cq(rows * dq), q(rows * h * dqk), ckv_raw(rows * (dc + dr)), ckv(rows * dc), k_rope(rows * dr),
        kv(rows * h * kvw), y(rows * h * dv), lse(rows * h);
    const ml::latent_attention_activations<float> act{cq_raw, cq, q, ckv_raw, ckv, k_rope, kv, y, lse, {}};
    const auto xs = filled(rows * d, 1.0f);
    std::vector<float> out(rows * d);
    ml::latent_attention_forward_sequence<C>(snicholls::linalg::reference, w, xs, rope, act, out, sequences);

    // The backward pass: every weight's gradient added into, the input's too
    std::vector<float> g_q_down(q_down.size()), g_q_up(q_up.size()), g_kv_down(kv_down.size()), g_kv_up(kv_up.size()), g_o(o.size());
    std::vector<float> w_y(y.size()), w_q(q.size()), w_kv(kv.size()), w_kr(k_rope.size()), w_ckv(ckv.size()), w_ckv_raw(ckv_raw.size()),
        w_cq(cq.size()), w_cq_raw(cq.size()), dxs(xs.size());
    const ml::latent_attention_gradients<float> grads{g_q_down, {}, g_q_up, g_kv_down, {}, g_kv_up, g_o, {},
                                                      w_y,      w_q, w_kv,   w_kr,      w_ckv, w_ckv_raw, w_cq, w_cq_raw};
    const auto dout = filled(rows * d, 2.0f);
    ml::latent_attention_backward<C>(snicholls::linalg::reference, w, xs, rope, act, dout, dxs, grads, sequences);
    //! [latent_training]

    //! [latent_decode]
    // Decoding: each token's latent and rotary key (8 + 2 numbers) cached, each head's query taken into the latent space
    ml::latent_kv_cache<C, float, 1, 16> cache;
    float worst = 0.0f;
    for (std::size_t t = 0; t < n; ++t) {
        std::vector<float> token_out(d);
        ml::latent_attention_forward<C>(w, std::span<const float>(xs).subspan(t * d, d), t, cache.view(), 0, rope, token_out);
        for (std::size_t i = 0; i < d; ++i)
            worst = std::fmax(worst, std::fabs(token_out[i] - out[t * d + i]));
    }
    // The cache per token and layer: DeepSeek-V3 keeps 576 numbers where multi-head attention would keep 32,768
    constexpr std::size_t mla = ml::kv_cache_elements_per_token(ml::deepseek_v3), mha = 2 * ml::deepseek_v3.n_heads * 128;
    //! [latent_decode]

    //! [latent_buffers]
    // The same layer from one buffer of parameters, one of activations and one of working space, sized by constant
    // expressions and cut into views (the norms' weights included: here 1, the parameter-free norm's values)
    std::vector<float> params;
    for (const auto* t : {&q_down, &q_up, &kv_down, &kv_up, &o})
        params.insert(params.end(), t->begin(), t->end());
    params.insert(params.begin() + std::ptrdiff_t(dq * d), dq, 1.0f);                            // q_norm after q_down
    params.insert(params.begin() + std::ptrdiff_t(dq * d + dq + q_up.size() + kv_down.size()), dc, 1.0f);   // kv_norm after kv_down
    std::vector<float> values(ml::latent_attention_activation_values(C, rows)), pgrads(params.size()),
        work(ml::latent_attention_gradient_work(C, rows)), out2(rows * d), dxs2(rows * d);
    const auto w2 = ml::latent_attention_weights_in<C>(std::span<const float>(params));
    const auto act2 = ml::latent_attention_activations_in<C>(std::span<float>(values), rows);
    ml::latent_attention_forward_sequence<C>(snicholls::linalg::reference, w2, xs, rope, act2, out2, sequences);
    ml::latent_attention_backward<C>(snicholls::linalg::reference, w2, xs, rope, act2, dout, dxs2,
                                     ml::latent_attention_gradients_in<C>(std::span<float>(pgrads), std::span<float>(work), rows), sequences);
    //! [latent_buffers]
    std::printf("  one buffer of %zu parameters: the same outputs %s\n", params.size(), out2 == out ? "to the bit" : "NOT");
    ok &= params.size() == ml::latent_attention_parameter_count(C) && out2 == out && dxs2 == dxs;

    double grad_norm = 0;
    for (const float g : g_kv_up)
        grad_norm += static_cast<double>(g) * static_cast<double>(g);
    std::printf("  training: %zu rows, |d kv_up| = %.4f\n", rows, std::sqrt(grad_norm));
    std::printf("  decoding in the absorbed form: largest difference from training %.2e\n", static_cast<double>(worst));
    std::printf("  KV cache per token and layer: MLA %zu, MHA %zu (%.1f%%)\n", mla, mha, 100.0 * static_cast<double>(mla) / static_cast<double>(mha));
    ok &= worst < 1e-5f && grad_norm > 0 && mla == 576 && ml::kv_cache_elements(C, 1, 16) == 16 * 10;
    std::printf(ok ? "latent attention demo: ok\n" : "latent attention demo: FAILED\n");
    return ok ? 0 : 1;
}
