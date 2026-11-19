//
//  multi_token_prediction_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/multi_token_prediction.hpp>: one DeepSeek-V3 multi-token prediction module on top of
//  a model's final hidden states: the next-but-one token's embedding combined with them, a latent-attention block, the
//  model's own head; the main loss and the module's weighted loss; the backward pass into the main model's hidden
//  states, the shared embedding table and the shared head. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/latent_block.hpp>
#include <snicholls/core/ml/lm_head.hpp>
#include <snicholls/core/ml/multi_token_prediction.hpp>
#include <snicholls/core/ml/rope.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

std::vector<float> filled(std::size_t n, float phase)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 0.4f * std::sin(0.47f * static_cast<float>(i) + phase);
    return v;
}

} // namespace

int main()
{
    std::printf("snicholls-core multi-token prediction demo\n");
    bool ok = true;
    const auto& backend = snicholls::linalg::reference;

    // The module's block: tiny_mla's latent attention and SwiGLU MLP
    constexpr auto C = ml::tiny_mla;
    constexpr std::size_t d = C.d_model, f = C.d_ff, h = C.n_heads, dqk = C.head_dim, dr = C.qk_rope_dim, dv = C.v_head_dim;
    constexpr std::size_t dc = C.kv_lora_rank, dq = C.q_lora_rank, kvw = C.qk_nope_dim() + dv, n = 6, vocab = 9;
    const auto q_down = filled(dq * d, 0.1f), q_up = filled(h * dqk * dq, 0.2f), kv_down = filled((dc + dr) * d, 0.3f),
               kv_up = filled(h * kvw * dc, 0.4f), o = filled(d * h * dv, 0.5f), up = filled(f * d, 0.6f), gate = filled(f * d, 0.7f),
               down = filled(d * f, 0.8f), projection = filled(d * 2 * d, 0.9f);
    const ml::latent_block_weights<ml::dense_weights<float>, float> block{
        {}, {}, {{q_down, dq, d}, {}, {q_up, h * dqk, dq}, {kv_down, dc + dr, d}, {}, {kv_up, h * kvw, dc}, {o, d, h * dv}, {}},
        {}, {}, {{up, f, d}, {gate, f, d}, {down, d, f}}};
    static const ml::rope_table<float, 16, dr> rope{ml::rope_base_v<C, float>};
    std::vector<float> x0(n * d), h1(n * d), x1(n * d), h2(n * d), mup(n * f), mgate(n * f), mhidden(n * f), cq_raw(n * dq), cq(n * dq),
        q(n * h * dqk), ckv_raw(n * (dc + dr)), ckv(n * dc), kr(n * dr), kv(n * h * kvw), y(n * h * dv), lse(n * h);
    std::vector<float> g_q_down(q_down.size()), g_q_up(q_up.size()), g_kv_down(kv_down.size()), g_kv_up(kv_up.size()), g_o(o.size()),
        g_up(up.size()), g_gate(gate.size()), g_down(down.size()), g_scratch(2 * n * f), scratch(n * d), w_y(y.size()), w_q(q.size()),
        w_kv(kv.size()), w_kr(kr.size()), w_ckv(ckv.size()), w_ckv_raw(ckv_raw.size()), w_cq(cq.size()), w_cq_raw(cq.size());
    ml::mlp_gradients<float> mg{};
    mg.up = g_up;
    mg.gate = g_gate;
    mg.down = g_down;
    mg.scratch = g_scratch;

    //! [mtp]
    // The main model's output for six tokens, its embedding table and its head, shared with the module
    const std::vector<std::uint32_t> ids{4, 2, 7, 1, 8, 3}, targets{2, 7, 1, 8, 3, 5};   // targets[i] follows ids[i]
    const auto hidden = filled(n * d, 1.0f), table = filled(vocab * d, 1.5f), head_rows = filled(vocab * d, 2.5f);
    const ml::dense_weights<float> head{head_rows, vocab, d};
    std::vector<float> logits(n * vocab), losses(n), dhidden(n * d), dtable(vocab * d), dhead(vocab * d);
    const float main_loss = ml::lm_head_loss_sequence(backend, head, hidden, targets, logits, losses, dhidden, dhead);

    // Depth 1: position i combines h_i with the embedding of token i + 1 and predicts token i + 2
    std::vector<std::uint32_t> shifted(n);
    ml::mtp_input_ids(ids, 1, 1, 0, shifted);   // the last position reads the pad, 0, and is not scored
    std::vector<float> embedded(n * d);
    ml::embed_rows(backend, std::span<const float>(table), std::span<const float>{}, shifted, std::span<float>(embedded));
    const ml::mtp_weights<ml::latent_block_weights<ml::dense_weights<float>, float>, float> module{{}, {}, {projection, d, 2 * d}, block, {}};
    std::vector<float> combined(2 * n * d), output(n * d), head_input(n * d);
    const ml::mtp_activations<ml::latent_block_activations<float>, float> act{
        combined, {x0, h1, x1, h2, {cq_raw, cq, q, ckv_raw, ckv, kr, kv, y, lse, {}}, mup, mgate, mhidden}, output, head_input};
    ml::mtp_forward_sequence<C>(backend, module, hidden, embedded, rope, act);

    // Its loss through the shared head, weighted by lambda / D and divided by the whole length (eq. 24)
    const float lambda = ml::mtp_loss_weight_v<ml::deepseek_v3, float> / float(ml::deepseek_v3.mtp_depth);
    std::vector<float> mtp_losses(n), dhead_input(n * d);
    const float mtp_loss = ml::mtp_loss_sequence(backend, head, head_input, targets, 1, logits, mtp_losses, dhead_input, dhead, lambda);

    // Backward: into the main model's hidden states, the shared table (through the embedded rows) and the module
    std::vector<float> dembedded(n * d), dprojection(d * 2 * d), wcombined(2 * n * d), woutput(n * d);
    const ml::mtp_gradients<ml::latent_block_gradients<float>, float> grads{
        {}, {}, dprojection,
        {{}, {}, {}, {}, {g_q_down, {}, g_q_up, g_kv_down, {}, g_kv_up, g_o, {}, w_y, w_q, w_kv, w_kr, w_ckv, w_ckv_raw, w_cq, w_cq_raw}, mg, scratch},
        {}, wcombined, woutput};
    ml::mtp_backward<C>(backend, module, hidden, embedded, rope, act, dhead_input, std::span<const float>{}, dhidden, dembedded, grads);
    ml::embed_rows_backward(backend, std::span<const float>(dembedded), shifted, std::span<float>(dtable), std::span<float>{});
    //! [mtp]

    double gp = 0, gt = 0;
    for (const float g : dprojection)
        gp += static_cast<double>(g) * static_cast<double>(g);
    for (const float g : dtable)
        gt += static_cast<double>(g) * static_cast<double>(g);
    std::printf("  main loss %.4f nats, MTP loss (x lambda %.2f) %.4f; total %.4f\n", static_cast<double>(main_loss),
                static_cast<double>(lambda), static_cast<double>(mtp_loss), static_cast<double>(main_loss + mtp_loss));
    std::printf("  |d projection| = %.4f, |d embedding table| = %.4f\n", std::sqrt(gp), std::sqrt(gt));
    // The weighted loss is lambda times the scored positions' cross-entropies over all six positions
    float sum = 0.0f;
    for (const float l : mtp_losses)
        sum += l;
    ok &= std::fabs(mtp_loss - lambda * sum / float(n)) < 1e-6f && mtp_losses[n - 1] == 0.0f && gp > 0 && gt > 0;
    ok &= main_loss > 0.0f && std::isfinite(dhidden[0]);
    std::printf(ok ? "multi-token prediction demo: ok\n" : "multi-token prediction demo: FAILED\n");
    return ok ? 0 : 1;
}
