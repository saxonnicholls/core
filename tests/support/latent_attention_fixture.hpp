//
//  latent_attention_fixture.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  A latent attention's (and a latent-attention block's) weights, activations and gradients for the tests of ml/latent_attention.hpp,
//  ml/latent_block.hpp and ml/multi_token_prediction.hpp: deterministic data, a batch of sequences, the reference
//  backend's passes, and the sum of the output weighted by a fixed gradient as the loss central differences read.
//

#ifndef snicholls_core_tests_support_latent_attention_fixture_hpp
#define snicholls_core_tests_support_latent_attention_fixture_hpp

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/latent_attention.hpp>
#include <snicholls/core/ml/latent_block.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/sparse_attention.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace snicholls::core_testing
{

namespace ml = snicholls::ml;
using snicholls::linalg::reference;

inline constexpr std::size_t positions = 8;

template <ml::transformer_config C, typename T>
const auto& table()
{
    static const ml::rope_table<T, positions, C.qk_rope_dim> t{ml::rope_base_v<C, T>, ml::rope_scaling_v<C, T>};
    return t;
}

template <typename T>
std::vector<T> wave(std::size_t n, double a, double b)
{
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<T>(a * std::sin(0.61 * static_cast<double>(i) + b));
    return v;
}

// One latent attention's weights, activations and gradients for `sequences` sequences of `n` tokens
template <ml::transformer_config C, typename T>
struct latent_problem
{
    static constexpr std::size_t h = C.n_heads, dqk = C.head_dim, dn = C.qk_nope_dim(), dr = C.qk_rope_dim, dv = C.v_head_dim,
                                 dc = C.kv_lora_rank, dq = C.q_lora_rank, d = C.d_model, kvw = dn + dv;
    std::size_t n, sequences, rows;
    std::vector<T> q_down, q_norm, q_up, kv_down, kv_norm, kv_up, o, sink;
    std::vector<T> xs, out, dout, dxs;
    std::vector<T> cq_raw, cq, q, ckv_raw, ckv, k_rope, kv, y, lse;
    std::vector<std::uint32_t> selected;
    std::vector<T> g_q_down, g_q_norm, g_q_up, g_kv_down, g_kv_norm, g_kv_up, g_o, g_sink;
    std::vector<T> w_y, w_q, w_kv, w_k_rope, w_ckv, w_ckv_raw, w_cq, w_cq_raw;

    latent_problem(std::size_t tokens, std::size_t seqs)
        : n(tokens), sequences(seqs), rows(tokens * seqs), q_down(wave<T>(dq * d, 0.5, 0.1)), q_norm(wave<T>(dq, 0.2, 0.7)),
          q_up(wave<T>(h * dqk * (dq > 0 ? dq : d), 0.5, 1.3)), kv_down(wave<T>((dc + dr) * d, 0.5, 2.9)),
          kv_norm(wave<T>(dc, 0.2, 0.4)), kv_up(wave<T>(h * kvw * dc, 0.5, 3.7)), o(wave<T>(d * h * dv, 0.4, 0.8)),
          sink(wave<T>(h, 0.8, 0.5)), xs(wave<T>(rows * d, 1.0, 0.25)), out(rows * d), dout(wave<T>(rows * d, 1.0, 1.9)),
          dxs(rows * d), cq_raw(rows * dq), cq(rows * dq), q(rows * h * dqk), ckv_raw(rows * (dc + dr)), ckv(rows * dc),
          k_rope(rows * dr), kv(rows * h * kvw), y(rows * h * dv), lse(rows * h), g_q_down(q_down.size()), g_q_norm(dq),
          g_q_up(q_up.size()), g_kv_down(kv_down.size()), g_kv_norm(dc), g_kv_up(kv_up.size()), g_o(o.size()), g_sink(h),
          w_y(y.size()), w_q(q.size()), w_kv(kv.size()), w_k_rope(k_rope.size()), w_ckv(ckv.size()), w_ckv_raw(ckv_raw.size()),
          w_cq(cq.size()), w_cq_raw(cq.size())
    {
        for (T& x : q_norm)
            x += T(1);
        for (T& x : kv_norm)
            x += T(1);
        if constexpr (C.sparse_top_k > 0) {   // each token's keys chosen by select_top_k over a score per visible key
            selected.assign(rows * C.sparse_top_k, 0);
            for (std::size_t s = 0; s < sequences; ++s)
                for (std::size_t t = 0; t < n; ++t) {
                    const ml::key_range r = ml::attention_window(t, C.window);
                    std::vector<double> score(r.size());
                    for (std::size_t j = 0; j < r.size(); ++j)
                        score[j] = std::cos(1.7 * static_cast<double>(r.first + j + 3 * t + s));
                    const auto sel = std::span<std::uint32_t>(selected).subspan((s * n + t) * C.sparse_top_k, C.sparse_top_k);
                    const std::size_t k = ml::select_top_k(score, sel);
                    for (std::size_t j = 0; j < k; ++j)
                        sel[j] += static_cast<std::uint32_t>(r.first);
                }
        }
    }

    [[nodiscard]] ml::latent_attention_weights<ml::dense_weights<T>, T> weights() const
    {
        return {{q_down, dq, d},          q_norm, {q_up, h * dqk, dq > 0 ? dq : d}, {kv_down, dc + dr, d}, kv_norm,
                {kv_up, h * kvw, dc}, {o, d, h * dv}, C.attention_sinks ? std::span<const T>(sink) : std::span<const T>{}};
    }
    [[nodiscard]] ml::latent_attention_activations<T> act()
    {
        return {cq_raw, cq, q, ckv_raw, ckv, k_rope, kv, y, lse, selected};
    }
    [[nodiscard]] ml::latent_attention_gradients<T> grads()
    {
        return {g_q_down, g_q_norm, g_q_up, g_kv_down, g_kv_norm, g_kv_up, g_o, C.attention_sinks ? std::span<T>(g_sink) : std::span<T>{},
                w_y, w_q, w_kv, w_k_rope, w_ckv, w_ckv_raw, w_cq, w_cq_raw};
    }
    void forward() { ml::latent_attention_forward_sequence<C>(reference, weights(), xs, table<C, T>(), act(), out, sequences); }
    void backward()
    {
        ml::latent_attention_backward<C>(reference, weights(), xs, table<C, T>(), act(), dout, dxs, grads(), sequences);
    }
    [[nodiscard]] T loss()
    {
        forward();
        T l = T(0);
        for (std::size_t i = 0; i < out.size(); ++i)
            l += dout[i] * out[i];
        return l;
    }
};

// A latent-attention block's weights, activations and gradients around a latent_problem
template <ml::transformer_config C, typename T>
struct block_problem
{
    static constexpr std::size_t d = C.d_model, f = C.d_ff;
    latent_problem<C, T> attn;
    std::size_t rows;
    std::vector<T> norm1, norm2, up, gate, down, xs, dxs, x0, h1, x1, h2, mup, mgate, mhidden;
    std::vector<T> g_norm1, g_norm2, g_up, g_gate, g_down, g_scratch, scratch;

    block_problem(std::size_t n, std::size_t sequences)
        : attn(n, sequences), rows(n * sequences), norm1(wave<T>(d, 0.2, 0.3)), norm2(wave<T>(d, 0.2, 1.4)), up(wave<T>(f * d, 0.4, 0.6)),
          gate(wave<T>(f * d, 0.4, 2.2)), down(wave<T>(d * f, 0.4, 1.1)), xs(wave<T>(rows * d, 1.0, 0.9)), dxs(wave<T>(rows * d, 1.0, 2.4)),
          x0(rows * d), h1(rows * d), x1(rows * d), h2(rows * d), mup(rows * f), mgate(rows * f), mhidden(rows * f), g_norm1(d), g_norm2(d),
          g_up(f * d), g_gate(f * d), g_down(d * f), g_scratch(2 * rows * f), scratch(rows * d)
    {
        for (T& x : norm1)
            x += T(1);
        for (T& x : norm2)
            x += T(1);
    }
    [[nodiscard]] ml::latent_block_weights<ml::dense_weights<T>, T> weights() const
    {
        return {norm1, {}, attn.weights(), norm2, {}, {{up, f, d}, {gate, f, d}, {down, d, f}}};
    }
    [[nodiscard]] ml::latent_block_activations<T> act() { return {x0, h1, x1, h2, attn.act(), mup, mgate, mhidden}; }
    [[nodiscard]] ml::latent_block_gradients<T> grads()
    {
        ml::mlp_gradients<T> mg{};
        mg.up = g_up;
        mg.gate = g_gate;
        mg.down = g_down;
        mg.scratch = g_scratch;
        return {g_norm1, {}, g_norm2, {}, attn.grads(), mg, scratch};
    }
    // The block over every row, from a copy of xs
    std::vector<T> forward()
    {
        auto s = xs;
        ml::block_forward_sequence<C>(reference, weights(), std::span<T>(s), table<C, T>(), act(), attn.sequences);
        return s;
    }
};

} // namespace snicholls::core_testing

#endif // snicholls_core_tests_support_latent_attention_fixture_hpp
