//
//  transformer_block_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/transformer_block.hpp>: a two-layer stack of small blocks (RMS norm, grouped-query
//  attention with RoPE, SwiGLU) prefilling a prompt and then decoding one more token, and the norm each configuration
//  applies. Prefill and decode are one code path, so prefilling the prompt and decoding it token by token agree to the
//  bit; then one block trained by gradient descent, its loss falling. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/ml/transformer_block.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <type_traits>
#include <vector>


//! [own_backend]
// A device library's backend runs the batched passes' steps itself: it overloads core's operations for its type, in
// its namespace, and the passes call those (argument-dependent lookup). Here one step, counted, on the host.
namespace my_device {
struct backend : snicholls::linalg::reference_backend {
    inline static int norms = 0;
};

template <snicholls::ml::transformer_config C, typename T>
void block_norm_rows(const backend& b, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<const T>> weight,
                     std::type_identity_t<std::span<const T>> bias, std::span<T> out)
{
    ++backend::norms;   // a device would launch its kernel here, on data it already holds
    snicholls::ml::block_norm_rows<C>(static_cast<const snicholls::linalg::reference_backend&>(b), x, weight, bias, out);
}
} // namespace my_device
//! [own_backend]

namespace {

namespace ml = snicholls::ml;
using W = ml::dense_weights<float>;

constexpr ml::transformer_config c{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 16, .norm = ml::norm_kind::rms,
                                   .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half};

std::vector<float> pattern(std::size_t n, float scale, std::size_t salt)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * (float((i * 37 + salt * 11) % 23) / 23.0f - 0.5f);
    return v;
}

// One layer's weights, owned here so the views stay valid
struct layer {
    std::vector<float> q, kv, o, up, gate, down;
    explicit layer(std::size_t salt)
        : q(pattern(8 * 8, 0.6f, salt)), kv(pattern(4 * 8, 0.6f, salt + 1)), o(pattern(8 * 8, 0.6f, salt + 2)), up(pattern(16 * 8, 0.6f, salt + 3)),
          gate(pattern(16 * 8, 0.6f, salt + 4)), down(pattern(8 * 16, 0.3f, salt + 5))
    {
    }
    ml::block_weights<W> weights() const
    {
        ml::block_weights<W> w{};
        w.attention = {W{q, 8, 8}, W{kv, 4, 8}, W{kv, 4, 8}, W{o, 8, 8}};
        w.mlp = {W{up, 16, 8}, W{gate, 16, 8}, W{down, 8, 16}};
        return w;
    }
};

} // namespace

int main()
{
    std::printf("snicholls-core transformer_block demo\n");
    bool ok = true;
    const layer l0(1), l1(10);

    //! [transformer_block]
    // Two layers sharing one cache, 16 positions each; a prompt of 3 tokens' residual streams, then a fourth
    static const ml::rope_table<float, 16, 4> rope{};
    std::vector<float> keys(ml::kv_cache_view<c>::required_size(2, 16)), values(keys.size());
    const ml::kv_cache_view<c> cache(keys, values, 2, 16);
    const std::array<ml::block_weights<W>, 2> layers{l0.weights(), l1.weights()};

    std::vector<float> prompt = pattern(3 * 8, 2.0f, 20);
    for (std::size_t i = 0; i < layers.size(); ++i)
        ml::block_prefill<c>(layers[i], prompt, 0, cache, i, rope);   // the whole prompt through layer i

    std::vector<float> next = pattern(8, 2.0f, 30);
    for (std::size_t i = 0; i < layers.size(); ++i)
        ml::block_forward<c>(layers[i], next, 3, cache, i, rope);     // one decode step at position 3
    //! [transformer_block]

    // The same prompt decoded token by token into a second cache: the same bits
    std::vector<float> k2(keys.size()), v2(keys.size());
    const ml::kv_cache_view<c> cache2(k2, v2, 2, 16);
    std::vector<float> stepped = pattern(3 * 8, 2.0f, 20);
    for (std::size_t t = 0; t < 3; ++t)
        for (std::size_t i = 0; i < layers.size(); ++i)
            ml::block_forward<c>(layers[i], std::span<float>(stepped).subspan(t * 8, 8), t, cache2, i, rope);

    std::vector<float> next2 = pattern(8, 2.0f, 30);
    for (std::size_t i = 0; i < layers.size(); ++i)
        ml::block_forward<c>(layers[i], next2, 3, cache2, i, rope);

    // The norm the configuration applies: RMS here, layer norm for GPT-2
    std::array<float, 8> normed{};
    ml::block_norm<c>(std::span<const float>(next), {}, {}, normed);

    std::printf("  prompt token 2 after both layers: [%.4f %.4f %.4f ...]\n", static_cast<double>(prompt[16]), static_cast<double>(prompt[17]),
                static_cast<double>(prompt[18]));
    std::printf("  decoded token 3: [%.4f %.4f %.4f ...]\n", static_cast<double>(next[0]), static_cast<double>(next[1]), static_cast<double>(next[2]));
    double ms = 0;
    for (float v : normed)
        ms += double(v) * double(v);
    std::printf("  prefill and step-by-step decode %s; RMS of the normed row %.4f\n", prompt == stepped ? "agree to the bit" : "DIFFER",
                std::sqrt(ms / 8));
    ok &= prompt == stepped && next == next2 && keys == k2 && std::abs(std::sqrt(ms / 8) - 1.0) < 1e-4;
    for (float v : next)
        ok &= std::isfinite(v);
    //! [block_backward]
    // Training one block: 1/2 |block(x) - target|^2 over 3 tokens, by plain gradient descent on every weight
    layer t(1);   // a copy of layer 0's weights, to train
    const std::vector<float> x = pattern(3 * 8, 2.0f, 20), target = pattern(3 * 8, 1.0f, 40);
    std::vector<float> x0(x.size()), h1(x.size()), x1(x.size()), h2(x.size()), sq(3 * 8), sk(3 * 4), sv(3 * 4), sy(3 * 8),
        slse(3 * 2), sup(3 * 16), sgate(3 * 16), scratch(x.size());
    const ml::block_activations<float> act{x0, h1, x1, h2, {sq, sk, sv, {}, {}, sy, slse}, sup, sgate};
    std::vector<float> losses;
    for (int step = 0; step < 5; ++step) {
        std::vector<float> out = x;
        ml::block_forward_sequence<c>(t.weights(), out, rope, act);
        std::vector<float> dout(out.size());
        float loss = 0.0f;
        for (std::size_t i = 0; i < out.size(); ++i) {
            dout[i] = out[i] - target[i];
            loss += 0.5f * dout[i] * dout[i];
        }
        losses.push_back(loss);
        // Gradients have the weights' shapes; the backward pass adds into them
        std::vector<float> gq(t.q.size()), gkv(t.kv.size()), gv(t.kv.size()), go(t.o.size()), gup(t.up.size()),
            ggate(t.gate.size()), gdown(t.down.size()), dq(sq.size()), dk(sk.size()), dv(sv.size());
        const ml::block_gradients<float> grads{{}, {}, {}, {}, {dq, dk, dv, gq, gkv, gv, go}, {gup, ggate, gdown}, scratch};
        ml::block_backward<c>(t.weights(), rope, act, dout, grads);   // dout becomes the gradient in x
        // K and V share one matrix in this layer, so its gradient is the sum of both uses
        auto descend = [](std::vector<float>& w, const std::vector<float>& g) {
            for (std::size_t i = 0; i < w.size(); ++i)
                w[i] -= 0.02f * g[i];
        };
        for (std::size_t i = 0; i < gkv.size(); ++i)
            gkv[i] += gv[i];
        descend(t.q, gq), descend(t.kv, gkv), descend(t.o, go), descend(t.up, gup), descend(t.gate, ggate), descend(t.down, gdown);
    }
    //! [block_backward]
    std::printf("  training one block, loss by step:");
    for (float l : losses)
        std::printf(" %.4f", static_cast<double>(l));
    std::printf("\n");
    for (std::size_t i = 1; i < losses.size(); ++i)
        ok &= losses[i] < losses[i - 1];

    // The sequence forward is block_prefill, to the bit
    std::vector<float> seq = pattern(3 * 8, 2.0f, 20), pre = seq;
    std::vector<float> k3(ml::kv_cache_view<c>::required_size(1, 3)), v3(k3.size());
    ml::block_forward_sequence<c>(l0.weights(), seq, rope, act);
    ml::block_prefill<c>(l0.weights(), pre, 0, ml::kv_cache_view<c>(k3, v3, 1, 3), 0, rope);
    ok &= seq == pre;

    //! [block_batched]
    // The same block with every projection a matrix product through a linalg backend (a BLAS or a GPU backend from
    // another library the same way). It also saves the MLP's hidden rows, and its backward has two more working
    // buffers: the MLP's hidden gradients (twice d_ff for SwiGLU) and the heads' output gradients
    std::vector<float> hidden(3 * 16), mlp_scratch(2 * 3 * 16), dy(3 * 8);
    ml::block_activations<float> bact = act;
    bact.mlp_hidden = hidden;
    auto one_step = [&](bool batched, std::vector<float>& out, std::vector<float>& dout, std::vector<float>& gq) {
        std::vector<float> gkv(t.kv.size()), gv(t.kv.size()), go(t.o.size()), gup(t.up.size()), ggate(t.gate.size()),
            gdown(t.down.size()), dq(sq.size()), dk(sk.size()), dv(sv.size());
        const ml::block_gradients<float> grads{
            {}, {}, {}, {}, {dq, dk, dv, gq, gkv, gv, go, {}, {}, {}, {}, {}, {}, dy}, {gup, ggate, gdown, {}, {}, {}, mlp_scratch}, scratch};
        if (batched) {
            ml::block_forward_sequence<c>(snicholls::linalg::reference, t.weights(), out, rope, bact);
            ml::block_backward<c>(snicholls::linalg::reference, t.weights(), rope, bact, dout, grads);
        } else {
            ml::block_forward_sequence<c>(t.weights(), out, rope, act);
            ml::block_backward<c>(t.weights(), rope, act, dout, grads);
        }
    };
    std::vector<float> out_p = x, out_b = x, dout_p = target, dout_b = target, gq_p(t.q.size()), gq_b(t.q.size());
    one_step(false, out_p, dout_p, gq_p);
    one_step(true, out_b, dout_b, gq_b);
    //! [block_batched]
    float worst = 0.0f;
    for (std::size_t i = 0; i < out_p.size(); ++i)
        worst = std::max({worst, std::abs(out_p[i] - out_b[i]), std::abs(dout_p[i] - dout_b[i])});
    for (std::size_t i = 0; i < gq_p.size(); ++i)
        worst = std::max(worst, std::abs(gq_p[i] - gq_b[i]));
    std::printf("  batched through the reference backend: within %.2g of the per-token path\n", static_cast<double>(worst));
    ok &= worst < 1e-5f;

    // The batched block through my_device::backend: its own norm step runs, twice, and the result is the reference's
    std::vector<float> out_mine = x;
    ml::block_forward_sequence<c>(my_device::backend{}, t.weights(), out_mine, rope, bact);
    std::vector<float> out_ref = x;
    ml::block_forward_sequence<c>(snicholls::linalg::reference, t.weights(), out_ref, rope, bact);
    std::printf("  a backend of its own: its norm step called %d times, the reference's result %s\n", my_device::backend::norms,
                out_mine == out_ref ? "to the bit" : "DIFFERS");
    ok &= my_device::backend::norms == 2 && out_mine == out_ref;

    //! [block_sequences]
    // A batch: two sequences of 3 tokens end to end, one call per layer instead of one per sequence. The activations and
    // the working spaces cover all six rows; attention stays within each sequence, positions restarting at 0
    constexpr std::size_t batch = 2, rows = batch * 3;
    std::vector<float> xb = x;
    xb.insert(xb.end(), target.begin(), target.end());   // sequence 0 is x, sequence 1 is target
    std::vector<float> bx0(rows * 8), bh1(rows * 8), bx1(rows * 8), bh2(rows * 8), bq(rows * 8), bk(rows * 4), bv(rows * 4), by(rows * 8),
        blse(rows * 2), bup(rows * 16), bgate(rows * 16), bhidden(rows * 16);
    const ml::block_activations<float> batch_act{bx0, bh1, bx1, bh2, {bq, bk, bv, {}, {}, by, blse}, bup, bgate, bhidden};
    ml::block_forward_sequence<c>(snicholls::linalg::reference, t.weights(), xb, rope, batch_act, batch);
    // ... and its backward: ml::block_backward<c>(backend, weights, rope, batch_act, dxb, grads, batch)
    //! [block_sequences]
    std::vector<float> s0 = x, s1 = target;   // each sequence on its own
    ml::block_forward_sequence<c>(snicholls::linalg::reference, t.weights(), s0, rope, bact);
    ml::block_forward_sequence<c>(snicholls::linalg::reference, t.weights(), s1, rope, bact);
    const bool same_bits = std::equal(s0.begin(), s0.end(), xb.begin()) && std::equal(s1.begin(), s1.end(), xb.begin() + 24);
    std::printf("  a batch of two sequences in one call: each sequence's own call's bits %s\n", same_bits ? "yes" : "NO");
    ok &= same_bits;

    std::printf(ok ? "transformer_block demo: ok\n" : "transformer_block demo: FAILED\n");
    return ok ? 0 : 1;
}
