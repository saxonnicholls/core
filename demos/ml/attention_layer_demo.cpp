//
//  attention_layer_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/attention_layer.hpp>: grouped-query attention with RoPE and a KV cache, decoding three
//  tokens one at a time, and one head on its own. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/attention_layer.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core attention_layer demo\n");
    namespace ml = snicholls::ml;
    using W = ml::dense_weights<float>;
    bool ok = true;

    //! [attention_layer]
    // Two query heads share one key-value head of 4 channels; RoPE in the half layout
    constexpr ml::transformer_config c{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 8, .rope = ml::rope_kind::half};
    std::vector<float> wq(8 * 8), wkv(4 * 8), wo(8 * 8);
    for (std::size_t i = 0; i < 8; ++i)
        wq[i * 8 + i] = wo[i * 8 + i] = 1.0f;   // identity projections, so the arithmetic is visible
    for (std::size_t i = 0; i < 4; ++i)
        wkv[i * 8 + i] = 1.0f;
    const ml::attention_weights<W> w{W{wq, 8, 8}, W{wkv, 4, 8}, W{wkv, 4, 8}, W{wo, 8, 8}};

    static const ml::rope_table<float, 16, 4> rope{};   // cosines and sines for positions 0 to 15
    std::vector<float> keys(ml::kv_cache_view<c>::required_size(1, 16)), values(keys.size());
    const ml::kv_cache_view<c> cache(keys, values, 1, 16);

    std::array<std::array<float, 8>, 3> out{};
    for (std::size_t pos = 0; pos < 3; ++pos) {
        const std::array<float, 8> x{1, float(pos), 0, 0, 0.5f, 0.5f, 0.5f, 0.5f};
        ml::attention_forward<c>(w, x, pos, cache, 0, rope, out[pos]);   // appends this token, attends to all so far
    }

    // One head alone over what the cache holds: the unit to spread over threads
    std::array<float, 4> head{};
    const std::array<float, 4> q{1, 0, 0, 0};
    const float lse = ml::attention_head<c, float>(q, cache.visible(0, 0, ml::attention_window(2)), head);
    //! [attention_layer]

    //! [attention_backward]
    // Training: the same three tokens as one sequence, saving what the backward pass reads
    constexpr std::size_t n = 3;
    std::vector<float> xs;
    for (std::size_t pos = 0; pos < n; ++pos)
        for (float e : {1.0f, float(pos), 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f})
            xs.push_back(e);
    std::vector<float> sq(n * c.q_size()), sk(n * c.kv_size()), sv(sk.size()), sy(sq.size()), slse(n * c.n_heads), seq(n * 8);
    const ml::attention_activations<float> act{sq, sk, sv, {}, {}, sy, slse};   // no QK norm: no raw projections
    ml::attention_forward_sequence<c>(w, xs, rope, act, seq);

    // A gradient in the last token's first output channel, back to every weight and every input
    std::vector<float> dout(n * 8);
    dout[2 * 8 + 0] = 1.0f;
    std::vector<float> dq(sq.size()), dk(sk.size()), dv(sv.size()), dwq(wq.size()), dwk(wkv.size()), dwv(wkv.size()),
        dwo(wo.size()), dx(xs.size());
    ml::attention_backward<c>(w, xs, rope, act, dout, dx, ml::attention_gradients<float>{dq, dk, dv, dwq, dwk, dwv, dwo});

    // One head's backward, soft cap or not, over the keys it saw
    std::array<float, 4> dhead_q{};
    std::vector<float> dhead_k(sk.size()), dhead_v(sv.size());
    ml::attention_head_backward<c, float>(std::span<const float>(sq).subspan(2 * 8, 4), sk, sv,
                                          std::span<const float>(sy).subspan(2 * 8, 4), std::array<float, 4>{1, 0, 0, 0},
                                          slse[2 * 2], dhead_q, dhead_k, dhead_v);
    //! [attention_backward]

    for (std::size_t pos = 0; pos < 3; ++pos)
        std::printf("  position %zu: [%.3f %.3f %.3f %.3f | %.3f ...]\n", pos, static_cast<double>(out[pos][0]), static_cast<double>(out[pos][1]),
                    static_cast<double>(out[pos][2]), static_cast<double>(out[pos][3]), static_cast<double>(out[pos][4]));
    std::printf("  one head: [%.3f %.3f %.3f %.3f], logsumexp %.3f\n", static_cast<double>(head[0]), static_cast<double>(head[1]),
                static_cast<double>(head[2]), static_cast<double>(head[3]), static_cast<double>(lse));
    // Position 0 attends to itself alone: each head's output is the token's value, the first 4 channels
    ok &= out[0][0] == 1.0f && out[0][1] == 0.0f && out[0][4] == 1.0f && out[0][5] == 0.0f;
    // Both query heads read the one key-value head, so their outputs are averages of the same values: weights sum to 1
    ok &= std::abs(out[2][0] - 1.0f) < 1e-6f && out[2][1] > 0.0f && out[2][1] < 2.0f && std::isfinite(lse);
    // The sequence gives the decoded tokens' bits; the gradient reaches only what token 2's channel 0 depends on
    bool same = true;
    for (std::size_t pos = 0; pos < n; ++pos)
        for (std::size_t i = 0; i < 8; ++i)
            same &= seq[pos * 8 + i] == out[pos][i];
    std::printf("  sequence = decode: %s; d Wo[0][0] %.3f (= y[2][0]); dx[2][0] %.3f\n", same ? "yes" : "no",
                static_cast<double>(dwo[0]), static_cast<double>(dx[2 * 8 + 0]));
    ok &= same && dwo[0] == sy[2 * 8 + 0] && dwo[1 * 8 + 0] == 0.0f;   // only output row 0 had a gradient
    // One head with a gradient of 1 in its output channel 0: the values' gradients in channel 0 are its weights
    float vsum = 0.0f;
    for (std::size_t j = 0; j < n; ++j)
        vsum += dhead_v[j * 4 + 0];
    ok &= std::abs(vsum - 1.0f) < 1e-6f;   // the softmax weights over the three keys sum to 1

    //! [attention_batched]
    // The same sequence with its projections as matrix products through a linalg backend (a BLAS or a GPU backend
    // from another library the same way); the backward needs one more working buffer, the heads' output gradients
    std::vector<float> bq(sq.size()), bk(sk.size()), bv(sv.size()), by(sy.size()), blse(slse.size()), bout(seq.size());
    const ml::attention_activations<float> bact{bq, bk, bv, {}, {}, by, blse};
    ml::attention_forward_sequence<c>(snicholls::linalg::reference, w, xs, rope, bact, bout);
    std::vector<float> gq(sq.size()), gk(sk.size()), gv(sv.size()), gwq(wq.size()), gwk(wkv.size()), gwv(wkv.size()),
        gwo(wo.size()), gx(xs.size()), gy(sq.size());
    ml::attention_backward<c>(snicholls::linalg::reference, w, xs, rope, bact, dout, gx,
                              ml::attention_gradients<float>{gq, gk, gv, gwq, gwk, gwv, gwo, {}, {}, {}, {}, {}, {}, gy});
    //! [attention_batched]
    float worst = 0.0f;
    for (std::size_t i = 0; i < seq.size(); ++i)
        worst = std::max(worst, std::abs(bout[i] - seq[i]));
    for (std::size_t i = 0; i < dx.size(); ++i)
        worst = std::max(worst, std::abs(gx[i] - dx[i]));
    std::printf("  batched through the reference backend: within %.2g of the per-token path\n", static_cast<double>(worst));
    ok &= worst < 1e-6f;

    //! [attention_sequences]
    // A batch of two sequences in one call: the three tokens, then the same three again, end to end. Every projection
    // is one product over all six rows; attention stays within each sequence (positions restart at 0), and the
    // activations are each sequence's own side by side
    constexpr std::size_t batch = 2;
    std::vector<float> xs2(xs);
    xs2.insert(xs2.end(), xs.begin(), xs.end());
    std::vector<float> q2(batch * sq.size()), k2(batch * sk.size()), v2(batch * sv.size()), y2(batch * sy.size()),
        lse2(batch * slse.size()), out2(batch * seq.size());
    const ml::attention_activations<float> act2{q2, k2, v2, {}, {}, y2, lse2};
    ml::attention_forward_sequence<c>(snicholls::linalg::reference, w, xs2, rope, act2, out2, batch);
    //! [attention_sequences]
    // Each sequence's result is the one-sequence call's, to the bit
    const bool both = std::equal(bout.begin(), bout.end(), out2.begin()) && std::equal(bout.begin(), bout.end(), out2.begin() + std::ptrdiff_t(bout.size()))
                      && std::equal(bk.begin(), bk.end(), k2.begin() + std::ptrdiff_t(bk.size()));
    std::printf("  a batch of two sequences: each the one-sequence call's bits: %s\n", both ? "yes" : "no");
    ok &= both;

    std::printf(ok ? "attention_layer demo: ok\n" : "attention_layer demo: FAILED\n");
    return ok ? 0 : 1;
}
