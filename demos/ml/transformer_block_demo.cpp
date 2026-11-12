//
//  transformer_block_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/transformer_block.hpp>: a two-layer stack of small blocks (RMS norm, grouped-query
//  attention with RoPE, SwiGLU) prefilling a prompt and then decoding one more token, and the norm each configuration
//  applies. Prefill and decode are one code path, so prefilling the prompt and decoding it token by token agree to the
//  bit. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/transformer_block.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

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
    std::printf(ok ? "transformer_block demo: ok\n" : "transformer_block demo: FAILED\n");
    return ok ? 0 : 1;
}
