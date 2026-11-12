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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
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

    for (std::size_t pos = 0; pos < 3; ++pos)
        std::printf("  position %zu: [%.3f %.3f %.3f %.3f | %.3f ...]\n", pos, static_cast<double>(out[pos][0]), static_cast<double>(out[pos][1]),
                    static_cast<double>(out[pos][2]), static_cast<double>(out[pos][3]), static_cast<double>(out[pos][4]));
    std::printf("  one head: [%.3f %.3f %.3f %.3f], logsumexp %.3f\n", static_cast<double>(head[0]), static_cast<double>(head[1]),
                static_cast<double>(head[2]), static_cast<double>(head[3]), static_cast<double>(lse));
    // Position 0 attends to itself alone: each head's output is the token's value, the first 4 channels
    ok &= out[0][0] == 1.0f && out[0][1] == 0.0f && out[0][4] == 1.0f && out[0][5] == 0.0f;
    // Both query heads read the one key-value head, so their outputs are averages of the same values: weights sum to 1
    ok &= std::abs(out[2][0] - 1.0f) < 1e-6f && out[2][1] > 0.0f && out[2][1] < 2.0f && std::isfinite(lse);
    std::printf(ok ? "attention_layer demo: ok\n" : "attention_layer demo: FAILED\n");
    return ok ? 0 : 1;
}
