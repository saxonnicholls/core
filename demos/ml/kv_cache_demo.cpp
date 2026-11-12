//
//  kv_cache_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/kv_cache.hpp>: a cache over the caller's memory, a sliding window as a ring of three
//  slots whose visible keys come back as two runs in position order, and an owning cache. Exits non-zero if a result
//  is not what it documents.
//

#include <snicholls/core/ml/kv_cache.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core kv_cache demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [kv_cache]
    // One key-value head of 2 channels, a sliding window of 3: a ring of 3 slots is enough
    constexpr ml::transformer_config c{.d_model = 2, .n_heads = 1, .n_kv_heads = 1, .head_dim = 2, .d_ff = 2, .window = 3};
    std::vector<float> keys(ml::kv_cache_view<c>::required_size(1, 3)), values(keys.size());
    const ml::kv_cache_view<c> cache(keys, values, 1, 3);
    for (std::size_t pos = 0; pos < 5; ++pos) {
        const std::array<float, 2> k{float(pos), float(pos)}, v{-float(pos), -float(pos)};
        cache.append(0, pos, k, v);   // position 3 overwrites slot 0, position 4 slot 1
    }
    // Position 4 sees positions 2, 3 and 4: slot 2, then slots 0 and 1
    const ml::kv_runs<float> runs = cache.visible(0, 0, ml::attention_window(4, c.window));

    // The same cache owning its storage, for a model small enough to hold
    static ml::kv_cache<c, float, 1, 3> owned;
    owned.view().append(0, 0, std::array<float, 2>{7, 7}, std::array<float, 2>{8, 8});
    //! [kv_cache]

    std::printf("  run 0: %zu position(s) from %.0f; run 1: %zu from %.0f; %zu layer, %zu slots\n", runs.keys[0].size() / 2,
                static_cast<double>(runs.keys[0][0]), runs.keys[1].size() / 2, static_cast<double>(runs.keys[1][0]), cache.layers(), cache.capacity());
    ok &= runs.keys[0].size() == 2 && runs.keys[0][0] == 2.0f && runs.keys[1].size() == 4 && runs.keys[1][0] == 3.0f && runs.keys[1][2] == 4.0f;
    ok &= runs.values[1][3] == -4.0f && owned.view().visible(0, 0, {0, 1}).values[0][0] == 8.0f;
    std::printf(ok ? "kv_cache demo: ok\n" : "kv_cache demo: FAILED\n");
    return ok ? 0 : 1;
}
