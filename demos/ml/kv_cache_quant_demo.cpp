//
//  kv_cache_quant_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/kv_cache_quant.hpp>: a cache with Q8_0 keys and Q4_0 values (llama.cpp's
//  --cache-type-k q8_0 --cache-type-v q4_0), one query head attending to it next to the same attention over float
//  keys and values, and the K-shift of the quantised keys. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/kv_cache_quant.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core kv_cache_quant demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [kv_cache_quant]
    // One layer, one key-value head of 64 channels, 16 slots: keys as Q8_0, values as Q4_0 (owning arrays)
    constexpr ml::transformer_config c{.d_model = 64, .n_heads = 1, .n_kv_heads = 1, .head_dim = 64, .d_ff = 64, .rope = ml::rope_kind::half};
    ml::quantized_kv_cache<c, 1, 16, ml::block_q8_0, ml::block_q4_0> cache;
    const auto view = cache.view();
    std::vector<float> fk, fv;   // the same rows in float, to compare
    for (std::size_t p = 0; p < 16; ++p) {
        std::array<float, 64> k{}, v{};
        for (std::size_t i = 0; i < 64; ++i) {
            k[i] = std::sin(0.1f * float(i) + float(p));
            v[i] = std::cos(0.05f * float(i * p));
        }
        view.append(0, p, k, v);   // quantised on the way in
        fk.insert(fk.end(), k.begin(), k.end());
        fv.insert(fv.end(), v.begin(), v.end());
    }
    std::array<float, 64> q{}, out{}, exact{};
    for (std::size_t i = 0; i < 64; ++i)
        q[i] = std::sin(0.3f * float(i));
    ml::attend_quantized(view, 0, 0, q, ml::key_range{0, 16}, out);   // q to Q8_0, block dot products, values dequantised
    ml::attend(q, fk, fv, exact, 1.0f / 8.0f);

    // The K-shift on quantised keys: positions 8 to 15 move back by 8 after the first 8 are dropped
    ml::kv_cells<16> cells;
    for (std::size_t p = 0; p < 16; ++p)
        cells.place(p, ml::kv_pos(p), 0);
    cells.seq_rm(0, 0, 8);
    cells.seq_add(0, 8, -1, -8);
    ml::kv_shift_quantized_keys(view, cells);
    //! [kv_cache_quant]

    float worst = 0.0f;
    for (std::size_t i = 0; i < 64; ++i)
        worst = std::fmax(worst, std::abs(out[i] - exact[i]));
    std::printf("  64 channels x 16 keys: %zu bytes of blocks against %zu of floats; largest difference from float attention %.4f\n",
                16 * (2 * sizeof(ml::block_q8_0) + 2 * sizeof(ml::block_q4_0)), fk.size() * 2 * sizeof(float), static_cast<double>(worst));
    ok &= worst < 0.05f;
    std::printf("  after the shift: sequence 0 spans positions %d to %d, no shift pending: %s\n", cells.seq_pos_min(0), cells.seq_pos_max(0),
                cells.has_shift() ? "no" : "yes");
    ok &= cells.seq_pos_min(0) == 0 && cells.seq_pos_max(0) == 7 && !cells.has_shift();
    ok &= view.key_blocks(0, 0, 3).size() == 2 && view.value_blocks(0, 0, 3).size() == 2 && view.layers() == 1 && view.capacity() == 16;

    std::printf(ok ? "kv_cache_quant demo: ok\n" : "kv_cache_quant demo: FAILED\n");
    return ok ? 0 : 1;
}
