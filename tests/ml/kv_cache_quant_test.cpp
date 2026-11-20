//
//  kv_cache_quant_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/kv_cache_quant.hpp>: stored rows are quantize's blocks, attention over Q8_0, Q4_0 and
//  MXFP4 caches against the loop ggml's flash attention runs (the query quantised to Q8_0, dot per key, each value
//  dequantised into an attention_accumulator) to the bit, against float attention within the quantisation's error, a
//  ring that wraps, the soft cap, the quantised K-shift, and the owning cache.
//

#include <snicholls/core/ml/kv_cache_quant.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr ml::transformer_config cfg{.d_model = 64, .n_heads = 2, .n_kv_heads = 2, .head_dim = 64, .d_ff = 64, .rope = ml::rope_kind::half};
constexpr ml::transformer_config capped{.d_model = 64, .n_heads = 2, .n_kv_heads = 2, .head_dim = 64, .d_ff = 64,
                                        .rope = ml::rope_kind::half, .attention_softcap = 2.0};

std::vector<float> row(std::size_t n, float phase)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::sin(0.13f * float(i) + phase) * (1.0f + 0.01f * float(i % 7));
    return v;
}

// ggml_compute_forward_flash_attn_ext_f16_one_chunk's order, written with core's pieces
template <ml::transformer_config C, typename KB, typename VB>
float reference(const ml::quantized_kv_cache_view<C, KB, VB>& cache, std::size_t layer, std::size_t kv, std::span<const float> q, ml::key_range r,
                std::span<float> out)
{
    std::array<ml::block_q8_0, 2> qq{};
    ml::quantize(q, std::span<ml::block_q8_0, 2>(qq));
    const float scale = 1.0f / std::sqrt(64.0f);
    ml::attention_accumulator<float, 64> acc{};
    for (std::size_t p = r.first; p < r.last; ++p) {
        const std::size_t slot = p % cache.capacity();
        float s = scale * ml::dot(cache.key_blocks(layer, kv, slot), std::span<const ml::block_q8_0>(qq));
        if constexpr (C.attention_softcap > 0)
            s = ml::softcap(s, ml::attention_softcap_v<C, float>);
        std::array<float, 64> v{};
        ml::dequantize(cache.value_blocks(layer, kv, slot), v);
        acc.add(s, v);
    }
    acc.result(out);
    return acc.log_sum_exp();
}

template <ml::transformer_config C, typename KB, typename VB>
void attention_is_ggmls(std::size_t capacity, ml::key_range r)
{
    using view_t = ml::quantized_kv_cache_view<C, KB, VB>;
    std::vector<KB> kb(view_t::required_blocks(2, capacity));
    std::vector<VB> vb(kb.size());
    const view_t cache(kb, vb, 2, capacity);
    std::vector<float> fk, fv;   // the float cache's head 1 of layer 1, for comparison
    for (std::size_t p = r.first; p < r.last; ++p)
        for (std::size_t l = 0; l < 2; ++l) {
            const auto k = row(128, float(p) + 0.3f * float(l)), v = row(128, 2.0f * float(p) - float(l));
            cache.append(l, p, k, v);
            if (l == 1) {
                fk.insert(fk.end(), k.begin() + 64, k.end());
                fv.insert(fv.end(), v.begin() + 64, v.end());
            }
        }
    const auto q = row(64, 0.7f);
    std::array<float, 64> got{}, want{}, exact{};
    const float lse = ml::attend_quantized(cache, 1, 1, q, r, got);
    EXPECT_EQ(lse, reference(cache, 1, 1, q, r, want));
    EXPECT_EQ(got, want);
    // Within the quantisation's error of the float attention
    if constexpr (C.attention_softcap == 0) {
        ml::attend(q, fk, fv, exact, 1.0f / std::sqrt(64.0f));
        constexpr float tol = std::is_same_v<VB, ml::block_q8_0> && std::is_same_v<KB, ml::block_q8_0> ? 0.02f : 0.15f;
        for (std::size_t i = 0; i < 64; ++i)
            EXPECT_NEAR(got[i], exact[i], tol) << i;
    }
}

TEST(MlKvCacheQuant, StoredRowsAreQuantizesBlocks)
{
    using view_t = ml::quantized_kv_cache_view<cfg, ml::block_q8_0, ml::block_q4_0>;
    std::vector<ml::block_q8_0> kb(view_t::required_blocks(1, 4));
    std::vector<ml::block_q4_0> vb(kb.size());
    const view_t cache(kb, vb, 1, 4);
    const auto k = row(128, 1.0f), v = row(128, 2.0f);
    cache.append(0, 6, k, v);   // slot 6 % 4 = 2
    std::array<ml::block_q8_0, 2> kq{};
    std::array<ml::block_q4_0, 2> vq{};
    ml::quantize(std::span(k).subspan(64, 64), std::span<ml::block_q8_0, 2>(kq));
    ml::quantize(std::span(v).subspan(64, 64), std::span<ml::block_q4_0, 2>(vq));
    const auto sk = cache.key_blocks(0, 1, 2);
    const auto sv = cache.value_blocks(0, 1, 2);
    for (std::size_t b = 0; b < 2; ++b) {
        EXPECT_EQ(sk[b].d.bits, kq[b].d.bits);
        EXPECT_EQ(sk[b].qs, kq[b].qs);
        EXPECT_EQ(sv[b].qs, vq[b].qs);
    }
    static_assert(view_t::blocks_per_row == 2);
}

TEST(MlKvCacheQuant, AttentionIsGgmlsFlashAttentionLoopToTheBit)
{
    attention_is_ggmls<cfg, ml::block_q8_0, ml::block_q8_0>(16, {0, 16});
    attention_is_ggmls<cfg, ml::block_q4_0, ml::block_q4_0>(16, {3, 11});
    attention_is_ggmls<cfg, ml::block_q8_0, ml::block_q4_0>(8, {5, 13});   // wraps the ring
    attention_is_ggmls<cfg, ml::block_mxfp4, ml::block_q8_0>(8, {0, 8});
    attention_is_ggmls<capped, ml::block_q8_0, ml::block_q8_0>(8, {0, 8});
}

TEST(MlKvCacheQuant, NoKeysGiveZerosAndMinusInfinity)
{
    ml::quantized_kv_cache<cfg, 1, 4> owned;
    const auto view = owned.view();
    std::array<float, 64> out{};
    out.fill(1.0f);
    const auto q = row(64, 0.1f);
    EXPECT_EQ(ml::attend_quantized(view, 0, 0, q, ml::key_range{2, 2}, out), -std::numeric_limits<float>::infinity());
    EXPECT_EQ(out, (std::array<float, 64>{}));
}

TEST(MlKvCacheQuant, TheQuantisedKShiftIsDequantiseRotateQuantise)
{
    using view_t = ml::quantized_kv_cache_view<cfg, ml::block_q8_0, ml::block_q8_0>;
    std::vector<ml::block_q8_0> kb(view_t::required_blocks(1, 4)), vb(kb.size());
    const view_t cache(kb, vb, 1, 4);
    ml::kv_cells<4> cells;
    for (std::size_t t = 0; t < 4; ++t) {
        const auto k = row(128, float(t));
        cache.append(0, t, k, k);
        cells.place(t, ml::kv_pos(10 + t), 0);
    }
    const auto before = std::vector<ml::block_q8_0>(kb);
    cells.seq_add(0, 12, -1, -5);   // slots 2 and 3 move back by 5
    ml::kv_shift_quantized_keys(cache, cells);
    EXPECT_FALSE(cells.has_shift());
    for (std::size_t t = 0; t < 4; ++t)
        for (std::size_t h = 0; h < 2; ++h) {
            std::array<float, 64> r{};
            ml::dequantize(std::span<const ml::block_q8_0>(before).subspan((h * 4 + t) * 2, 2), r);
            if (t >= 2) {
                std::array<float, 32> c{}, s{};
                ml::rope_angles(5, c, s);
                ml::rope_inverse(r, c, s);
            }
            std::array<ml::block_q8_0, 2> want{};
            ml::quantize(r, std::span<ml::block_q8_0, 2>(want));
            const auto got = cache.key_blocks(0, h, t);
            for (std::size_t b = 0; b < 2; ++b)
                EXPECT_EQ(got[b].qs, t >= 2 ? want[b].qs : before[(h * 4 + t) * 2 + b].qs) << t << " " << h;
        }
}

} // namespace
