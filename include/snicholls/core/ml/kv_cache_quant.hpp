//
//  kv_cache_quant.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` quantised KV caches: keys and values stored as Q8_0, Q4_0 or MXFP4 rows, attention that reads them as ggml's flash attention does, and the K-shift of quantised keys.
/// \ingroup core
///
/// **Overview.** A cache of `float` keys and values is the largest buffer of long-context inference. llama.cpp stores
/// them quantised (`--cache-type-k q8_0 --cache-type-v q4_0`): each head's row of `head_dim` values is
/// `head_dim / 32` blocks of `ml/quantize.hpp`. Its CPU flash attention
/// (`ggml_compute_forward_flash_attn_ext_f16_one_chunk`, `ggml/src/ggml-cpu/ops.cpp`, llama.cpp 39039aa) quantises the
/// query once to the key type's dot-product partner (Q8_0 for every block type here), scores each key with the
/// block dot product (`kq_vec_dot`), and dequantises each value row (`v_to_float`) into the running sum. This header
/// does the same:
///
/// - `quantized_kv_cache_view<C, KBlock, VBlock>`: views over the caller's blocks, laid out as `kv_cache_view`'s
///   `[layer][kv_head][slot][block]`, with `append(layer, pos, k, v)` quantising each head (slot `pos % capacity`, a
///   ring under a sliding window), `key_blocks` and `value_blocks` by slot; `quantized_kv_cache<C, Layers, Capacity, ...>`
///   owns fixed-size arrays.
/// - `attend_quantized(cache, layer, kv_head, q, keys, out)`: one query head over the keys `keys` (a `key_range` of
///   positions, which may wrap the ring), with the configuration's scale and soft cap, in `attend`'s recurrence.
/// - `kv_shift_quantized_keys<C>(cache, layers, cells)`: llama.cpp's K-shift on a quantised cache: each moved key is
///   dequantised, rotated by its shift and quantised again (as ggml's graph does with a cast around `ggml_rope`).
///
/// \snippet ml/kv_cache_quant_demo.cpp kv_cache_quant
///
/// **Relationships.** `ml/quantize.hpp` (the blocks, `quantize`, `dequantize`, `dot`); `ml/kv_cache.hpp` (the layout
/// and `key_range`); `ml/kv_cells.hpp` (positions, sequences and shifts). Group: \ref core.
///
/// **Implementations.** One, for any `quant_block` keys and values.
///
/// **Guarantees.** `constexpr`, no allocation: the quantised query and one dequantised value row are fixed-size
/// arrays on the stack. The scores and the recurrence are those of a loop that quantises the query, takes `dot` per key
/// and adds each dequantised value to an `attention_accumulator`, to the bit (tested). Bytes per cached value: 1.0625
/// (Q8_0), 0.5625 (Q4_0), 0.53125 (MXFP4), against 4 for `float`.
///
/// **Errors.** Preconditions, asserted: `head_dim` a multiple of 32 (a compile error otherwise); the block spans
/// `required_blocks`' size; the rows `kv_size` floats; the range no longer than the capacity.
///
/// **Performance.** Not benchmarked here: the block dot products are `ml/quantize.hpp`'s, benchmarked there against
/// ggml's scalar loops.
///
/// **Tests and demos.** `tests/ml/kv_cache_quant_test.cpp`, `demos/ml/kv_cache_quant_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<limits>`, `<span>`, `ml/activations.hpp`,
/// `ml/attention.hpp`, `ml/kv_cache.hpp`, `ml/kv_cells.hpp`, `ml/quantize.hpp`, `ml/rope.hpp`,
/// `ml/transformer_config.hpp`.
///
/// **Known issues.** ggml accumulates the values in `float16` (`VKQ16`) when V is `float16`, and in `float` otherwise;
/// here always in `float`. The block formats of the quantisation track (K-quants, IQ) plug in once they model
/// `quant_block`. A quantised K-shift requantises, so each shift adds a rounding of the key (ggml's does too).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_kv_cache_quant_hpp
#define snicholls_core_ml_kv_cache_quant_hpp

#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/kv_cells.hpp>
#include <snicholls/core/ml/quantize.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <span>

namespace snicholls::ml
{

/// \brief A KV cache of quantised rows over the caller's blocks: `[layer][kv_head][slot][block]`.
/// \ingroup core
/// \tparam C The configuration; `head_dim` a multiple of 32.
/// \tparam KBlock The keys' block (`block_q8_0`, `block_q4_0`, `block_mxfp4`).
/// \tparam VBlock The values' block.
template <transformer_config C, quant_block KBlock = block_q8_0, quant_block VBlock = block_q8_0>
class quantized_kv_cache_view
{
    static_assert(valid(C), "snicholls::ml::quantized_kv_cache_view: the transformer_config is not valid");
    static_assert(C.head_dim % quant_block_size == 0, "snicholls::ml::quantized_kv_cache_view: head_dim must be a multiple of 32");

public:
    /// \brief The blocks of one head's row.
    static constexpr std::size_t blocks_per_row = C.head_dim / quant_block_size;

    /// \brief The blocks each of the key and value storage needs.
    /// \param layers The layers.
    /// \param capacity The slots.
    /// \return `layers · n_kv_heads · capacity · head_dim / 32`.
    [[nodiscard]] static constexpr std::size_t required_blocks(std::size_t layers, std::size_t capacity) noexcept
    {
        return layers * C.n_kv_heads * capacity * blocks_per_row;
    }

    /// \brief An empty view.
    constexpr quantized_kv_cache_view() noexcept = default;

    /// \brief A view over the caller's blocks.
    /// \param keys The keys' blocks, `required_blocks(layers, capacity)`.
    /// \param values The values' blocks, as many.
    /// \param layers The layers.
    /// \param capacity The slots (at least the window under a sliding window).
    constexpr quantized_kv_cache_view(std::span<KBlock> keys, std::span<VBlock> values, std::size_t layers, std::size_t capacity) noexcept
        : keys_(keys), values_(values), layers_(layers), capacity_(capacity)
    {
        assert(keys.size() == required_blocks(layers, capacity) && values.size() == keys.size());
        assert(C.window == 0 || capacity == 0 || capacity >= C.window);
    }

    /// \brief Quantises and stores one position's keys and values, every key-value head, at slot `pos % capacity`.
    /// \param layer The layer.
    /// \param pos The position (or the slot, below the capacity).
    /// \param k The keys, `kv_size` floats.
    /// \param v The values, `kv_size` floats.
    template <row K, row V>
        requires std::same_as<row_value_t<K>, float> && std::same_as<row_value_t<V>, float>
    constexpr void append(std::size_t layer, std::size_t pos, const K& k, const V& v) const
    {
        const auto ks = as_row(k);
        const auto vs = as_row(v);
        assert(layer < layers_ && capacity_ > 0 && ks.size() == C.kv_size() && vs.size() == C.kv_size());
        for (std::size_t h = 0; h < C.n_kv_heads; ++h) {
            const std::size_t at = offset(layer, h, pos % capacity_);
            quantize(ks.subspan(h * C.head_dim, C.head_dim), keys_.subspan(at, blocks_per_row));
            quantize(vs.subspan(h * C.head_dim, C.head_dim), values_.subspan(at, blocks_per_row));
        }
    }

    /// \brief One slot's key row.
    /// \param layer The layer.
    /// \param kv_head The key-value head.
    /// \param slot The slot.
    /// \return `head_dim / 32` blocks.
    [[nodiscard]] constexpr std::span<const KBlock> key_blocks(std::size_t layer, std::size_t kv_head, std::size_t slot) const noexcept
    {
        return std::span<const KBlock>(keys_).subspan(offset(layer, kv_head, slot), blocks_per_row);
    }

    /// \brief One slot's key row, to rewrite it (the K-shift).
    /// \param layer The layer.
    /// \param kv_head The key-value head.
    /// \param slot The slot.
    /// \return `head_dim / 32` blocks.
    [[nodiscard]] constexpr std::span<KBlock> mutable_key_blocks(std::size_t layer, std::size_t kv_head, std::size_t slot) const noexcept
    {
        return keys_.subspan(offset(layer, kv_head, slot), blocks_per_row);
    }

    /// \brief One slot's value row.
    /// \param layer The layer.
    /// \param kv_head The key-value head.
    /// \param slot The slot.
    /// \return `head_dim / 32` blocks.
    [[nodiscard]] constexpr std::span<const VBlock> value_blocks(std::size_t layer, std::size_t kv_head, std::size_t slot) const noexcept
    {
        return std::span<const VBlock>(values_).subspan(offset(layer, kv_head, slot), blocks_per_row);
    }

    /// \brief The layers.
    /// \return Their number.
    [[nodiscard]] constexpr std::size_t layers() const noexcept { return layers_; }
    /// \brief The slots.
    /// \return Their number.
    [[nodiscard]] constexpr std::size_t capacity() const noexcept { return capacity_; }

private:
    constexpr std::size_t offset(std::size_t layer, std::size_t h, std::size_t slot) const noexcept
    {
        return ((layer * C.n_kv_heads + h) * capacity_ + slot) * blocks_per_row;
    }

    std::span<KBlock> keys_{};
    std::span<VBlock> values_{};
    std::size_t layers_ = 0;
    std::size_t capacity_ = 0;
};

/// \brief A quantised KV cache owning fixed-size arrays (an FPGA's or a test's).
/// \ingroup core
/// \tparam C The configuration.
/// \tparam Layers The layers.
/// \tparam Capacity The slots.
/// \tparam KBlock The keys' block.
/// \tparam VBlock The values' block.
template <transformer_config C, std::size_t Layers, std::size_t Capacity, quant_block KBlock = block_q8_0, quant_block VBlock = block_q8_0>
class quantized_kv_cache
{
    static constexpr std::size_t size = quantized_kv_cache_view<C, KBlock, VBlock>::required_blocks(Layers, Capacity);

public:
    /// \brief A view of the arrays.
    /// \return The view.
    [[nodiscard]] constexpr quantized_kv_cache_view<C, KBlock, VBlock> view() noexcept { return {keys_, values_, Layers, Capacity}; }

private:
    std::array<KBlock, size> keys_{};
    std::array<VBlock, size> values_{};
};

/// \brief One query head over quantised keys and values at the positions `keys`: the query quantised to Q8_0 once,
/// each score `scale · dot(key, q)` (soft-capped if the configuration says), each value dequantised into `attend`'s
/// recurrence, as ggml's CPU flash attention.
/// \tparam C The configuration.
/// \tparam KBlock The keys' block.
/// \tparam VBlock The values' block.
/// \param cache The cache.
/// \param layer The layer.
/// \param kv_head The key-value head.
/// \param q The query head, `head_dim` floats.
/// \param keys The positions attended to (slots `p % capacity`), no more than the capacity.
/// \param out Receives the head's output, `head_dim` floats.
/// \return The log-sum-exp of the scores; -infinity for no keys.
template <transformer_config C, quant_block KBlock, quant_block VBlock, row Q, mutable_row Out>
    requires std::same_as<row_value_t<Q>, float> && std::same_as<row_value_t<Out>, float>
constexpr float attend_quantized(const quantized_kv_cache_view<C, KBlock, VBlock>& cache, std::size_t layer, std::size_t kv_head, const Q& q,
                                 key_range keys, Out&& out)
{
    constexpr std::size_t d = C.head_dim, blocks = d / quant_block_size;
    const auto qs = as_row(q);
    const auto y = as_mutable_row(out);
    assert(qs.size() == d && y.size() == d && keys.size() <= cache.capacity());
    std::array<block_q8_0, blocks> qq{};
    quantize(qs, std::span<block_q8_0, blocks>(qq));
    const float scale = 1.0f / math::sqrt(float(d));
    std::array<float, d> v{};
    std::fill(y.begin(), y.end(), 0.0f);
    float max = -std::numeric_limits<float>::infinity(), sum = 0.0f;
    for (std::size_t p = keys.first; p < keys.last; ++p) {
        const std::size_t slot = p % cache.capacity();
        float s = scale * dot(cache.key_blocks(layer, kv_head, slot), std::span<const block_q8_0>(qq));
        if constexpr (C.attention_softcap > 0)
            s = softcap(s, attention_softcap_v<C, float>);
        dequantize(cache.value_blocks(layer, kv_head, slot), v);
        detail::attention::fold(max, sum, y, s, std::span<const float>(v));
    }
    for (float& x : y)
        x = sum == 0.0f ? 0.0f : x / sum;
    return sum == 0.0f ? -std::numeric_limits<float>::infinity() : max + math::log(sum);
}

/// \brief The K-shift of a quantised cache: each key whose cell moved is dequantised, rotated by the shift (the
/// configuration's base, layout and rescaling) and quantised again, in every layer and key-value head; the shifts are
/// then cleared.
/// \tparam C A configuration with RoPE and standard attention.
/// \tparam KBlock The keys' block.
/// \tparam VBlock The values' block.
/// \tparam Capacity The cells' capacity, the cache's.
/// \param cache The cache.
/// \param cells The cells, their shifts cleared.
template <transformer_config C, quant_block KBlock, quant_block VBlock, std::size_t Capacity>
constexpr void kv_shift_quantized_keys(const quantized_kv_cache_view<C, KBlock, VBlock>& cache, kv_cells<Capacity>& cells)
{
    static_assert(C.rope != rope_kind::none && !C.latent(),
                  "snicholls::ml::kv_shift_quantized_keys: the configuration rotates no keys, or its keys are latent");
    assert(cache.capacity() == Capacity);
    constexpr std::size_t d = C.head_dim, h = d / 2;
    if (!cells.has_shift())
        return;
    std::array<float, d> row{};
    for (std::size_t i = 0; i < Capacity; ++i) {
        const kv_pos delta = cells.shift(i);
        if (cells.is_empty(i) || delta == 0)
            continue;
        std::array<float, h> c{}, s{};
        rope_angles(static_cast<std::size_t>(delta < 0 ? -delta : delta), c, s, rope_base_v<C, float>, rope_scaling_v<C, float>);
        for (std::size_t layer = 0; layer < cache.layers(); ++layer)
            for (std::size_t kv = 0; kv < C.n_kv_heads; ++kv) {
                const auto blocks = cache.mutable_key_blocks(layer, kv, i);
                dequantize(std::span<const KBlock>(blocks), row);
                if (delta > 0)
                    rope(row, c, s, C.layout());
                else
                    rope_inverse(row, c, s, C.layout());
                quantize(row, blocks);
            }
    }
    cells.reset_shift();
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_kv_cache_quant_hpp
