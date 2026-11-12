//
//  transformer_block.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::block_forward`: one pre-norm transformer block, `x += attention(norm(x)); x += mlp(norm(x))`, for a token or a run of tokens.
/// \ingroup core
///
/// **Overview.** The layer every decoder-only transformer stacks. `block_forward` updates one token's residual
/// stream in place at its position, appending its keys and values to the cache; `block_prefill` does the same for
/// a run of consecutive tokens, which is the same thing token by token, so prefill and decode cannot disagree. The
/// normalisation (layer norm with weight and bias, or RMS norm with or without a weight), the attention and the MLP
/// are the configuration's. A whole model, the stack of blocks between an embedding and an LM head, belongs above
/// core (`INFERENCE.md` I9).
///
/// \snippet ml/transformer_block_demo.cpp transformer_block
///
/// **Relationships.** `ml/attention_layer.hpp`, `ml/mlp.hpp`, `ml/norm.hpp`, `ml/kv_cache.hpp`,
/// `ml/transformer_config.hpp`; the plans `INFERENCE.md` and `TRAINING.md`. Group: \ref core.
///
/// **Implementations.** One, configured at compile time.
///
/// **Guarantees.** `constexpr`, no allocation (the normalised row and each half's output are `std::array`s of
/// `d_model`), the cost of `attention_forward` plus `mlp_forward` per token.
///
/// **Errors.** Preconditions, asserted, as `attention_forward` and `mlp_forward`; the norm weights and biases are
/// empty or have `d_model`; a prefill's rows are `d_model` each.
///
/// **Performance.** One decode step of a GPT-2-small layer (768 wide, 12 heads, MLP of 3072) at position 256, against
/// the same layer written as loops over raw pointers with every operation in the same order, after a bit-for-bit
/// fairness check (`benchmarks/ml/transformer_block_benchmark.cpp`; Release, Apple clang 17, Xeon W-3245, macOS 26,
/// 2026-11-12, medians of five, on a shared machine with load average 45 to 120):
///
/// | Weights | Core | By hand |
/// |---|---|---|
/// | float | 8.45 ms | 8.31 ms |
/// | Q8_0 | 3.07 ms | 3.00 ms |
///
/// Level within the machine's noise. A decode step reads every weight once, so it is bound by memory: Q8_0's quarter
/// of the bytes makes it 2.8x faster than float.
///
/// **Tests and demos.** `tests/ml/transformer_block_test.cpp`, `demos/ml/transformer_block_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `ml/attention_layer.hpp`, `ml/mlp.hpp`,
/// `ml/norm.hpp`.
///
/// **Known issues.** Post-norm and sandwich-norm blocks (Gemma's extra norms) are not configurable.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_transformer_block_hpp
#define snicholls_core_ml_transformer_block_hpp

#include <snicholls/core/ml/attention_layer.hpp>
#include <snicholls/core/ml/mlp.hpp>
#include <snicholls/core/ml/norm.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief The weights of a block: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix`.
/// \tparam T The number type of the norms, biases and activations.
template <weight_matrix W, math::math_number T = float>
struct block_weights
{
    std::span<const T> norm1_weight{};    ///< `d_model`, or empty: the norm before attention.
    std::span<const T> norm1_bias{};      ///< `d_model`, or empty; layer norm only.
    attention_weights<W, T> attention{};  ///< The attention.
    std::span<const T> norm2_weight{};    ///< `d_model`, or empty: the norm before the MLP.
    std::span<const T> norm2_bias{};      ///< `d_model`, or empty; layer norm only.
    mlp_weights<W, T> mlp{};              ///< The MLP.
};

/// \brief The configuration's normalisation of one row.
/// \tparam C The configuration.
/// \param x The row, `d_model`.
/// \param weight The scale, or empty.
/// \param bias The shift (layer norm), or empty.
/// \param out The normalised row; may be `x`.
template <transformer_config C, row X, mutable_row Out>
    requires std::same_as<row_value_t<X>, row_value_t<Out>>
constexpr void block_norm(const X& x, std::span<const row_value_t<X>> weight, std::span<const row_value_t<X>> bias, Out&& out)
{
    using T = row_value_t<X>;
    if constexpr (C.norm == norm_kind::layer)
        layer_norm(x, weight, bias, out, norm_eps_v<C, T>);
    else
        rms_norm(x, weight, out, norm_eps_v<C, T>);
}

/// \brief One block for one token: `x += attention(norm1(x)); x += mlp(norm2(x))`, appending to the cache at `pos`.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The token's residual stream, `d_model`, updated in place.
/// \param pos Its position.
/// \param cache The KV cache.
/// \param layer This block's layer in the cache.
/// \param rope_src The cosines and sines by position, or `no_rope`.
template <transformer_config C, weight_matrix W, math::math_number T, mutable_row X, typename R>
    requires std::same_as<row_value_t<X>, T>
constexpr void block_forward(const block_weights<W, T>& w, X&& x, std::size_t pos, const kv_cache_view<C, T>& cache, std::size_t layer,
                             const R& rope_src)
{
    const auto s = as_mutable_row(x);
    assert(s.size() == C.d_model);
    std::array<T, C.d_model> h{}, a{};
    block_norm<C>(s, w.norm1_weight, w.norm1_bias, h);
    attention_forward<C>(w.attention, h, pos, cache, layer, rope_src, a);
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
    block_norm<C>(s, w.norm2_weight, w.norm2_bias, h);
    mlp_forward<C>(w.mlp, h, a);
    for (std::size_t i = 0; i < C.d_model; ++i)
        s[i] += a[i];
}

/// \brief One block for a run of consecutive tokens, from position `first`: `block_forward` on each, in order.
/// \tparam C The configuration.
/// \param w The weights.
/// \param xs The tokens' residual streams, `d_model` each, end to end; updated in place.
/// \param first The position of the first token.
/// \param cache The KV cache.
/// \param layer This block's layer in the cache.
/// \param rope_src The cosines and sines by position, or `no_rope`.
template <transformer_config C, weight_matrix W, math::math_number T, mutable_row X, typename R>
    requires std::same_as<row_value_t<X>, T>
constexpr void block_prefill(const block_weights<W, T>& w, X&& xs, std::size_t first, const kv_cache_view<C, T>& cache, std::size_t layer,
                             const R& rope_src)
{
    const auto s = as_mutable_row(xs);
    assert(s.size() % C.d_model == 0);
    for (std::size_t t = 0; t < s.size() / C.d_model; ++t)
        block_forward<C>(w, s.subspan(t * C.d_model, C.d_model), first + t, cache, layer, rope_src);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_transformer_block_hpp
