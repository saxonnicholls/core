//
//  attention_layer.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::attention_forward`: a transformer's attention for one token at one position: the Q, K and V projections, QK norm, RoPE, the KV cache, grouped-query attention, the soft cap and the output projection, as the configuration says.
/// \ingroup core
///
/// **Overview.** The generic multi-head attention layer. For a token `x` at position `pos`:
///
/// 1. `q = Wq x`, `k = Wk x`, `v = Wv x` (plus biases), from one prepared input;
/// 2. per head, an RMS norm of `q` and `k` before or after the rotation (`qk_norm`) and RoPE at `pos` (`rope`);
/// 3. `k` and `v` appended to the `kv_cache` at `pos`;
/// 4. each query head `h` attends to key-value head `h / group()` over the visible positions
///    (`attention_window(pos, window)`), each score `scale · q·k` capped by `attention_softcap · tanh(s / cap)` if set;
/// 5. `out = Wo y` (plus its bias) over the heads' outputs.
///
/// Prefill is this for each token of a run in order, so a prompt and a decode step are one code path. Attention
/// never forms the scores of a sequence: each head folds its keys in one pass (`attention_accumulator`), and a
/// sliding window's two runs are folded in position order, so a ring cache gives the flat cache's bits.
///
/// `attention_head` is one head on its own: the unit to run on threads or devices.
///
/// \snippet ml/attention_layer_demo.cpp attention_layer
///
/// **Relationships.** `ml/transformer_config.hpp`, `ml/linear.hpp`, `ml/kv_cache.hpp`, `ml/attention.hpp`,
/// `ml/rope.hpp` (`rope_table` is a rope source), `ml/norm.hpp`, `ml/activations.hpp` (`softcap`),
/// `ml/transformer_block.hpp` (the user). Group: \ref core.
///
/// **Implementations.** One, configured at compile time.
///
/// **Guarantees.** `constexpr`, no allocation (the projections are `std::array`s of the configuration's sizes),
/// O(`d_model · (q_size + 2 kv_size)` + `n_heads · visible · head_dim` + `q_size · d_model`) per token.
///
/// **Errors.** Preconditions, asserted: the matrices have the configuration's shapes (`q` is `q_size x d_model`,
/// `k` and `v` `kv_size x d_model`, `o` `d_model x q_size`), biases and QK-norm weights are empty or of their row's
/// size, the rope source covers `pos`, the cache's capacity covers the visible range. An invalid configuration is a
/// compile error.
///
/// **Performance.** Benchmarked through `ml/transformer_block.hpp`'s decode step.
///
/// **Tests and demos.** `tests/ml/attention_layer_test.cpp`, `demos/ml/attention_layer_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, and the `ml/` headers above.
///
/// **Known issues.** The score scale is `1 / sqrt(head_dim)`, rounded once to `T`.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_attention_layer_hpp
#define snicholls_core_ml_attention_layer_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/attention.hpp>
#include <snicholls/core/ml/kv_cache.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/rope.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief The rope source of a configuration without RoPE.
struct no_rope_t
{
};

/// \brief Pass where a configuration has `rope_kind::none`.
inline constexpr no_rope_t no_rope{};

/// \brief A source of RoPE's cosines and sines by position, such as `rope_table`.
template <typename R>
concept rope_source = requires(const R& r, std::size_t pos) {
    r.cos_at(pos);
    r.sin_at(pos);
};

/// \brief The weights of an attention layer: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix`.
/// \tparam T The number type of the biases, norms and activations.
template <weight_matrix W, math::math_number T = float>
struct attention_weights
{
    W q{};                              ///< `q_size x d_model`.
    W k{};                              ///< `kv_size x d_model`.
    W v{};                              ///< `kv_size x d_model`.
    W o{};                              ///< `d_model x q_size`.
    std::span<const T> q_bias{};        ///< `q_size`, or empty.
    std::span<const T> k_bias{};        ///< `kv_size`, or empty.
    std::span<const T> v_bias{};        ///< `kv_size`, or empty.
    std::span<const T> o_bias{};        ///< `d_model`, or empty.
    std::span<const T> q_norm{};        ///< `head_dim` QK-norm weights, or empty for a parameter-free norm.
    std::span<const T> k_norm{};        ///< `head_dim` QK-norm weights, or empty.
};

/// \brief One query head over its visible keys: `out = softmax(cap(scale · q·k)) · v`, in one pass.
/// \tparam C The configuration (its `head_dim` and `attention_softcap`).
/// \param q The head's query, `head_dim`.
/// \param runs The visible keys and values, from `kv_cache_view::visible`.
/// \param out The head's output, `head_dim`.
/// \return `logsumexp` of the scores, for a backward pass; -infinity with no keys.
template <transformer_config C, math::math_number T, row Q, mutable_row Out>
    requires std::same_as<row_value_t<Q>, T> && std::same_as<row_value_t<Out>, T>
constexpr T attention_head(const Q& q, const kv_runs<T>& runs, Out&& out)
{
    const auto qs = as_row(q);
    assert(qs.size() == C.head_dim);
    const T scale = T(1) / math::sqrt(T(C.head_dim));
    attention_accumulator<T, C.head_dim> acc{};
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t j = 0; j < runs.keys[r].size() / C.head_dim; ++j) {
            T s = scale * detail::attention::dot(qs, runs.keys[r].subspan(j * C.head_dim, C.head_dim));
            if constexpr (C.attention_softcap > 0)
                s = softcap(s, attention_softcap_v<C, T>);
            acc.add(s, runs.values[r].subspan(j * C.head_dim, C.head_dim));
        }
    acc.result(out);
    return acc.log_sum_exp();
}

namespace detail::attention_layer
{
    // RoPE then, or before, the QK norm of one head, as the configuration says
    template <transformer_config C, typename T, typename R>
    constexpr void position(std::span<T> head, std::span<const T> norm, std::size_t pos, const R& rope_src)
    {
        constexpr T eps = norm_eps_v<C, T>;
        if constexpr (C.qk_norm == qk_norm_kind::before_rope)
            rms_norm(head, norm, head, eps);
        if constexpr (C.rope != rope_kind::none)
            ml::rope(head, rope_src.cos_at(pos), rope_src.sin_at(pos), C.layout());
        if constexpr (C.qk_norm == qk_norm_kind::after_rope)
            rms_norm(head, norm, head, eps);
    }
} // namespace detail::attention_layer

/// \brief Attention for one token at position `pos`: projects, rotates, caches, attends and projects back.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The token's (normalised) input, `d_model`.
/// \param pos Its position.
/// \param cache The KV cache; this token's keys and values are appended to `layer` at `pos`.
/// \param layer The layer.
/// \param rope_src The cosines and sines by position (a `rope_table` of `head_dim`), or `no_rope`.
/// \param out The output, `d_model`; may not alias `x`.
template <transformer_config C, weight_matrix W, math::math_number T, row X, typename R, mutable_row Out>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Out>, T>
             && (C.rope == rope_kind::none || rope_source<R>)
constexpr void attention_forward(const attention_weights<W, T>& w, const X& x, std::size_t pos, const kv_cache_view<C, T>& cache,
                                 std::size_t layer, const R& rope_src, Out&& out)
{
    static_assert(valid(C), "snicholls::ml::attention_forward: the transformer_config is not valid");
    assert(w.q.rows == C.q_size() && w.k.rows == C.kv_size() && w.v.rows == C.kv_size() && w.o.rows == C.d_model
           && w.q.cols == C.d_model && w.k.cols == C.d_model && w.v.cols == C.d_model && w.o.cols == C.q_size());
    constexpr std::size_t d = C.head_dim;
    std::array<T, C.q_size()> q{};
    std::array<T, C.kv_size()> k{}, v{};
    const auto in = prepare_input<C.d_model>(w.q, x);
    linear(w.q, in, q, w.q_bias);
    linear(w.k, in, k, w.k_bias);
    linear(w.v, in, v, w.v_bias);
    for (std::size_t h = 0; h < C.n_heads; ++h)
        detail::attention_layer::position<C>(std::span<T>(q).subspan(h * d, d), w.q_norm, pos, rope_src);
    for (std::size_t h = 0; h < C.n_kv_heads; ++h)
        detail::attention_layer::position<C>(std::span<T>(k).subspan(h * d, d), w.k_norm, pos, rope_src);
    cache.append(layer, pos, k, v);
    std::array<T, C.q_size()> y{};
    const key_range visible = attention_window(pos, C.window);
    for (std::size_t h = 0; h < C.n_heads; ++h)
        attention_head<C>(std::span<const T>(q).subspan(h * d, d), cache.visible(layer, h / C.group(), visible),
                          std::span<T>(y).subspan(h * d, d));
    linear(w.o, prepare_input<C.q_size()>(w.o, y), out, w.o_bias);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_attention_layer_hpp
