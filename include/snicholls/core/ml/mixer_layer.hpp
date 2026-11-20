//
//  mixer_layer.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` linear-time mixer layers in place of attention: Mamba-2's and the gated delta rule's (Qwen3-Next's), batched for training with their backward passes and one token at a time for decoding with a fixed-size state; their buffers' sizes and layouts; the caches of a hybrid stack.
/// \ingroup core
///
/// **Overview.** A mixer layer maps the normalised residual rows to rows of `d_model`, as attention does, so it is a
/// third kind of attention part for the composed block (`ml/composed_block.hpp`): a hybrid stack is composed blocks
/// whose attention part is a mixer or attention, layer by layer as `layer_is_attention` says.
///
/// **Mamba-2** (Dao and Gu, 2024, section 7.1 and 8.1; llama.cpp's `build_mamba2_layer` in `src/models/mamba-base.cpp`
/// at commit 39039aa), with `d_inner = H · P`:
/// `[z | xBC | dt] = in_proj(h)`; `xBC = silu(conv1d(xBC) + b)` (`ml/short_conv.hpp`), split into `x` (`d_inner`), `B`
/// and `C` (`G · N` each); `dt = softplus(dt + dt_bias)`, `A = -exp(A_log)`; `y = SSD(x, dt, A, B, C)` (`ml/ssd.hpp`,
/// the chunked form); `y = (y + D ⊙ x) ⊙ silu(z)`; an RMS norm over each group's `d_inner / G` channels with its
/// weight (llama.cpp's grouped `ssm_norm`, mamba_ssm's `RMSNormGated` with `norm_before_gate=False`); `out_proj`.
///
/// **The gated delta rule** as Qwen3-Next's linear attention (llama.cpp's `build_layer_attn_linear` in
/// `src/models/qwen3next.cpp` at commit 39039aa): `[q | k | v | z | b | a] = in_proj(h)` (`G · N`, `G · N`, `H · P`,
/// `H · P`, `H`, `H`); `qkv = silu(conv1d(qkv))` without a bias; `q` and `k` normalised per key head,
/// `x / sqrt(Σx² + eps)` with the configuration's `norm_eps` (llama.cpp's `build_gdn_l2_norm`); `β = sigmoid(b)`,
/// `g = -exp(A_log) · softplus(a + dt_bias)`; `o = GatedDeltaRule(q, k, v, g, β)` (`ml/delta_rule.hpp`, the chunked WY
/// form); `o = rms_norm(o) ⊙ silu(z)` per head with one weight of `P` shared by the heads (`build_norm_gated`);
/// `out_proj`. The projections are contiguous in that order: a checkpoint whose `in_proj_qkvz` interleaves them by key
/// head is permuted when it is loaded.
///
/// - `mixer_forward_sequence<C>(backend, w, xs, act, out, sequences, states)`: the batched pass, saving `act`; with
///   `states` it starts from them and leaves the final ones (a prompt prefilled for decoding).
/// - `mixer_backward<C>(backend, w, xs, act, dout, dxs, grads, sequences)`: `dxs` added into, every weight's gradient
///   added into, for a forward pass from zero states.
/// - `mixer_forward<C>(w, x, state, out)`: one token, `state` of `mixer_state_values(C)` (the convolution's then the
///   recurrence's), on core's reference backend.
/// - `mixer_parameter_count`, `mixer_activation_values`, `mixer_gradient_work`, `mixer_state_values` size the buffers;
///   `mixer_weights_in`, `mixer_activations_in`, `mixer_gradients_in` lay them out (the order of the weights' fields).
/// - `recurrent_cache<T>` holds every mixer layer's state; `hybrid_cache<KV, T>` a KV cache for the attention layers
///   beside it, indexed by `hybrid_attention_index` and `hybrid_mixer_index`.
///
/// \snippet ml/mixer_layer_demo.cpp mixer_layer
///
/// **Relationships.** `ml/ssd.hpp`, `ml/delta_rule.hpp`, `ml/short_conv.hpp`, `ml/norm.hpp`, `ml/linear.hpp`,
/// `ml/composed_block.hpp` (which takes these weights as its attention part), `ml/transformer_config.hpp` (the mixer
/// fields, `layer_is_attention`). Group: \ref core.
///
/// **Implementations.** Mamba-2 and the gated delta rule, an overload each, chosen by the weights' type. Gated linear
/// attention and RWKV have their ops (`ml/gla.hpp`, `ml/rwkv.hpp`) and no layer yet.
///
/// **Guarantees.** `constexpr`, no allocation, every step in a fixed order. Decoding one token at a time gives the
/// batched pass's outputs within the chunked forms' rounding (tested at every position), and a prompt fed in pieces
/// through `states` gives the whole prompt's. The products are the backend's (`linear_sequence`), so on core's reference
/// backend the results are its bits.
///
/// **Errors.** Preconditions, asserted: the rows a multiple of `d_model · sequences`, every view of its size. A
/// configuration without that mixer is a compile error.
///
/// **Performance.** Not benchmarked as a whole: its parts are (`ml/ssd.hpp`, `ml/delta_rule.hpp`, `ml/short_conv.hpp`,
/// `ml/linear.hpp`).
///
/// **Tests and demos.** `tests/ml/mixer_layer_test.cpp`, `demos/ml/mixer_layer_demo.cpp`; the hybrid stacks in
/// `tests/ml/composed_block_test.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `linalg/reference.hpp` (decoding), `ml/activations.hpp`, `ml/delta_rule.hpp`, `ml/linear.hpp`, `ml/norm.hpp`,
/// `ml/recurrence.hpp`, `ml/short_conv.hpp`, `ml/ssd.hpp`, `ml/transformer_config.hpp`.
///
/// **Known issues.** The splits, gates and norms between the products run on the host (the products and the triangular
/// solve are the backend's): a device library overloads `mixer_forward_sequence` and `mixer_backward` for its backend
/// type. The backward pass is of a forward pass from zero states (training on whole sequences). Decoding keeps a row of
/// each activation on the stack, so a very wide layer decodes through `mixer_forward_sequence` with one row instead.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_mixer_layer_hpp
#define snicholls_core_ml_mixer_layer_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/delta_rule.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/recurrence.hpp>
#include <snicholls/core/ml/short_conv.hpp>
#include <snicholls/core/ml/ssd.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief Mamba-2's layer weights: views of the caller's memory.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct mamba2_weights
{
    dense_weights<T> in_proj{};      ///< `(2 H·P + 2 G·N + H) x d_model`: `z`, `xBC`, `dt`.
    short_conv_weights<T> conv{};    ///< `H·P + 2 G·N` channels of `conv_width` taps, with a bias.
    std::span<const T> dt_bias{};    ///< `H`.
    std::span<const T> a_log{};      ///< `H`: `A = -exp(a_log)`.
    std::span<const T> d_skip{};     ///< `H`: the `D` skip.
    std::span<const T> norm{};       ///< `H·P`: the grouped RMS norm's weight.
    dense_weights<T> out_proj{};     ///< `d_model x H·P`.
};

/// \brief The gated delta rule's layer weights (Qwen3-Next's linear attention): views of the caller's memory.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct gated_deltanet_weights
{
    dense_weights<T> in_proj{};      ///< `(2 G·N + 2 H·P + 2 H) x d_model`: `q`, `k`, `v`, `z`, `b`, `a`.
    short_conv_weights<T> conv{};    ///< `2 G·N + H·P` channels of `conv_width` taps, no bias.
    std::span<const T> dt_bias{};    ///< `H`.
    std::span<const T> a_log{};      ///< `H`: the gate's `-exp(a_log)`.
    std::span<const T> norm{};       ///< `P`: the per-head RMS norm's weight, shared by the heads.
    dense_weights<T> out_proj{};     ///< `d_model x H·P`.
};

/// \brief What a mixer layer's batched pass saves: views the caller owns (`mixer_activations_in` lays them out).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct mixer_activations
{
    std::span<T> proj{};           ///< `rows · in_proj.rows`: the input projection.
    std::span<T> conv_in{};        ///< `rows · conv channels`: the convolution's input.
    std::span<T> conv_out{};       ///< `rows · conv channels`: its output, before SiLU.
    std::span<T> q{};              ///< `rows · G·N`: Mamba-2's `C`, the rule's normalised queries.
    std::span<T> k{};              ///< `rows · G·N`: Mamba-2's `B`, the rule's normalised keys.
    std::span<T> v{};              ///< `rows · H·P`: Mamba-2's `x`, the rule's values.
    std::span<T> gate{};           ///< `rows · H`: Mamba-2's `dt` after `softplus`, the rule's log-gate `g`.
    std::span<T> beta{};           ///< `rows · H`: the rule's `β`; empty for Mamba-2.
    std::span<T> y{};              ///< `rows · H·P`: the recurrence's output.
    std::span<T> gated{};          ///< `rows · H·P`: Mamba-2's `(y + D x) silu(z)`, the norm's input; empty for the rule.
    std::span<T> rstd{};           ///< `rows · G` (Mamba-2) or `rows · H` (the rule): the norms' reciprocal RMS.
    std::span<T> normed{};         ///< `rows · H·P`: `out_proj`'s input.
    std::span<T> checkpoints{};    ///< `sequences · recurrence_checkpoints(n, mixer_chunk) · mixer_state_size`; or empty (decoding).
    std::span<T> final_states{};   ///< `sequences · mixer_state_size`: the recurrence's final states when no states are passed.
    std::span<T> work{};           ///< `delta_rule_chunk_work` for the rule; empty for Mamba-2.
};

/// \brief A mixer layer's gradients, every one added into, and the backward pass's working space: views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct mixer_gradients
{
    std::span<T> in_proj{};     ///< As the weight.
    std::span<T> conv{};        ///< As the taps.
    std::span<T> conv_bias{};   ///< As the bias; empty for the rule.
    std::span<T> dt_bias{};     ///< `H`.
    std::span<T> a_log{};       ///< `H`.
    std::span<T> d_skip{};      ///< `H`; empty for the rule.
    std::span<T> norm{};        ///< As the weight.
    std::span<T> out_proj{};    ///< As the weight.
    std::span<T> work{};        ///< `mixer_gradient_work(C, rows, sequences)` values.
};

/// \brief Every mixer layer's decoding state, one block of `per_layer` values per mixer layer.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct recurrent_cache
{
    std::span<T> data{};        ///< `layers · per_layer` values.
    std::size_t per_layer = 0;  ///< `mixer_state_values(C)`.

    /// \brief One mixer layer's state. \param i The mixer layer (`hybrid_mixer_index`). \return Its values.
    [[nodiscard]] constexpr std::span<T> layer(std::size_t i) const { return data.subspan(i * per_layer, per_layer); }
};

/// \brief A hybrid stack's caches: a KV cache for its attention layers and the mixer layers' states.
/// \ingroup core
/// \tparam KV The KV cache (a `kv_store`). \tparam T The number type.
template <typename KV, math::math_number T>
struct hybrid_cache
{
    KV kv{};                         ///< The attention layers', indexed by `hybrid_attention_index`.
    recurrent_cache<T> recurrent{};  ///< The mixer layers', indexed by `hybrid_mixer_index`.
};

/// \brief The attention layers before layer `layer` of a stack: its slot in a hybrid stack's KV cache.
/// \param c The configuration. \param layer The layer. \return The count.
[[nodiscard]] constexpr std::size_t hybrid_attention_index(const transformer_config& c, std::size_t layer) noexcept
{
    std::size_t n = 0;
    for (std::size_t i = 0; i < layer; ++i)
        n += layer_is_attention(c, i) ? 1 : 0;
    return n;
}

/// \brief The mixer layers before layer `layer` of a stack: its slot in a `recurrent_cache`.
/// \param c The configuration. \param layer The layer. \return The count.
[[nodiscard]] constexpr std::size_t hybrid_mixer_index(const transformer_config& c, std::size_t layer) noexcept
{
    return layer - hybrid_attention_index(c, layer);
}

namespace detail::mixer_layer
{
    // The widths of a configuration's layer
    struct shape
    {
        std::size_t d, H, P, N, G, K, inner, keys, conv, in, S;
    };
    constexpr shape shape_of(const transformer_config& c) noexcept
    {
        const std::size_t inner = c.mixer_heads * c.mixer_head_dim, keys = c.mixer_groups * c.mixer_state;
        const bool mamba = c.mixer == mixer_kind::mamba2;
        const std::size_t conv = mamba ? inner + 2 * keys : 2 * keys + inner;
        const std::size_t in = mamba ? 2 * inner + 2 * keys + c.mixer_heads : 2 * keys + 2 * inner + 2 * c.mixer_heads;
        return {c.d_model, c.mixer_heads, c.mixer_head_dim, c.mixer_state, c.mixer_groups, c.conv_width, inner, keys, conv, in, mixer_state_size(c)};
    }
    constexpr bool layer_kind(const transformer_config& c) noexcept
    {
        return c.mixer == mixer_kind::mamba2 || c.mixer == mixer_kind::gated_deltanet;
    }

    // to[r · tw + to_at + i] = from[r · fw + from_at + i] for i < width
    template <typename T>
    constexpr void columns(std::span<const T> from, std::size_t fw, std::size_t from_at, std::span<T> to, std::size_t tw, std::size_t to_at,
                           std::size_t width, std::size_t rows)
    {
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t i = 0; i < width; ++i)
                to[r * tw + to_at + i] = from[r * fw + from_at + i];
    }

    template <typename T>
    constexpr std::span<const T> in(std::span<T> s) noexcept
    {
        return std::span<const T>(s);
    }
} // namespace detail::mixer_layer

/// \name Sizes and layouts
/// @{

/// \brief A mixer layer's parameters, in the weights' field order.
/// \param c A configuration with a Mamba-2 or gated-delta-rule mixer. \return The number of values.
[[nodiscard]] constexpr std::size_t mixer_parameter_count(const transformer_config& c) noexcept
{
    const auto s = detail::mixer_layer::shape_of(c);
    const bool mamba = c.mixer == mixer_kind::mamba2;
    return s.in * s.d + s.conv * s.K + (mamba ? s.conv : 0) + 2 * s.H + (mamba ? s.H : 0) + (mamba ? s.inner : s.P) + s.d * s.inner;
}

/// \brief The state a sequence decodes with: the convolution's `(conv_width - 1) · channels`, then the recurrence's.
/// \param c The configuration. \return The number of values.
[[nodiscard]] constexpr std::size_t mixer_state_values(const transformer_config& c) noexcept
{
    const auto s = detail::mixer_layer::shape_of(c);
    return short_conv_state_size(s.conv, s.K) + s.S;
}

/// \brief What `mixer_forward_sequence` saves for `sequences` sequences of `rows / sequences` rows.
/// \param c The configuration. \param rows The rows. \param sequences The sequences.
/// \return The number of values, in `mixer_activations`' field order.
[[nodiscard]] constexpr std::size_t mixer_activation_values(const transformer_config& c, std::size_t rows, std::size_t sequences) noexcept
{
    const auto s = detail::mixer_layer::shape_of(c);
    const bool mamba = c.mixer == mixer_kind::mamba2;
    const std::size_t m = sequences == 0 ? 0 : recurrence_checkpoints(rows / sequences, c.mixer_chunk);
    return rows * (s.in + 2 * s.conv + 2 * s.keys + s.inner + s.H + (mamba ? 0 : s.H) + s.inner + (mamba ? s.inner : 0) + (mamba ? s.G : s.H)
                   + s.inner)
           + sequences * m * s.S + sequences * s.S + (mamba ? 0 : delta_rule_chunk_work(c));
}

/// \brief The working space `mixer_backward` needs.
/// \tparam C The configuration. \tparam T The number type.
/// \param rows The rows. \param sequences The sequences. \return The number of values.
template <transformer_config C, math::math_number T>
[[nodiscard]] constexpr std::size_t mixer_gradient_work(std::size_t rows, std::size_t sequences) noexcept
{
    const auto s = detail::mixer_layer::shape_of(C);
    const std::size_t op = C.mixer == mixer_kind::mamba2 ? ssd_backward_work<C, T>() : delta_rule_backward_work<C, T>();
    return rows * (s.in + 3 * s.inner + 2 * s.keys + 2 * s.H + 2 * s.keys + 2 * s.conv) + s.H + sequences * s.S + op;
}

/// \brief Mamba-2's weights laid out in one buffer (`mixer_parameter_count`'s order).
/// \tparam C A Mamba-2 configuration. \param params `mixer_parameter_count(C)` values. \return The views.
template <transformer_config C, math::math_number T>
    requires(C.mixer == mixer_kind::mamba2)
constexpr mamba2_weights<T> mixer_weights_in(std::span<const T> params)
{
    constexpr auto s = detail::mixer_layer::shape_of(C);
    assert(params.size() == mixer_parameter_count(C));
    detail::cutter<const T> cut{params};
    mamba2_weights<T> w{};
    w.in_proj = {cut.take(s.in * s.d), s.in, s.d};
    w.conv = {cut.take(s.conv * s.K), s.conv, s.K, {}};
    w.conv.bias = cut.take(s.conv);
    w.dt_bias = cut.take(s.H);
    w.a_log = cut.take(s.H);
    w.d_skip = cut.take(s.H);
    w.norm = cut.take(s.inner);
    w.out_proj = {cut.take(s.d * s.inner), s.d, s.inner};
    return w;
}

/// \brief The gated delta rule's weights laid out in one buffer (`mixer_parameter_count`'s order).
/// \tparam C A gated-delta-rule configuration. \param params `mixer_parameter_count(C)` values. \return The views.
template <transformer_config C, math::math_number T>
    requires(C.mixer == mixer_kind::gated_deltanet)
constexpr gated_deltanet_weights<T> mixer_weights_in(std::span<const T> params)
{
    constexpr auto s = detail::mixer_layer::shape_of(C);
    assert(params.size() == mixer_parameter_count(C));
    detail::cutter<const T> cut{params};
    gated_deltanet_weights<T> w{};
    w.in_proj = {cut.take(s.in * s.d), s.in, s.d};
    w.conv = {cut.take(s.conv * s.K), s.conv, s.K, {}};
    w.dt_bias = cut.take(s.H);
    w.a_log = cut.take(s.H);
    w.norm = cut.take(s.P);
    w.out_proj = {cut.take(s.d * s.inner), s.d, s.inner};
    return w;
}

/// \brief A mixer layer's activations laid out in one buffer.
/// \tparam C The configuration. \param values `mixer_activation_values(C, rows, sequences)` values.
/// \param rows The rows. \param sequences The sequences. \return The views.
template <transformer_config C, math::math_number T>
    requires(detail::mixer_layer::layer_kind(C))
constexpr mixer_activations<T> mixer_activations_in(std::span<T> values, std::size_t rows, std::size_t sequences)
{
    constexpr auto s = detail::mixer_layer::shape_of(C);
    constexpr bool mamba = C.mixer == mixer_kind::mamba2;
    assert(values.size() == mixer_activation_values(C, rows, sequences));
    const std::size_t m = recurrence_checkpoints(rows / sequences, C.mixer_chunk);
    detail::cutter<T> cut{values};
    mixer_activations<T> a{};
    a.proj = cut.take(rows * s.in);
    a.conv_in = cut.take(rows * s.conv);
    a.conv_out = cut.take(rows * s.conv);
    a.q = cut.take(rows * s.keys);
    a.k = cut.take(rows * s.keys);
    a.v = cut.take(rows * s.inner);
    a.gate = cut.take(rows * s.H);
    a.beta = cut.take(mamba ? 0 : rows * s.H);
    a.y = cut.take(rows * s.inner);
    a.gated = cut.take(mamba ? rows * s.inner : 0);
    a.rstd = cut.take(rows * (mamba ? s.G : s.H));
    a.normed = cut.take(rows * s.inner);
    a.checkpoints = cut.take(sequences * m * s.S);
    a.final_states = cut.take(sequences * s.S);
    a.work = cut.take(mamba ? 0 : delta_rule_chunk_work(C));
    return a;
}

/// \brief A mixer layer's gradients laid out as its weights, with the working space.
/// \tparam C The configuration. \param grads `mixer_parameter_count(C)` values, added into.
/// \param work `mixer_gradient_work<C, T>(rows, sequences)` values. \return The views.
template <transformer_config C, math::math_number T>
    requires(detail::mixer_layer::layer_kind(C))
constexpr mixer_gradients<T> mixer_gradients_in(std::span<T> grads, std::span<T> work)
{
    constexpr auto s = detail::mixer_layer::shape_of(C);
    constexpr bool mamba = C.mixer == mixer_kind::mamba2;
    assert(grads.size() == mixer_parameter_count(C));
    detail::cutter<T> cut{grads};
    mixer_gradients<T> g{};
    g.in_proj = cut.take(s.in * s.d);
    g.conv = cut.take(s.conv * s.K);
    g.conv_bias = cut.take(mamba ? s.conv : 0);
    g.dt_bias = cut.take(s.H);
    g.a_log = cut.take(s.H);
    g.d_skip = cut.take(mamba ? s.H : 0);
    g.norm = cut.take(mamba ? s.inner : s.P);
    g.out_proj = cut.take(s.d * s.inner);
    g.work = work;
    return g;
}

/// @}

namespace detail::mixer_layer
{
    // The backward pass's working space, cut once
    template <typename T>
    struct backward_work
    {
        std::span<T> dproj, dnormed, dgated, dv, dq, dk, dgate, dbeta, pre, dconv_out, dconv_in, da, dstates, op;
    };
    template <transformer_config C, typename T>
    constexpr backward_work<T> backward_work_in(std::span<T> work, std::size_t rows, std::size_t sequences)
    {
        constexpr auto s = shape_of(C);
        assert((work.size() == mixer_gradient_work<C, T>(rows, sequences)));
        cutter<T> cut{work};
        backward_work<T> b{};
        b.dproj = cut.take(rows * s.in);
        b.dnormed = cut.take(rows * s.inner);
        b.dgated = cut.take(rows * s.inner);
        b.dv = cut.take(rows * s.inner);
        b.dq = cut.take(rows * s.keys);
        b.dk = cut.take(rows * s.keys);
        b.dgate = cut.take(rows * s.H);
        b.dbeta = cut.take(rows * s.H);
        b.pre = cut.take(rows * 2 * s.keys);
        b.dconv_out = cut.take(rows * s.conv);
        b.dconv_in = cut.take(rows * s.conv);
        b.da = cut.take(s.H);
        b.dstates = cut.take(sequences * s.S);
        b.op = cut.rest;
        return b;
    }
} // namespace detail::mixer_layer

/// \brief Mamba-2's layer over `sequences` sequences laid end to end, saving what `mixer_backward` reads.
/// \tparam C A Mamba-2 configuration.
/// \param backend A `linalg::blas_backend`: the products.
/// \param w The weights.
/// \param xs `rows · d_model`: the normalised residual rows.
/// \param act Receives the activations (`mixer_activations_in`).
/// \param out Receives `rows · d_model`.
/// \param sequences The sequences.
/// \param states `sequences · mixer_state_values(C)`, the convolution's for every sequence then the recurrence's: the
/// initial states, replaced by the final; or empty for zeros (the final recurrent states land in `act.final_states`).
template <transformer_config C, typename B, math::math_number T>
    requires(C.mixer == mixer_kind::mamba2 && linalg::blas_backend<B, T>)
constexpr void mixer_forward_sequence(const B& backend, const mamba2_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                      const mixer_activations<T>& act, std::type_identity_t<std::span<T>> out, std::size_t sequences = 1,
                                      std::type_identity_t<std::span<T>> states = {})
{
    using namespace detail::mixer_layer;
    constexpr auto s = shape_of(C);
    const std::size_t rows = xs.size() / s.d, sk = short_conv_state_size(s.conv, s.K);
    assert(xs.size() == rows * s.d && out.size() == xs.size() && sequences > 0 && rows % sequences == 0
           && (states.empty() || states.size() == sequences * (sk + s.S)));
    linear_sequence(backend, w.in_proj, xs, act.proj);
    columns<T>(in(act.proj), s.in, s.inner, act.conv_in, s.conv, 0, s.conv, rows);
    short_conv_forward(backend, w.conv, in(act.conv_in), act.conv_out, sequences, states.empty() ? std::span<T>{} : states.first(sequences * sk));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < s.inner; ++i)
            act.v[r * s.inner + i] = silu(act.conv_out[r * s.conv + i]);
        for (std::size_t i = 0; i < s.keys; ++i) {
            act.k[r * s.keys + i] = silu(act.conv_out[r * s.conv + s.inner + i]);
            act.q[r * s.keys + i] = silu(act.conv_out[r * s.conv + s.inner + s.keys + i]);
        }
        for (std::size_t h = 0; h < s.H; ++h)
            act.gate[r * s.H + h] = softplus(act.proj[r * s.in + 2 * s.inner + 2 * s.keys + h] + w.dt_bias[h]);
    }
    std::array<T, s.H> a{};
    for (std::size_t h = 0; h < s.H; ++h)
        a[h] = -math::exp(w.a_log[h]);
    const auto st = states.empty() ? act.final_states : states.subspan(sequences * sk);
    if (states.empty())
        fill_rows(backend, act.final_states, T(0));
    ssd_forward_chunked<C>(backend, ssd_inputs<T>{in(act.v), in(act.gate), std::span<const T>(a), in(act.k), in(act.q)}, act.y, sequences, st,
                           act.checkpoints);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < s.inner; ++i) {
            const std::size_t j = r * s.inner + i;
            act.gated[j] = (act.y[j] + w.d_skip[i / s.P] * act.v[j]) * silu(act.proj[r * s.in + i]);
        }
        constexpr std::size_t width = s.inner / s.G;
        for (std::size_t g = 0; g < s.G; ++g)
            act.rstd[r * s.G + g] = rms_norm(in(act.gated).subspan(r * s.inner + g * width, width), w.norm.subspan(g * width, width),
                                             act.normed.subspan(r * s.inner + g * width, width), norm_eps_v<C, T>);
    }
    linear_sequence(backend, w.out_proj, in(act.normed), out);
}

/// \brief The backward pass of Mamba-2's layer, for a forward pass from zero states: `dxs` and every weight's gradient
/// added into.
/// \tparam C The configuration the forward pass took.
/// \param backend A `linalg::blas_backend`. \param w The weights. \param xs The forward pass's input.
/// \param act What it saved. \param dout The gradient in its output. \param dxs Added into. \param grads The gradients.
/// \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T>
    requires(C.mixer == mixer_kind::mamba2 && linalg::blas_backend<B, T>)
constexpr void mixer_backward(const B& backend, const mamba2_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                              const mixer_activations<T>& act, std::type_identity_t<std::span<const T>> dout,
                              std::type_identity_t<std::span<T>> dxs, const mixer_gradients<T>& grads, std::size_t sequences = 1)
{
    using namespace detail::mixer_layer;
    constexpr auto s = shape_of(C);
    const std::size_t rows = xs.size() / s.d;
    assert(dout.size() == xs.size() && dxs.size() == xs.size());
    const auto b = backward_work_in<C, T>(grads.work, rows, sequences);
    fill_rows(backend, b.dnormed, T(0));
    linear_sequence_backward(backend, w.out_proj, in(act.normed), dout, b.dnormed, grads.out_proj);
    fill_rows(backend, b.dproj, T(0));
    constexpr std::size_t width = s.inner / s.G;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t g = 0; g < s.G; ++g)
            rms_norm_backward(in(b.dnormed).subspan(r * s.inner + g * width, width), in(act.gated).subspan(r * s.inner + g * width, width),
                              w.norm.subspan(g * width, width), act.rstd[r * s.G + g], b.dgated.subspan(r * s.inner + g * width, width),
                              grads.norm.subspan(g * width, width));
        for (std::size_t i = 0; i < s.inner; ++i) {   // gated = (y + D x) silu(z): dgated becomes dy
            const std::size_t j = r * s.inner + i, h = i / s.P;
            const T z = act.proj[r * s.in + i], yv = act.y[j] + w.d_skip[h] * act.v[j], dg = b.dgated[j];
            b.dproj[r * s.in + i] = dg * yv * silu_derivative(z);
            b.dgated[j] = dg * silu(z);
            grads.d_skip[h] += b.dgated[j] * act.v[j];
        }
    }
    std::array<T, s.H> a{};
    for (std::size_t h = 0; h < s.H; ++h)
        a[h] = -math::exp(w.a_log[h]);
    fill_rows(backend, b.da, T(0));
    fill_rows(backend, b.dstates, T(0));
    ssd_backward<C>(backend, ssd_inputs<T>{in(act.v), in(act.gate), std::span<const T>(a), in(act.k), in(act.q)}, in(b.dgated),
                    ssd_gradients<T>{b.dv, b.dgate, b.da, b.dk, b.dq}, sequences, in(act.checkpoints), b.dstates, b.op);
    for (std::size_t h = 0; h < s.H; ++h)
        grads.a_log[h] += b.da[h] * a[h];
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t h = 0; h < s.H; ++h) {   // dt = softplus(raw + dt_bias)
            const std::size_t col = r * s.in + 2 * s.inner + 2 * s.keys + h;
            const T d = b.dgate[r * s.H + h] * sigmoid(act.proj[col] + w.dt_bias[h]);
            b.dproj[col] = d;
            grads.dt_bias[h] += d;
        }
        for (std::size_t i = 0; i < s.conv; ++i) {   // xBC = silu(conv_out)
            const std::size_t j = r * s.conv + i;
            const T dpost = i < s.inner ? b.dv[r * s.inner + i] + w.d_skip[i / s.P] * b.dgated[r * s.inner + i]
                            : i < s.inner + s.keys ? b.dk[r * s.keys + i - s.inner]
                                                   : b.dq[r * s.keys + i - s.inner - s.keys];
            b.dconv_out[j] = dpost * silu_derivative(act.conv_out[j]);
        }
    }
    short_conv_backward(backend, w.conv, in(act.conv_in), in(b.dconv_out), b.dconv_in, grads.conv, grads.conv_bias, sequences);
    columns<T>(in(b.dconv_in), s.conv, 0, b.dproj, s.in, s.inner, s.conv, rows);
    linear_sequence_backward(backend, w.in_proj, xs, in(b.dproj), dxs, grads.in_proj);
}

/// \brief The gated delta rule's layer (Qwen3-Next's linear attention) over `sequences` sequences laid end to end,
/// saving what `mixer_backward` reads.
/// \tparam C A gated-delta-rule configuration.
/// \param backend A `linalg::blas_backend`: the products and the triangular solves.
/// \param w The weights. \param xs `rows · d_model`. \param act Receives the activations. \param out Receives
/// `rows · d_model`. \param sequences The sequences.
/// \param states `sequences · mixer_state_values(C)` (the convolution's, then the recurrence's), replaced by the final;
/// or empty for zeros.
template <transformer_config C, typename B, math::math_number T>
    requires(C.mixer == mixer_kind::gated_deltanet && linalg::blas_backend<B, T>)
constexpr void mixer_forward_sequence(const B& backend, const gated_deltanet_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                                      const mixer_activations<T>& act, std::type_identity_t<std::span<T>> out, std::size_t sequences = 1,
                                      std::type_identity_t<std::span<T>> states = {})
{
    using namespace detail::mixer_layer;
    constexpr auto s = shape_of(C);
    const std::size_t rows = xs.size() / s.d, sk = short_conv_state_size(s.conv, s.K);
    assert(xs.size() == rows * s.d && out.size() == xs.size() && sequences > 0 && rows % sequences == 0
           && (states.empty() || states.size() == sequences * (sk + s.S)));
    linear_sequence(backend, w.in_proj, xs, act.proj);
    columns<T>(in(act.proj), s.in, 0, act.conv_in, s.conv, 0, s.conv, rows);
    short_conv_forward(backend, w.conv, in(act.conv_in), act.conv_out, sequences, states.empty() ? std::span<T>{} : states.first(sequences * sk));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < s.keys; ++i) {
            act.q[r * s.keys + i] = silu(act.conv_out[r * s.conv + i]);
            act.k[r * s.keys + i] = silu(act.conv_out[r * s.conv + s.keys + i]);
        }
        for (std::size_t i = 0; i < s.inner; ++i)
            act.v[r * s.inner + i] = silu(act.conv_out[r * s.conv + 2 * s.keys + i]);
        for (std::size_t h = 0; h < s.H; ++h) {
            const std::size_t at = r * s.in + 2 * s.keys + 2 * s.inner;
            act.beta[r * s.H + h] = sigmoid(act.proj[at + h]);
            act.gate[r * s.H + h] = -math::exp(w.a_log[h]) * softplus(act.proj[at + s.H + h] + w.dt_bias[h]);
        }
    }
    l2_normalise_rows(backend, in(act.q), act.q, s.N, norm_eps_v<C, T>);
    l2_normalise_rows(backend, in(act.k), act.k, s.N, norm_eps_v<C, T>);
    const auto st = states.empty() ? act.final_states : states.subspan(sequences * sk);
    if (states.empty())
        fill_rows(backend, act.final_states, T(0));
    delta_rule_forward_chunked<C>(backend, delta_rule_inputs<T>{in(act.q), in(act.k), in(act.v), in(act.gate), in(act.beta)}, act.y, sequences,
                                  st, act.checkpoints, act.work);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t h = 0; h < s.H; ++h) {
            const std::size_t at = r * s.inner + h * s.P;
            act.rstd[r * s.H + h] = rms_norm(in(act.y).subspan(at, s.P), w.norm, act.normed.subspan(at, s.P), norm_eps_v<C, T>);
            for (std::size_t p = 0; p < s.P; ++p)
                act.normed[at + p] *= silu(act.proj[r * s.in + 2 * s.keys + s.inner + h * s.P + p]);
        }
    linear_sequence(backend, w.out_proj, in(act.normed), out);
}

/// \brief The backward pass of the gated delta rule's layer, for a forward pass from zero states: `dxs` and every
/// weight's gradient added into.
/// \tparam C The configuration the forward pass took.
/// \param backend A `linalg::blas_backend`. \param w The weights. \param xs The forward pass's input.
/// \param act What it saved. \param dout The gradient in its output. \param dxs Added into. \param grads The gradients.
/// \param sequences The sequences.
template <transformer_config C, typename B, math::math_number T>
    requires(C.mixer == mixer_kind::gated_deltanet && linalg::blas_backend<B, T>)
constexpr void mixer_backward(const B& backend, const gated_deltanet_weights<T>& w, std::type_identity_t<std::span<const T>> xs,
                              const mixer_activations<T>& act, std::type_identity_t<std::span<const T>> dout,
                              std::type_identity_t<std::span<T>> dxs, const mixer_gradients<T>& grads, std::size_t sequences = 1)
{
    using namespace detail::mixer_layer;
    constexpr auto s = shape_of(C);
    const std::size_t rows = xs.size() / s.d;
    assert(dout.size() == xs.size() && dxs.size() == xs.size());
    const auto b = backward_work_in<C, T>(grads.work, rows, sequences);
    fill_rows(backend, b.dnormed, T(0));
    linear_sequence_backward(backend, w.out_proj, in(act.normed), dout, b.dnormed, grads.out_proj);
    fill_rows(backend, b.dproj, T(0));
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t h = 0; h < s.H; ++h) {   // normed = rms_norm(y) silu(z)
            const std::size_t at = r * s.inner + h * s.P, zat = r * s.in + 2 * s.keys + s.inner + h * s.P;
            const T rstd = act.rstd[r * s.H + h];
            for (std::size_t p = 0; p < s.P; ++p) {
                const T z = act.proj[zat + p], n = act.y[at + p] * rstd * w.norm[p], d = b.dnormed[at + p];
                b.dproj[zat + p] = d * n * silu_derivative(z);
                b.dnormed[at + p] = d * silu(z);
            }
            rms_norm_backward(in(b.dnormed).subspan(at, s.P), in(act.y).subspan(at, s.P), w.norm, rstd, b.dgated.subspan(at, s.P), grads.norm);
        }
    fill_rows(backend, b.dstates, T(0));
    delta_rule_backward<C>(backend, delta_rule_inputs<T>{in(act.q), in(act.k), in(act.v), in(act.gate), in(act.beta)}, in(b.dgated),
                           delta_rule_gradients<T>{b.dq, b.dk, b.dv, b.dgate, b.dbeta}, sequences, in(act.checkpoints), b.dstates, b.op);
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t at = r * s.in + 2 * s.keys + 2 * s.inner;
        for (std::size_t h = 0; h < s.H; ++h) {
            const T beta = act.beta[r * s.H + h];
            b.dproj[at + h] = b.dbeta[r * s.H + h] * beta * (T(1) - beta);
            const T e = math::exp(w.a_log[h]), dg = b.dgate[r * s.H + h];   // g = -e softplus(a + dt_bias)
            const T da = -e * sigmoid(act.proj[at + s.H + h] + w.dt_bias[h]) * dg;
            b.dproj[at + s.H + h] = da;
            grads.dt_bias[h] += da;
            grads.a_log[h] += dg * act.gate[r * s.H + h];
        }
        for (std::size_t i = 0; i < s.keys; ++i) {   // the normalisations' inputs, silu(conv_out)
            b.pre[r * 2 * s.keys + i] = silu(act.conv_out[r * s.conv + i]);
            b.pre[r * 2 * s.keys + s.keys + i] = silu(act.conv_out[r * s.conv + s.keys + i]);
        }
    }
    for (std::size_t r = 0; r < rows; ++r)   // through the L2 normalisation, in place in dq and dk
        for (std::size_t g = 0; g < s.G; ++g) {
            const auto qx = in(b.pre).subspan(r * 2 * s.keys + g * s.N, s.N), kx = in(b.pre).subspan(r * 2 * s.keys + s.keys + g * s.N, s.N);
            l2_normalise_rows_backward(backend, qx, in(b.dq).subspan(r * s.keys + g * s.N, s.N), b.dq.subspan(r * s.keys + g * s.N, s.N), s.N,
                                       norm_eps_v<C, T>);
            l2_normalise_rows_backward(backend, kx, in(b.dk).subspan(r * s.keys + g * s.N, s.N), b.dk.subspan(r * s.keys + g * s.N, s.N), s.N,
                                       norm_eps_v<C, T>);
        }
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t i = 0; i < s.conv; ++i) {
            const std::size_t j = r * s.conv + i;
            const T dpost = i < s.keys ? b.dq[r * s.keys + i] : i < 2 * s.keys ? b.dk[r * s.keys + i - s.keys] : b.dv[r * s.inner + i - 2 * s.keys];
            b.dconv_out[j] = dpost * silu_derivative(act.conv_out[j]);
        }
    short_conv_backward(backend, w.conv, in(act.conv_in), in(b.dconv_out), b.dconv_in, grads.conv, std::span<T>{}, sequences);
    columns<T>(in(b.dconv_in), s.conv, 0, b.dproj, s.in, 0, s.conv, rows);
    linear_sequence_backward(backend, w.in_proj, xs, in(b.dproj), dxs, grads.in_proj);
}

namespace detail::mixer_layer
{
    template <transformer_config C, typename W, typename T>
    constexpr void decode(const W& w, std::span<const T> x, std::span<T> state, std::span<T> out)
    {
        constexpr std::size_t full = mixer_activation_values(C, 1, 1), states = 2 * shape_of(C).S;
        std::array<T, full - states> values{};
        mixer_activations<T> a{};
        cutter<T> cut{std::span<T>(values)};
        constexpr auto s = shape_of(C);
        constexpr bool mamba = C.mixer == mixer_kind::mamba2;
        a.proj = cut.take(s.in);
        a.conv_in = cut.take(s.conv);
        a.conv_out = cut.take(s.conv);
        a.q = cut.take(s.keys);
        a.k = cut.take(s.keys);
        a.v = cut.take(s.inner);
        a.gate = cut.take(s.H);
        a.beta = cut.take(mamba ? 0 : s.H);
        a.y = cut.take(s.inner);
        a.gated = cut.take(mamba ? s.inner : 0);
        a.rstd = cut.take(mamba ? s.G : s.H);
        a.normed = cut.take(s.inner);
        a.checkpoints = {};
        a.final_states = {};
        a.work = cut.rest;
        mixer_forward_sequence<C>(linalg::reference, w, x, a, out, 1, state);
    }
} // namespace detail::mixer_layer

/// \brief One token of Mamba-2's layer for decoding, on core's reference backend.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The token's normalised residual row, `d_model`.
/// \param state `mixer_state_values(C)`: the sequence's state (a prefill's `states`), updated.
/// \param out Receives `d_model`.
template <transformer_config C, math::math_number T>
    requires(C.mixer == mixer_kind::mamba2)
constexpr void mixer_forward(const mamba2_weights<T>& w, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<T>> state,
                             std::type_identity_t<std::span<T>> out)
{
    assert(x.size() == C.d_model && out.size() == C.d_model && state.size() == mixer_state_values(C));
    detail::mixer_layer::decode<C>(w, x, state, out);
}

/// \brief One token of the gated delta rule's layer for decoding, on core's reference backend.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The token's normalised residual row, `d_model`.
/// \param state `mixer_state_values(C)`: the sequence's state, updated.
/// \param out Receives `d_model`.
template <transformer_config C, math::math_number T>
    requires(C.mixer == mixer_kind::gated_deltanet)
constexpr void mixer_forward(const gated_deltanet_weights<T>& w, std::type_identity_t<std::span<const T>> x,
                             std::type_identity_t<std::span<T>> state, std::type_identity_t<std::span<T>> out)
{
    assert(x.size() == C.d_model && out.size() == C.d_model && state.size() == mixer_state_values(C));
    detail::mixer_layer::decode<C>(w, x, state, out);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_mixer_layer_hpp
