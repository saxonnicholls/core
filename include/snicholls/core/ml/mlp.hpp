//
//  mlp.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::mlp_forward` and `mlp_backward`: a transformer's MLP for one token, `down(act(up x))` or a gated unit's `down(glu(gate x, up x))` (SwiGLU's `silu(gate x) · up x`, ReGLU, the GEGLUs, gpt-oss's clamped SwiGLU), as the configuration says, and its backward pass; `moe_gate` and its backward.
/// \ingroup core
///
/// **Overview.** The feed-forward half of a transformer block. `transformer_config::mlp` chooses the form: GELU
/// (exact or tanh, GPT-2; quick, CLIP), ReLU² (nanochat) or a gated linear unit (`ml/glu.hpp`: SwiGLU, Llama; ReGLU; GEGLU,
/// Gemma; gpt-oss's clamped SwiGLU). The weights are `mlp_weights`, views over any
/// `weight_matrix` storage, so the same call runs on float, bfloat16 or quantised weights. The input is prepared
/// once for `up` and `gate`.
///
/// For training, `mlp_forward(w, x, y, save)` also writes the pre-activations into `mlp_activations` (the output is
/// the same to the bit), and `mlp_backward` reads them back: it recomputes the hidden row, then adds the gradients
/// into `mlp_gradients` and the input's. Only dense weights train; a quantised MLP's backward is a compile error.
/// `mlp_forward_sequence` and `mlp_backward_sequence` are the batched pair over a sequence: each product one `gemm`
/// through a `linalg` backend, the hidden rows saved (`mlp_activations::hidden`), the backward's working space in
/// `mlp_gradients::scratch`.
///
/// For a mixture of experts, `moe_gate<K>(router_logits)` picks the `K` experts a token goes to and their weights
/// (a softmax over the `K` largest router logits, as Mixtral); the token's output is the weighted sum of those
/// experts' `mlp_forward`s, each expert's weights a `mlp_weights` of its own. `moe_gate_backward` takes the gradient in
/// those weights back to the router logits.
///
/// \snippet ml/mlp_demo.cpp mlp
///
/// \snippet ml/mlp_demo.cpp mlp_gated
///
/// \snippet ml/mlp_demo.cpp mlp_backward
///
/// \snippet ml/mlp_demo.cpp mlp_sequence
///
/// **Relationships.** `ml/transformer_config.hpp`, `ml/linear.hpp`, `ml/activations.hpp`,
/// `ml/transformer_block.hpp` (the user). Group: \ref core.
///
/// **Implementations.** One, by `mlp_kind`.
///
/// **Guarantees.** `constexpr`, no allocation (the hidden row is a `std::array` of `d_ff`), linear in the weights.
/// `moe_gate` is O(experts · K); on ties the lower-numbered expert is chosen first.
///
/// **Errors.** Preconditions, asserted: the matrices have the configuration's shapes (`up` and `gate` are
/// `d_ff x d_model`, `down` is `d_model x d_ff`), biases are empty or of the output's size, `x` and `y` have `d_model`.
/// An invalid configuration is a compile error.
///
/// **Performance.** Benchmarked through `ml/transformer_block.hpp`: its decode step, and its training passes, where
/// `mlp_backward` runs inside the block's backward, within 3% of the same loops by hand.
///
/// **Tests and demos.** `tests/ml/mlp_test.cpp`, `demos/ml/mlp_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `ml/activations.hpp`, `ml/linear.hpp`, `ml/row.hpp`, `ml/sampling.hpp` (`top_k`), `ml/transformer_config.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.8.0; the gated kinds other than SwiGLU, `mlp_glu_kind`, `mlp_glu` and
/// `mlp_glu_backward` new in 0.17.0.

#ifndef snicholls_core_ml_mlp_hpp
#define snicholls_core_ml_mlp_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/glu.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The weights of an MLP: views the caller fills.
/// \ingroup core
/// \tparam W A `weight_matrix` (`dense_weights<S>` or `quantized_weights<Block>`).
/// \tparam T The number type of the biases and activations.
template <weight_matrix W, math::math_number T = float>
struct mlp_weights
{
    W up{};                             ///< `d_ff x d_model`.
    W gate{};                           ///< `d_ff x d_model`, gated kinds only.
    W down{};                           ///< `d_model x d_ff`.
    std::span<const T> up_bias{};       ///< `d_ff`, or empty.
    std::span<const T> gate_bias{};     ///< `d_ff`, or empty; gated kinds only.
    std::span<const T> down_bias{};     ///< `d_model`, or empty.
};

/// \brief The activation of a configuration's MLP; for a gated kind, the gate's activation alone (the MLP combines
/// gate and up with `mlp_glu` instead; for the clamped SwiGLUs this is the unclamped SiLU).
/// \tparam K The MLP's kind.
/// \param x A hidden value.
/// \return The activated value.
template <mlp_kind K, math::math_number T>
constexpr T mlp_activation(T x)
{
    switch (K) {
    case mlp_kind::gelu: return gelu(x);
    case mlp_kind::gelu_tanh: return gelu_tanh(x);
    case mlp_kind::relu_squared: return relu_squared(x);
    case mlp_kind::swiglu: return silu(x);
    case mlp_kind::gelu_quick: return gelu_quick(x);
    case mlp_kind::reglu: return relu(x);
    case mlp_kind::geglu: return gelu_tanh(x);
    case mlp_kind::geglu_erf: return gelu(x);
    case mlp_kind::geglu_quick: return gelu_quick(x);
    case mlp_kind::swiglu_oai:
    case mlp_kind::swiglu_clamp: return silu(x);
    }
    return x;
}

/// \brief What an MLP's forward pass saves for its backward: views the caller owns, each `d_ff`.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct mlp_activations
{
    std::span<T> up{};     ///< `up x + b`, before the activation.
    std::span<T> gate{};   ///< `gate x + b`, before SiLU; gated kinds only, else empty.
    std::span<T> hidden{}; ///< The hidden row after the activation, `down`'s input: saved by `mlp_forward_sequence` only.
};

/// \brief The gradients of an MLP's weights: views the caller owns, each added into.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct mlp_gradients
{
    std::span<T> up{};          ///< `d_ff · d_model`.
    std::span<T> gate{};        ///< `d_ff · d_model`; gated kinds only, else empty.
    std::span<T> down{};        ///< `d_model · d_ff`.
    std::span<T> up_bias{};     ///< `d_ff`, or empty.
    std::span<T> gate_bias{};   ///< `d_ff`, or empty.
    std::span<T> down_bias{};   ///< `d_model`, or empty.
    std::span<T> scratch{};     ///< `n · d_ff` (twice that for a gated kind): `mlp_backward_sequence`'s working space; else empty.
};

/// \brief The derivative of a configuration's activation; for a gated kind, of the gate's activation alone.
/// \tparam K The MLP's kind.
/// \param x A hidden value before the activation.
/// \return `act'(x)`.
template <mlp_kind K, math::math_number T>
constexpr T mlp_activation_derivative(T x)
{
    switch (K) {
    case mlp_kind::gelu: return gelu_derivative(x);
    case mlp_kind::gelu_tanh: return gelu_tanh_derivative(x);
    case mlp_kind::relu_squared: return relu_squared_derivative(x);
    case mlp_kind::swiglu: return silu_derivative(x);
    case mlp_kind::gelu_quick: return gelu_quick_derivative(x);
    case mlp_kind::reglu: return relu_derivative(x);
    case mlp_kind::geglu: return gelu_tanh_derivative(x);
    case mlp_kind::geglu_erf: return gelu_derivative(x);
    case mlp_kind::geglu_quick: return gelu_quick_derivative(x);
    case mlp_kind::swiglu_oai:
    case mlp_kind::swiglu_clamp: return silu_derivative(x);
    }
    return T(1);
}

/// \brief The `glu_kind` of a gated MLP kind; `swiglu` for any other (unused there).
/// \param k The kind.
/// \return The form `ml/glu.hpp` computes.
[[nodiscard]] constexpr glu_kind mlp_glu_kind(mlp_kind k) noexcept
{
    switch (k) {
    case mlp_kind::reglu: return glu_kind::reglu;
    case mlp_kind::geglu: return glu_kind::geglu;
    case mlp_kind::geglu_erf: return glu_kind::geglu_erf;
    case mlp_kind::geglu_quick: return glu_kind::geglu_quick;
    case mlp_kind::swiglu_oai: return glu_kind::swiglu_oai;
    case mlp_kind::swiglu_clamp: return glu_kind::swiglu_clamp;
    default: return glu_kind::swiglu;
    }
}

/// \brief A gated MLP's hidden element from its gate and up pre-activations: `glu(mlp_glu_kind(C.mlp), g, u)` with the
/// configuration's `glu_alpha` and `glu_limit`; for SwiGLU `silu(g) · u`, as always.
/// \tparam C A gated configuration.
/// \tparam T A `math::math_number`.
/// \param g `gate x + b`.
/// \param u `up x + b`.
/// \return The hidden element.
template <transformer_config C, math::math_number T>
constexpr T mlp_glu(T g, T u)
{
    return glu(mlp_glu_kind(C.mlp), g, u, glu_parameters<T>{glu_alpha_v<C, T>, glu_limit_v<C, T>});
}

/// \brief The backward pass of `mlp_glu`.
/// \tparam C A gated configuration.
/// \tparam T A `math::math_number`.
/// \param g `gate x + b`.
/// \param u `up x + b`.
/// \param dh The gradient in the hidden element.
/// \return The gradients in `g` and `u`.
template <transformer_config C, math::math_number T>
constexpr glu_gradient<T> mlp_glu_backward(T g, T u, T dh)
{
    return glu_backward(mlp_glu_kind(C.mlp), g, u, dh, glu_parameters<T>{glu_alpha_v<C, T>, glu_limit_v<C, T>});
}

namespace detail::mlp
{
    using ::snicholls::ml::linear;   // the function, not namespace detail::linear; still found by ADL for other storages

    // The forward pass, saving the pre-activations into non-empty spans
    template <transformer_config C, typename W, typename T, typename X, typename Y>
    constexpr void forward(const mlp_weights<W, T>& w, const X& x, Y&& y, const mlp_activations<T>& save)
    {
        static_assert(valid(C), "snicholls::ml::mlp_forward: the transformer_config is not valid");
        assert(w.up.rows == C.d_ff && w.up.cols == C.d_model && w.down.rows == C.d_model && w.down.cols == C.d_ff);
        assert((save.up.empty() || save.up.size() == C.d_ff) && (save.gate.empty() || save.gate.size() == C.d_ff));
        std::array<T, C.d_ff> h{};
        const auto in = prepare_input<C.d_model>(w.up, x);
        linear(w.up, in, h, w.up_bias);
        if (!save.up.empty())
            std::copy(h.begin(), h.end(), save.up.begin());
        if constexpr (gated(C.mlp)) {
            assert(w.gate.rows == C.d_ff && w.gate.cols == C.d_model);
            std::array<T, C.d_ff> g{};
            linear(w.gate, in, g, w.gate_bias);
            if (!save.gate.empty())
                std::copy(g.begin(), g.end(), save.gate.begin());
            for (std::size_t i = 0; i < C.d_ff; ++i)
                h[i] = mlp_glu<C>(g[i], h[i]);
        } else {
            for (T& v : h)
                v = mlp_activation<C.mlp>(v);
        }
        linear(w.down, prepare_input<C.d_ff>(w.down, h), y, w.down_bias);
    }
} // namespace detail::mlp

/// \brief The MLP for one token: `y = down(act(up x + b) ) + b`, or a gated unit's `down(mlp_glu(gate x + b, up x + b)) + b`.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The input, `d_model`.
/// \param y The output, `d_model`; may not alias `x`.
template <transformer_config C, weight_matrix W, math::math_number T, row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Y>, T>
constexpr void mlp_forward(const mlp_weights<W, T>& w, const X& x, Y&& y)
{
    detail::mlp::forward<C>(w, x, y, mlp_activations<T>{});
}

/// \brief The MLP for one token, saving its pre-activations for `mlp_backward`: the same output, to the bit.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The input, `d_model`.
/// \param y The output, `d_model`; may not alias `x`.
/// \param save Receives `up x + b` and, for a gated kind, `gate x + b`.
template <transformer_config C, weight_matrix W, math::math_number T, row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Y>, T>
constexpr void mlp_forward(const mlp_weights<W, T>& w, const X& x, Y&& y, const mlp_activations<T>& save)
{
    assert(save.up.size() == C.d_ff && (!gated(C.mlp) || save.gate.size() == C.d_ff));
    detail::mlp::forward<C>(w, x, y, save);
}

/// \brief The backward pass of `mlp_forward`, from its saved pre-activations.
///
/// The hidden row is recomputed from them (`d_ff` operations, no buffer saved for it); then `down`'s backward
/// gives its gradient, the activation's derivative (or the gated unit's product rule) takes it through, and `up`'s and
/// `gate`'s backward give the input's. Only dense weights train.
/// \tparam C The configuration.
/// \param w The weights the forward pass used.
/// \param x The input it read, `d_model`.
/// \param saved What it saved.
/// \param dy The gradient in its output, `d_model`.
/// \param dx Added into, `d_model`; or empty to skip it.
/// \param grads Added into.
template <transformer_config C, typename S, math::math_number T, row X, row DY, mutable_row DX>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<DY>, T> && std::same_as<row_value_t<DX>, T>
constexpr void mlp_backward(const mlp_weights<dense_weights<S>, T>& w, const X& x, const mlp_activations<T>& saved,
                            const DY& dy, DX&& dx, const mlp_gradients<T>& grads)
{
    static_assert(valid(C), "snicholls::ml::mlp_backward: the transformer_config is not valid");
    constexpr bool swiglu = gated(C.mlp);
    assert(saved.up.size() == C.d_ff && (!swiglu || saved.gate.size() == C.d_ff));
    std::array<T, C.d_ff> h{}, dh{};
    for (std::size_t i = 0; i < C.d_ff; ++i)
        if constexpr (swiglu)
            h[i] = mlp_glu<C>(saved.gate[i], saved.up[i]);
        else
            h[i] = mlp_activation<C.mlp>(saved.up[i]);
    linear_backward(w.down, h, dy, dh, grads.down, grads.down_bias);
    if constexpr (swiglu) {
        std::array<T, C.d_ff> dgate{};
        for (std::size_t i = 0; i < C.d_ff; ++i) {
            const auto d = mlp_glu_backward<C>(saved.gate[i], saved.up[i], dh[i]);
            dgate[i] = d.gate;
            dh[i] = d.up;
        }
        linear_backward(w.gate, x, dgate, dx, grads.gate, grads.gate_bias);
    } else {
        for (std::size_t i = 0; i < C.d_ff; ++i)
            dh[i] = dh[i] * mlp_activation_derivative<C.mlp>(saved.up[i]);
    }
    linear_backward(w.up, x, dh, dx, grads.up, grads.up_bias);
}

/// \brief The MLP's activation over a sequence, on the backend (TRAINING.md T9): `hidden = act(up)`, or a gated
/// kind's `mlp_glu(gate, up)`. The overload for any backend runs on the host; a device library overloads it for its own.
/// \tparam C The configuration.
/// \param saved `up` (and `gate` for a gated kind) read, `hidden` written, `n · d_ff` each.
template <transformer_config C, typename B, math::math_number T>
constexpr void mlp_activate_rows(const B&, const mlp_activations<T>& saved)
{
    const std::size_t m = saved.up.size();
    if constexpr (gated(C.mlp)) {
        for (std::size_t i = 0; i < m; ++i)
            saved.hidden[i] = mlp_glu<C>(saved.gate[i], saved.up[i]);
    } else {
        for (std::size_t i = 0; i < m; ++i)
            saved.hidden[i] = mlp_activation<C.mlp>(saved.up[i]);
    }
}

/// \brief The backward pass of `mlp_activate_rows`, on the backend: the gradient in `hidden` (`dh`) becomes the
/// gradient in `up`, in place, and for a gated kind also writes the gradient in `gate`.
/// \tparam C The configuration.
/// \param saved What the forward pass saved.
/// \param dh The gradient in the hidden rows, replaced by the gradient in `up`.
/// \param dgate Receives the gradient in `gate`, gated kinds only (else empty).
template <transformer_config C, typename B, math::math_number T>
constexpr void mlp_activate_rows_backward(const B&, const mlp_activations<T>& saved, std::span<T> dh, std::span<T> dgate)
{
    const std::size_t m = dh.size();
    if constexpr (gated(C.mlp)) {
        for (std::size_t i = 0; i < m; ++i) {
            const auto d = mlp_glu_backward<C>(saved.gate[i], saved.up[i], dh[i]);
            dgate[i] = d.gate;
            dh[i] = d.up;
        }
    } else {
        for (std::size_t i = 0; i < m; ++i)
            dh[i] = dh[i] * mlp_activation_derivative<C.mlp>(saved.up[i]);
        (void)dgate;
    }
}

/// \brief The MLP over a sequence of `n` tokens, its products through a `linalg` backend, saving what
/// `mlp_backward_sequence` needs.
///
/// The batched form of `mlp_forward` for dense weights in the sequence's type: `U = X Wupᵀ` (and a gated kind's gate), the
/// activation, `Y = H Wdownᵀ`, each product one `gemm` (`linear_sequence`). The hidden rows are saved, so the backward
/// does not recompute them.
/// \tparam C The configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights.
/// \param xs `n` inputs of `d_model`, end to end.
/// \param ys Receives `n` outputs of `d_model`; may not alias `xs`.
/// \param save Receives the pre-activations and the hidden rows, `n · d_ff` each (the gate for a gated kind only).
template <transformer_config C, typename B, math::math_number T>
    requires linalg::blas_backend<B, T>
constexpr void mlp_forward_sequence(const B& backend, const mlp_weights<dense_weights<T>, T>& w, std::type_identity_t<std::span<const T>> xs,
                                    std::type_identity_t<std::span<T>> ys, const mlp_activations<T>& save)
{
    static_assert(valid(C), "snicholls::ml::mlp_forward_sequence: the transformer_config is not valid");
    constexpr bool swiglu = gated(C.mlp);
    const std::size_t n = xs.size() / C.d_model;
    [[maybe_unused]] const std::size_t m = n * C.d_ff;
    assert(xs.size() == n * C.d_model && ys.size() == xs.size() && save.up.size() == m && save.hidden.size() == m
           && (!swiglu || save.gate.size() == m));
    linear_sequence(backend, w.up, xs, save.up, w.up_bias);
    if constexpr (swiglu)
        linear_sequence(backend, w.gate, xs, save.gate, w.gate_bias);
    mlp_activate_rows<C>(backend, save);
    linear_sequence(backend, w.down, std::span<const T>(save.hidden), ys, w.down_bias);
}

/// \brief The backward pass of `mlp_forward_sequence`, its products through the backend.
/// \tparam C The configuration.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights the forward pass used.
/// \param xs Its `n` inputs.
/// \param saved What it saved.
/// \param dys The gradients in its outputs, `n · d_model`.
/// \param dxs Added into, `n · d_model`; or empty to skip it.
/// \param grads Added into; its `scratch` (`n · d_ff`, twice for a gated kind) is written.
template <transformer_config C, typename B, math::math_number T>
    requires linalg::blas_backend<B, T>
constexpr void mlp_backward_sequence(const B& backend, const mlp_weights<dense_weights<T>, T>& w,
                                     std::type_identity_t<std::span<const T>> xs, const mlp_activations<T>& saved,
                                     std::type_identity_t<std::span<const T>> dys, std::type_identity_t<std::span<T>> dxs,
                                     const mlp_gradients<T>& grads)
{
    static_assert(valid(C), "snicholls::ml::mlp_backward_sequence: the transformer_config is not valid");
    constexpr bool swiglu = gated(C.mlp);
    const std::size_t n = xs.size() / C.d_model, m = n * C.d_ff;
    assert(dys.size() == xs.size() && saved.up.size() == m && saved.hidden.size() == m && (!swiglu || saved.gate.size() == m)
           && grads.scratch.size() >= (swiglu ? 2 : 1) * m);
    const auto dh = grads.scratch.first(m);
    const auto dgate = swiglu ? grads.scratch.subspan(m, m) : std::span<T>{};
    fill_rows(backend, dh, T(0));
    linear_sequence_backward(backend, w.down, std::span<const T>(saved.hidden), dys, dh, grads.down, grads.down_bias);
    mlp_activate_rows_backward<C>(backend, saved, dh, dgate);
    if constexpr (swiglu)
        linear_sequence_backward(backend, w.gate, xs, std::span<const T>(dgate), dxs, grads.gate, grads.gate_bias);
    linear_sequence_backward(backend, w.up, xs, std::span<const T>(dh), dxs, grads.up, grads.up_bias);
}

/// \brief The experts a token is routed to and their weights.
/// \ingroup core
/// \tparam T The number type.
/// \tparam K The experts per token.
template <math::math_number T, std::size_t K>
struct moe_route
{
    std::array<std::size_t, K> expert{};   ///< The chosen experts, highest router logit first.
    std::array<T, K> weight{};             ///< Their weights, summing to 1.
    std::size_t count = 0;                 ///< `min(K, experts)`.
};

/// \brief Routes a token to `K` experts: the `K` largest router logits, weighted by a softmax over those `K`.
/// \tparam K The experts per token, at least 1.
/// \param router_logits One logit per expert (`W_router x`), not empty.
/// \return The experts and their weights.
template <std::size_t K, row R>
    requires(K > 0)
constexpr moe_route<row_value_t<R>, K> moe_gate(const R& router_logits)
{
    using T = row_value_t<R>;
    const auto best = top_k<K>(router_logits);
    assert(best.count > 0);
    moe_route<T, K> out{};
    out.count = best.count;
    T sum = T(0);
    for (std::size_t i = 0; i < best.count; ++i) {
        out.expert[i] = best.index[i];
        out.weight[i] = math::exp(best.value[i] - best.value[0]);
        sum += out.weight[i];
    }
    for (std::size_t i = 0; i < best.count; ++i)
        out.weight[i] /= sum;
    return out;
}

/// \brief The backward pass of `moe_gate`: the gradient in the router logits from the gradient in the route's weights.
///
/// The weights are a softmax over the chosen experts' logits, so a chosen expert's logit gets
/// `w_i (dw_i - sum_k w_k dw_k)` and every other expert's 0 (the choice itself has no gradient).
/// \param route What `moe_gate` returned.
/// \param dweight The gradient in `route.weight`, `K` (the first `route.count` are read).
/// \param dlogits The gradient in the router logits, one per expert: written.
template <math::math_number T, std::size_t K, row DW, mutable_row Out>
    requires std::same_as<row_value_t<DW>, T> && std::same_as<row_value_t<Out>, T>
constexpr void moe_gate_backward(const moe_route<T, K>& route, const DW& dweight, Out&& dlogits)
{
    const auto dw = as_row(dweight);
    const auto g = as_mutable_row(dlogits);
    assert(dw.size() >= route.count && route.count <= g.size());
    T dot = T(0);
    for (std::size_t i = 0; i < route.count; ++i)
        dot += route.weight[i] * dw[i];
    std::fill(g.begin(), g.end(), T(0));
    for (std::size_t i = 0; i < route.count; ++i)
        g[route.expert[i]] = route.weight[i] * (dw[i] - dot);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_mlp_hpp
