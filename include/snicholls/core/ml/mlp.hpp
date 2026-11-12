//
//  mlp.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::mlp_forward`: a transformer's MLP for one token, `down(act(up x))` or SwiGLU's `down(silu(gate x) · up x)`, as the configuration says.
/// \ingroup core
///
/// **Overview.** The feed-forward half of a transformer block. `transformer_config::mlp` chooses the form: GELU
/// (exact or tanh, GPT-2), ReLU² (nanochat) or SwiGLU (Llama). The weights are `mlp_weights`, views over any
/// `weight_matrix` storage, so the same call runs on float, bfloat16 or quantised weights. The input is prepared
/// once for `up` and `gate`.
///
/// For a mixture of experts, `moe_gate<K>(router_logits)` picks the `K` experts a token goes to and their weights
/// (a softmax over the `K` largest router logits, as Mixtral); the token's output is the weighted sum of those
/// experts' `mlp_forward`s, each expert's weights a `mlp_weights` of its own.
///
/// \snippet ml/mlp_demo.cpp mlp
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
/// **Performance.** Benchmarked through `ml/transformer_block.hpp`'s decode step.
///
/// **Tests and demos.** `tests/ml/mlp_test.cpp`, `demos/ml/mlp_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<span>`, `ml/activations.hpp`, `ml/linear.hpp`, `ml/row.hpp`,
/// `ml/sampling.hpp` (`top_k`),
/// `ml/transformer_config.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.8.0.

#ifndef snicholls_core_ml_mlp_hpp
#define snicholls_core_ml_mlp_hpp

#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/ml/sampling.hpp>
#include <snicholls/core/ml/transformer_config.hpp>

#include <array>
#include <cassert>
#include <span>

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
    W gate{};                           ///< `d_ff x d_model`, SwiGLU only.
    W down{};                           ///< `d_model x d_ff`.
    std::span<const T> up_bias{};       ///< `d_ff`, or empty.
    std::span<const T> gate_bias{};     ///< `d_ff`, or empty; SwiGLU only.
    std::span<const T> down_bias{};     ///< `d_model`, or empty.
};

/// \brief The activation of a configuration's MLP (not SwiGLU, whose gate is a second product).
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
    }
    return x;
}

/// \brief The MLP for one token: `y = down(act(up x + b) ) + b`, or SwiGLU's `down(silu(gate x) · up x)`.
/// \tparam C The configuration.
/// \param w The weights.
/// \param x The input, `d_model`.
/// \param y The output, `d_model`; may not alias `x`.
template <transformer_config C, weight_matrix W, math::math_number T, row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, T> && std::same_as<row_value_t<Y>, T>
constexpr void mlp_forward(const mlp_weights<W, T>& w, const X& x, Y&& y)
{
    static_assert(valid(C), "snicholls::ml::mlp_forward: the transformer_config is not valid");
    assert(w.up.rows == C.d_ff && w.up.cols == C.d_model && w.down.rows == C.d_model && w.down.cols == C.d_ff);
    std::array<T, C.d_ff> h{};
    const auto in = prepare_input<C.d_model>(w.up, x);
    linear(w.up, in, h, w.up_bias);
    if constexpr (C.mlp == mlp_kind::swiglu) {
        assert(w.gate.rows == C.d_ff && w.gate.cols == C.d_model);
        std::array<T, C.d_ff> g{};
        linear(w.gate, in, g, w.gate_bias);
        for (std::size_t i = 0; i < C.d_ff; ++i)
            h[i] = silu(g[i]) * h[i];
    } else {
        for (T& v : h)
            v = mlp_activation<C.mlp>(v);
    }
    linear(w.down, prepare_input<C.d_ff>(w.down, h), y, w.down_bias);
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

} // namespace snicholls::ml

#endif // snicholls_core_ml_mlp_hpp
