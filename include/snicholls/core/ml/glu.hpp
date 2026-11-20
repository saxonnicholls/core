//
//  glu.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` gated linear units: ReGLU, GEGLU (tanh, erf and quick), SwiGLU, gpt-oss's clamped SwiGLU and SwiGLU with a clamp, for one element and over rows, forward and backward, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A gated MLP computes `down(act(gate x) · up x)` (Shazeer, 2020). The gate and the up projection are
/// two rows of the same width; a GLU combines them element by element. ggml's GLU operations (`ggml_reglu`,
/// `ggml_geglu`, `ggml_geglu_erf`, `ggml_geglu_quick`, `ggml_swiglu`, `ggml_swiglu_oai`, `ggml_swiglu_clamp`;
/// `ggml/src/ggml-cpu/ops.cpp` and `vec.h`, llama.cpp 39039aa) are reproduced:
///
/// | `glu_kind` | `h(g, u)` | Used by |
/// |---|---|---|
/// | `reglu` | `relu(g) · u` | T5 variants |
/// | `geglu` | `gelu_tanh(g) · u` (ggml's `ggml_gelu_f32` is the tanh form) | Gemma |
/// | `geglu_erf` | `gelu(g) · u` | the exact GELU |
/// | `geglu_quick` | `gelu_quick(g) · u` | |
/// | `swiglu` | `silu(g) · u` | Llama, Qwen, DeepSeek |
/// | `swiglu_oai` | `x · sigmoid(α x) · (y + 1)`, `x = min(g, limit)`, `y = clamp(u, -limit, limit)` | gpt-oss (`α` 1.702, `limit` 7: `src/llama-graph.cpp`, `LLM_FFN_SWIGLU_OAI_MOE`) |
/// | `swiglu_clamp` | `silu(x) · y` with the same clamps | DeepSeek V4, GLM-5 (`ggml_swiglu_clamp`) |
///
/// - `glu(kind, g, u, params)` and `glu_backward(kind, g, u, dh, params)` for one element; the backward returns the
///   gradients in `g` and `u` as `dh · ∂h/∂g` and `dh · ∂h/∂u`, multiplied in the order the MLP's SwiGLU always used,
///   so `mlp_forward` and `mlp_backward` keep their bits.
/// - `glu_rows` / `glu_rows_backward` over a gate row and an up row; `glu_split` over one row of `2n` holding both
///   halves, gate first unless `swapped`, as ggml's single-tensor form.
/// - `mlp_kind`'s gated kinds (`ml/transformer_config.hpp`) map to these, so the MLP, the mixture of experts and the
///   blocks run every one of them.
///
/// \snippet ml/glu_demo.cpp glu
///
/// **Relationships.** Built on `ml/activations.hpp`; used by `ml/mlp.hpp` and `ml/moe.hpp` through `mlp_kind`.
/// Group: \ref core.
///
/// **Implementations.** One each; the kind is a run-time value or, in the MLP, a compile-time one (`switch` over an
/// enumeration, never a chain over types).
///
/// **Guarantees.** `constexpr`, no allocation, linear in the row. The output may be the gate or the up row. At a
/// clamp's edge the gradient passes (PyTorch's `clamp`: the bounds are inside).
///
/// **Errors.** Preconditions, asserted: the rows have one size (twice it for `glu_split`); `limit` is positive for the
/// clamped kinds.
///
/// **Performance.** Against the same loops written by hand (`benchmarks/ml/glu_benchmark.cpp`, rows of 14,336): SwiGLU
/// 113 million elements per second against 114, gpt-oss's SwiGLU 106 against 103 (Xeon W-3245, Release, Apple clang 17,
/// 2026-11-19). The kind is resolved once per row, not per element.
///
/// **Tests and demos.** `tests/ml/glu_test.cpp`, `demos/ml/glu_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<span>`, `<type_traits>`, `ml/activations.hpp`, `ml/row.hpp`.
///
/// **Known issues.** ggml computes `swiglu_oai` as `x / (1 + e^(-α x))` and GEGLU in `float` through a `float16` table
/// between -10 and 10 where `GGML_GELU_FP16` is defined; here each is the activation of `ml/activations.hpp`, which
/// agrees to a few units in the last place (stated in the tests). A NaN gate is not clamped (`min` keeps it).
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_glu_hpp
#define snicholls_core_ml_glu_hpp

#include <snicholls/core/ml/activations.hpp>
#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief How a gated linear unit combines a gate `g` and an up projection `u`.
/// \ingroup core
enum class glu_kind
{
    reglu,          ///< `relu(g) · u`.
    geglu,          ///< `gelu_tanh(g) · u`, ggml's GEGLU.
    geglu_erf,      ///< `gelu(g) · u`, the exact GELU.
    geglu_quick,    ///< `gelu_quick(g) · u`.
    swiglu,         ///< `silu(g) · u`.
    swiglu_oai,     ///< gpt-oss: `x · sigmoid(α x) · (y + 1)`, `x = min(g, limit)`, `y = clamp(u, -limit, limit)`.
    swiglu_clamp,   ///< `silu(min(g, limit)) · clamp(u, -limit, limit)`.
};

/// \brief The constants of the clamped kinds; the defaults are gpt-oss's.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct glu_parameters
{
    T alpha = T(1.702);   ///< `swiglu_oai`'s sigmoid scale.
    T limit = T(7);       ///< The clamp of the gate (above) and of the up projection (both sides).
};

/// \brief The gradients of a GLU in its two inputs.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct glu_gradient
{
    T gate;   ///< `dh · ∂h/∂g`.
    T up;     ///< `dh · ∂h/∂u`.
};

namespace detail::glu
{
    template <typename T>
    constexpr T min(T a, T b) { return b < a ? b : a; }

    template <typename T>
    constexpr T clamp(T v, T lo, T hi) { return v < lo ? lo : hi < v ? hi : v; }

    template <typename T>
    constexpr T inside(T v, T lo, T hi) { return v >= lo && v <= hi ? T(1) : T(0); }

    // Calls f with the kind as a constant, so a row's loop is compiled once per kind with the switch folded away
    template <typename F>
    constexpr void dispatch(glu_kind k, F&& f)
    {
        using K = glu_kind;
        switch (k) {
        case K::reglu: f(std::integral_constant<K, K::reglu>{}); return;
        case K::geglu: f(std::integral_constant<K, K::geglu>{}); return;
        case K::geglu_erf: f(std::integral_constant<K, K::geglu_erf>{}); return;
        case K::geglu_quick: f(std::integral_constant<K, K::geglu_quick>{}); return;
        case K::swiglu: f(std::integral_constant<K, K::swiglu>{}); return;
        case K::swiglu_oai: f(std::integral_constant<K, K::swiglu_oai>{}); return;
        case K::swiglu_clamp: f(std::integral_constant<K, K::swiglu_clamp>{}); return;
        }
    }
} // namespace detail::glu

/// \brief One element of a GLU.
/// \tparam T A `math::math_number`.
/// \param kind The form.
/// \param g The gate.
/// \param u The up projection.
/// \param p The clamped kinds' constants.
/// \return `h(g, u)`.
template <math::math_number T>
constexpr T glu(glu_kind kind, T g, T u, const glu_parameters<T>& p = {})
{
    switch (kind) {
    case glu_kind::reglu: return relu(g) * u;
    case glu_kind::geglu: return gelu_tanh(g) * u;
    case glu_kind::geglu_erf: return gelu(g) * u;
    case glu_kind::geglu_quick: return gelu_quick(g) * u;
    case glu_kind::swiglu: return silu(g) * u;
    case glu_kind::swiglu_oai: {
        const T x = detail::glu::min(g, p.limit), y = detail::glu::clamp(u, -p.limit, p.limit);
        const T out = x * sigmoid(p.alpha * x);
        return out * (y + T(1));
    }
    case glu_kind::swiglu_clamp: {
        const T x = detail::glu::min(g, p.limit), y = detail::glu::clamp(u, -p.limit, p.limit);
        return silu(x) * y;
    }
    }
    return T(0);
}

/// \brief The backward pass of one element of a GLU.
/// \tparam T A `math::math_number`.
/// \param kind The form.
/// \param g The gate, as in the forward pass.
/// \param u The up projection, as in the forward pass.
/// \param dh The gradient in the output.
/// \param p The clamped kinds' constants.
/// \return The gradients in `g` and `u`.
template <math::math_number T>
constexpr glu_gradient<T> glu_backward(glu_kind kind, T g, T u, T dh, const glu_parameters<T>& p = {})
{
    switch (kind) {
    case glu_kind::reglu: return {dh * u * relu_derivative(g), dh * relu(g)};
    case glu_kind::geglu: return {dh * u * gelu_tanh_derivative(g), dh * gelu_tanh(g)};
    case glu_kind::geglu_erf: return {dh * u * gelu_derivative(g), dh * gelu(g)};
    case glu_kind::geglu_quick: return {dh * u * gelu_quick_derivative(g), dh * gelu_quick(g)};
    case glu_kind::swiglu: return {dh * u * silu_derivative(g), dh * silu(g)};
    case glu_kind::swiglu_oai: {
        const T x = detail::glu::min(g, p.limit), y = detail::glu::clamp(u, -p.limit, p.limit);
        const T dx = g <= p.limit ? T(1) : T(0), dy = detail::glu::inside(u, -p.limit, p.limit);
        const auto [s, one_minus_s] = detail::activations::sigmoid_both(p.alpha * x);
        const T dact = s + p.alpha * x * s * one_minus_s;   // d/dx (x sigmoid(α x))
        return {dh * (y + T(1)) * dact * dx, dh * x * s * dy};
    }
    case glu_kind::swiglu_clamp: {
        const T x = detail::glu::min(g, p.limit), y = detail::glu::clamp(u, -p.limit, p.limit);
        const T dx = g <= p.limit ? T(1) : T(0), dy = detail::glu::inside(u, -p.limit, p.limit);
        return {dh * y * silu_derivative(x) * dx, dh * silu(x) * dy};
    }
    }
    return {T(0), T(0)};
}

/// \brief A GLU over rows: `out[i] = h(gate[i], up[i])`.
/// \param kind The form.
/// \param gate The gate row.
/// \param up The up row, the gate's size.
/// \param out Receives the result, the gate's size; may be `gate` or `up`.
/// \param p The clamped kinds' constants.
template <row G, row U, mutable_row Out>
    requires std::same_as<row_value_t<G>, row_value_t<U>> && std::same_as<row_value_t<G>, row_value_t<Out>>
constexpr void glu_rows(glu_kind kind, const G& gate, const U& up, Out&& out, const glu_parameters<row_value_t<G>>& p = {})
{
    const auto g = as_row(gate);
    const auto u = as_row(up);
    const auto y = as_mutable_row(out);
    assert(u.size() == g.size() && y.size() == g.size());
    detail::glu::dispatch(kind, [&](auto k) {
        for (std::size_t i = 0; i < g.size(); ++i)
            y[i] = glu(decltype(k)::value, g[i], u[i], p);
    });
}

/// \brief The backward pass of `glu_rows`: writes the gradients in the gate and the up rows.
/// \param kind The form.
/// \param gate The gate row of the forward pass.
/// \param up The up row of the forward pass.
/// \param dout The gradient in the output.
/// \param dgate Receives the gradient in the gate.
/// \param dup Receives the gradient in the up row.
/// \param p The clamped kinds' constants.
template <row G, row U, row D, mutable_row DG, mutable_row DU>
    requires std::same_as<row_value_t<G>, row_value_t<U>> && std::same_as<row_value_t<G>, row_value_t<D>>
             && std::same_as<row_value_t<G>, row_value_t<DG>> && std::same_as<row_value_t<G>, row_value_t<DU>>
constexpr void glu_rows_backward(glu_kind kind, const G& gate, const U& up, const D& dout, DG&& dgate, DU&& dup,
                                 const glu_parameters<row_value_t<G>>& p = {})
{
    const auto g = as_row(gate);
    const auto u = as_row(up);
    const auto d = as_row(dout);
    const auto dg = as_mutable_row(dgate);
    const auto du = as_mutable_row(dup);
    assert(u.size() == g.size() && d.size() == g.size() && dg.size() == g.size() && du.size() == g.size());
    detail::glu::dispatch(kind, [&](auto k) {
        for (std::size_t i = 0; i < g.size(); ++i) {
            const auto r = glu_backward(decltype(k)::value, g[i], u[i], d[i], p);
            dg[i] = r.gate;
            du[i] = r.up;
        }
    });
}

/// \brief A GLU over one row of `2n` holding both inputs, as ggml's single-tensor form: the gate is the first half and
/// the up projection the second, or the other way round when `swapped`.
/// \param kind The form.
/// \param x The row of `2n`.
/// \param out Receives `n` results.
/// \param swapped Whether the up projection comes first.
/// \param p The clamped kinds' constants.
template <row X, mutable_row Out>
    requires std::same_as<row_value_t<X>, row_value_t<Out>>
constexpr void glu_split(glu_kind kind, const X& x, Out&& out, bool swapped = false, const glu_parameters<row_value_t<X>>& p = {})
{
    const auto in = as_row(x);
    const auto y = as_mutable_row(out);
    const std::size_t n = y.size();
    assert(in.size() == 2 * n);
    glu_rows(kind, in.subspan(swapped ? n : 0, n), in.subspan(swapped ? 0 : n, n), y, p);
}

/// \brief The backward pass of `glu_split`: writes the gradient in the row of `2n`.
/// \param kind The form.
/// \param x The row of `2n` of the forward pass.
/// \param dout The gradient in the `n` results.
/// \param dx Receives the gradient in `x`, `2n`.
/// \param swapped Whether the up projection comes first.
/// \param p The clamped kinds' constants.
template <row X, row D, mutable_row DX>
    requires std::same_as<row_value_t<X>, row_value_t<D>> && std::same_as<row_value_t<X>, row_value_t<DX>>
constexpr void glu_split_backward(glu_kind kind, const X& x, const D& dout, DX&& dx, bool swapped = false,
                                  const glu_parameters<row_value_t<X>>& p = {})
{
    const auto in = as_row(x);
    const auto d = as_row(dout);
    const auto g = as_mutable_row(dx);
    const std::size_t n = d.size();
    assert(in.size() == 2 * n && g.size() == 2 * n);
    const std::size_t gate = swapped ? n : 0, up = swapped ? 0 : n;
    glu_rows_backward(kind, in.subspan(gate, n), in.subspan(up, n), d, g.subspan(gate, n), g.subspan(up, n), p);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_glu_hpp
