//
//  activations.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` activations: sigmoid, softplus, logit, ReLU, ReLU², SiLU, GELU (exact, tanh and quick), ELU, leaky ReLU, hard sigmoid, hard swish, xIELU, step, sign and the logit soft cap, each with its derivative, all `constexpr`.
/// \ingroup core
///
/// **Overview.** The element-wise functions of a neural network, written once for every floating-point type and
/// usable at compile time (a lookup table of GELU for an 8-bit format), at run time on the CPU, and from a GPU
/// kernel. Each comes with its derivative for the backward pass. Where the output of the forward pass is what
/// backpropagation keeps (sigmoid, tanh), the derivative also has a form that takes the output. Every function
/// is computed in the form that cannot overflow or cancel:
///
/// - `sigmoid(x)` is `1 / (1 + e^-x)` for `x >= 0` and `e^x / (1 + e^x)` below, so neither form overflows.
/// - `softplus(x) = max(x, 0) + log1p(e^-|x|)` and `log_sigmoid(x) = -softplus(-x)`, so a loss built on them never
///   takes the log of an underflowed probability.
/// - The derivatives of sigmoid, SiLU and GELU-tanh take `1 - sigmoid(z)` as `sigmoid(-z)`, from the same `e^-|z|`, rather
///   than subtracting a sigmoid near 1 from 1, which cancels for large `z` (and is then scaled by `x` in SiLU and
///   GELU-tanh: 11.5 and 43.7 units in the last place in double, 7.2 and 16.5 in float, before 0.14.0; under 2 in float since).
/// - The activations of vision and audio encoders and of newer language models, as ggml computes them
///   (`ggml/src/ggml-cpu/unary-ops.cpp` and `vec.h`, llama.cpp 39039aa): `elu`, `leaky_relu`, `hardsigmoid`, `hardswish`,
///   `gelu_quick` (`x · sigmoid(1.702 x)`, CLIP), `xielu` (Apertus, with the coefficients after their softplus, as a GGUF
///   file stores them), `step` and `sign`. Each has a derivative; at a kink it takes PyTorch's convention. The gated
///   forms (GEGLU, SwiGLU and gpt-oss's clamped SwiGLU) are in `ml/glu.hpp`.
/// - `gelu(x) = x Φ(x)` uses `erfc` for the normal CDF, accurate in the left tail; `gelu_tanh` (GPT-2's
///   approximation) is computed as `x · sigmoid(2u)`, which is identical to `0.5 x (1 + tanh u)` and does not
///   cancel for large negative `x`.
///
/// \snippet ml/activations_demo.cpp activations
///
/// \snippet ml/activations_demo.cpp activations_encoders
///
/// **Relationships.** Built on `math/constexpr_math.hpp` (`exp`, `expm1`, `log1p`, `erfc`, `tanh`). Used with
/// `ml/softmax.hpp` and `ml/norm.hpp`. Group: \ref core.
///
/// **Implementations.** One each. A type of the caller's works through `math_limits`, as in `snicholls::math`.
///
/// **Guarantees.** Every function is `constexpr` and pure; NaN propagates. At run time the results are those of
/// the `<cmath>` functions they call; at compile time they agree to a few units in the last place.
///
/// **Errors.** None: a domain error (`logit` outside [0, 1]) is NaN.
///
/// **Performance.** Benchmarked against the same formulas written by hand (`benchmarks/ml/activations_benchmark.cpp`,
/// `BENCHMARKS.md`).
///
/// **Tests and demos.** `tests/ml/activations_test.cpp`, `demos/ml/activations_demo.cpp`.
///
/// **Dependencies.** `<limits>`, `<type_traits>`, `snicholls/core/math/constexpr_math.hpp`.
///
/// **Known issues.** `relu` and its derivative at exactly 0 take 0, as PyTorch does. ggml computes `gelu_quick` as
/// `x / (1 + e^(-1.702 x))`, and through a 16-bit table for `float16` inputs; here it is `x · sigmoid(1.702 x)` in the
/// argument's type, which agrees to a few units in the last place and never overflows.
///
/// **Status.** Experimental, new in 0.6.0; `elu`, `leaky_relu`, `hardsigmoid`, `hardswish`, `gelu_quick`, `xielu`, `step`
/// and `sign` new in 0.17.0.

#ifndef snicholls_core_ml_activations_hpp
#define snicholls_core_ml_activations_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <limits>
#include <type_traits>

namespace snicholls::ml
{

namespace detail::activations
{
    // sigmoid(v) and sigmoid(-v) = 1 - sigmoid(v), both from one e^-|v|, so neither cancels.
    template <typename T>
    struct sigmoid_pair
    {
        T s;
        T one_minus_s;
    };

    template <typename T>
    constexpr sigmoid_pair<T> sigmoid_both(T v)
    {
        const T e = math::exp(-math::abs(v));
        const T r = T(1) / (T(1) + e);
        if (v >= T(0))
            return {r, e * r};
        return {e * r, r};
    }
} // namespace detail::activations

/// \brief The logistic function `1 / (1 + e^-x)`, computed so that it never overflows.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return A value in [0, 1].
template <math::math_argument X>
constexpr math::real_t<X> sigmoid(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    if (v >= T(0))
        return T(1) / (T(1) + math::exp(-v));
    const T e = math::exp(v);
    return e / (T(1) + e);
}

/// \brief The derivative of `sigmoid` from its output: `s (1 - s)`, which is what a backward pass has kept.
/// \tparam T A `math::math_number`.
/// \param s `sigmoid(x)`.
/// \return `d sigmoid / dx`.
template <math::math_number T>
constexpr T sigmoid_derivative_from_output(T s)
{
    return s * (T(1) - s);
}

/// \brief The derivative of `sigmoid` at `x`: `sigmoid(x) sigmoid(-x)`, which keeps its digits in both tails.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return `d sigmoid / dx`.
template <math::math_argument X>
constexpr math::real_t<X> sigmoid_derivative(X x)
{
    using T = math::real_t<X>;
    const auto [s, one_minus_s] = detail::activations::sigmoid_both(static_cast<T>(x));
    return s * one_minus_s;
}

/// \brief `log(1 + e^x)`, a smooth ReLU, computed as `max(x, 0) + log1p(e^-|x|)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result; its derivative is `sigmoid(x)`.
template <math::math_argument X>
constexpr math::real_t<X> softplus(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    if (math::is_nan(v))
        return v;
    const T positive = v > T(0) ? v : T(0);
    return positive + math::log1p(math::exp(-math::abs(v)));
}

/// \brief `log(sigmoid(x))`, as `-softplus(-x)`: finite wherever the result is.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result; its derivative is `sigmoid(-x)`.
template <math::math_argument X>
constexpr math::real_t<X> log_sigmoid(X x)
{
    using T = math::real_t<X>;
    return -softplus(-static_cast<T>(x));
}

/// \brief The inverse of `sigmoid`, `log(p / (1 - p))`, as `log(p) - log1p(-p)`.
/// \tparam X An integer or a `math::math_number`.
/// \param p A probability.
/// \return The log-odds: -infinity at 0, infinity at 1, NaN outside [0, 1].
template <math::math_argument X>
constexpr math::real_t<X> logit(X p)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(p);
    if (!(v >= T(0) && v <= T(1)))   // NaN, or outside [0, 1]: no arithmetic on a NaN, which a constant expression refuses
        return std::numeric_limits<T>::quiet_NaN();
    return math::log(v) - math::log1p(-v);
}

/// \brief The derivative of `tanh` from its output: `1 - t²`.
/// \tparam T A `math::math_number`.
/// \param t `tanh(x)`.
/// \return `d tanh / dx`.
template <math::math_number T>
constexpr T tanh_derivative_from_output(T t)
{
    return T(1) - t * t;
}

/// \brief `max(x, 0)`; NaN propagates.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> relu(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v < T(0) ? T(0) : v;
}

/// \brief The derivative of `relu`: 1 above 0, else 0.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> relu_derivative(X x)
{
    using T = math::real_t<X>;
    return static_cast<T>(x) > T(0) ? T(1) : T(0);
}

/// \brief `relu(x)²`, the MLP activation of the Primer and modded-nanoGPT models.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> relu_squared(X x)
{
    const auto r = relu(x);
    return r * r;
}

/// \brief The derivative of `relu_squared`: `2 relu(x)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> relu_squared_derivative(X x)
{
    using T = math::real_t<X>;
    return T(2) * relu(x);
}

/// \brief SiLU (swish), `x · sigmoid(x)`, the activation of LLaMA's gated MLP.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> silu(X x)
{
    using T = math::real_t<X>;
    return static_cast<T>(x) * sigmoid(x);
}

/// \brief The derivative of `silu`: `s (1 + x (1 - s))` with `s = sigmoid(x)` and `1 - s` taken as `sigmoid(-x)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> silu_derivative(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    const auto [s, one_minus_s] = detail::activations::sigmoid_both(v);
    return s * (T(1) + v * one_minus_s);
}

/// \brief GELU as defined, `x Φ(x)`, with the normal CDF from `erfc` so the left tail keeps its digits.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> gelu(X x)
{
    using T = math::real_t<X>;
    using L = math::detail::limits<T>;
    const T v = static_cast<T>(x);
    return v * (T(0.5) * math::erfc(-v * T(L::inv_sqrt2)));
}

/// \brief The derivative of `gelu`: `Φ(x) + x φ(x)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> gelu_derivative(X x)
{
    using T = math::real_t<X>;
    using L = math::detail::limits<T>;
    const T v = static_cast<T>(x);
    const T cdf = T(0.5) * math::erfc(-v * T(L::inv_sqrt2));
    const T pdf = math::exp(T(-0.5) * v * v) / T(L::sqrt_2pi);
    return cdf + v * pdf;
}

namespace detail::activations
{
    // u = sqrt(2/pi) (x + 0.044715 x^3), GPT-2's approximation of GELU; returns 2u.
    template <typename T>
    constexpr T gelu_tanh_2u(T v)
    {
        using L = math::detail::limits<T>;
        const T sqrt_2_over_pi = T(2) / T(L::sqrt_2pi);
        return T(2) * sqrt_2_over_pi * (v + T(0.044715) * v * v * v);
    }
} // namespace detail::activations

/// \brief GELU's tanh approximation, `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))` (GPT-2, llm.c), as `x · sigmoid(2u)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> gelu_tanh(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v * sigmoid(detail::activations::gelu_tanh_2u(v));
}

/// \brief The derivative of `gelu_tanh`: `s + 2 x s (1 - s) √(2/π) (1 + 3 · 0.044715 x²)` with `s = sigmoid(2u)` and `1 - s` taken as `sigmoid(-2u)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> gelu_tanh_derivative(X x)
{
    using T = math::real_t<X>;
    using L = math::detail::limits<T>;
    const T v = static_cast<T>(x);
    const auto [s, one_minus_s] = detail::activations::sigmoid_both(detail::activations::gelu_tanh_2u(v));
    const T du = T(2) / T(L::sqrt_2pi) * (T(1) + T(3) * T(0.044715) * v * v);
    return s + T(2) * v * s * one_minus_s * du;
}

/// \brief The logit soft cap, `cap · tanh(x / cap)` (Gemma 2, nanochat): smooth, near `x` for small `x`, never beyond `cap`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The logit.
/// \param cap The cap, greater than 0.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> softcap(X x, std::type_identity_t<math::real_t<X>> cap)
{
    using T = math::real_t<X>;
    return cap * math::tanh(static_cast<T>(x) / cap);
}

/// \brief The derivative of `softcap` in `x`: `1 - tanh(x / cap)²`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The logit.
/// \param cap The cap, greater than 0.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> softcap_derivative(X x, std::type_identity_t<math::real_t<X>> cap)
{
    using T = math::real_t<X>;
    return tanh_derivative_from_output(math::tanh(static_cast<T>(x) / cap));
}

/// \brief ELU, `x` above 0 and `alpha (e^x - 1)` at or below (ggml's `elu` with `alpha` 1).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \param alpha The scale of the negative branch.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> elu(X x, std::type_identity_t<math::real_t<X>> alpha = 1)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v > T(0) ? v : alpha * math::expm1(v);
}

/// \brief The derivative of `elu`: 1 above 0, `alpha e^x` at or below.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \param alpha The scale of the negative branch.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> elu_derivative(X x, std::type_identity_t<math::real_t<X>> alpha = 1)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v > T(0) ? T(1) : alpha * math::exp(v);
}

/// \brief Leaky ReLU, `max(x, 0) + slope · min(x, 0)` (ggml's `leaky_relu`, written as it is there).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \param slope The slope below 0 (PyTorch's `negative_slope`, 0.01 by default).
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> leaky_relu(X x, std::type_identity_t<math::real_t<X>> slope = 0.01)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    const T positive = v > T(0) ? v : T(0), negative = v < T(0) ? v : T(0);
    return positive + slope * negative;
}

/// \brief The derivative of `leaky_relu`: 1 above 0, `slope` at or below (PyTorch's convention at 0).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \param slope The slope below 0.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> leaky_relu_derivative(X x, std::type_identity_t<math::real_t<X>> slope = 0.01)
{
    using T = math::real_t<X>;
    return static_cast<T>(x) > T(0) ? T(1) : slope;
}

/// \brief Hard sigmoid, `min(1, max(0, (x + 3) / 6))` (ggml, PyTorch, MobileNetV3).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return A value in [0, 1].
template <math::math_argument X>
constexpr math::real_t<X> hardsigmoid(X x)
{
    using T = math::real_t<X>;
    const T r = (static_cast<T>(x) + T(3)) / T(6);
    return r < T(0) ? T(0) : r > T(1) ? T(1) : r;
}

/// \brief The derivative of `hardsigmoid`: 1/6 strictly between -3 and 3, else 0 (PyTorch's convention at the corners).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> hardsigmoid_derivative(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v > T(-3) && v < T(3) ? T(1) / T(6) : T(0);
}

/// \brief Hard swish, `x · hardsigmoid(x)` (ggml, MobileNetV3).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> hardswish(X x)
{
    using T = math::real_t<X>;
    return static_cast<T>(x) * hardsigmoid(x);
}

/// \brief The derivative of `hardswish`: 0 at or below -3, 1 at or above 3, `(2x + 3) / 6` between (PyTorch's).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> hardswish_derivative(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v <= T(-3) ? T(0) : v >= T(3) ? T(1) : (T(2) * v + T(3)) / T(6);
}

/// \brief The coefficient of `gelu_quick`, 1.702 (Hendrycks and Gimpel; ggml's `GELU_QUICK_COEF` is its negative).
/// \tparam T A `math::math_number`.
template <math::math_number T>
inline constexpr T gelu_quick_coefficient = T(1.702);

/// \brief GELU's sigmoid approximation, `x · sigmoid(1.702 x)` (CLIP, ggml's `gelu_quick`).
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> gelu_quick(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v * sigmoid(gelu_quick_coefficient<T> * v);
}

/// \brief The derivative of `gelu_quick`: `s + 1.702 x s (1 - s)` with `s = sigmoid(1.702 x)` and `1 - s` taken as `sigmoid(-1.702 x)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> gelu_quick_derivative(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x), a = gelu_quick_coefficient<T>;
    const auto [s, one_minus_s] = detail::activations::sigmoid_both(a * v);
    return s + a * v * s * one_minus_s;
}

/// \brief The parameters of `xielu`, as a model stores them after their softplus (Apertus; ggml's `xielu` operation parameters).
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct xielu_parameters
{
    T alpha_p = T(0.8);   ///< The quadratic's coefficient above 0 (`softplus(α_p)` in the paper).
    T alpha_n = T(0.8);   ///< The negative branch's coefficient (`β + softplus(α_n)` in the paper).
    T beta = T(0.5);      ///< The linear term, on both sides.
    T eps = T(-1e-6);     ///< The negative branch's `expm1` argument is `min(x, eps)`.
};

/// \brief xIELU (Huang et al., 2025): `α_p x² + β x` above 0, `α_n (expm1(min(x, ε)) - x) + β x` at or below, as ggml's `op_xielu`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \param p The parameters.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> xielu(X x, const xielu_parameters<math::real_t<X>>& p)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    if (v > T(0))
        return p.alpha_p * v * v + p.beta * v;
    const T m = v < p.eps ? v : p.eps;
    return (math::expm1(m) - v) * p.alpha_n + p.beta * v;
}

/// \brief The derivative of `xielu`: `2 α_p x + β` above 0, `α_n (e^x [x < ε] - 1) + β` at or below.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \param p The parameters.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> xielu_derivative(X x, const xielu_parameters<math::real_t<X>>& p)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    if (v > T(0))
        return T(2) * p.alpha_p * v + p.beta;
    const T de = v < p.eps ? math::exp(v) : T(0);
    return (de - T(1)) * p.alpha_n + p.beta;
}

/// \brief The step function, 1 above 0 and 0 at or below (ggml's `step`); its derivative is 0 almost everywhere.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return 0 or 1.
template <math::math_argument X>
constexpr math::real_t<X> step(X x)
{
    using T = math::real_t<X>;
    return static_cast<T>(x) > T(0) ? T(1) : T(0);
}

/// \brief The sign, 1, -1 or 0 (ggml's `sgn`: a NaN and both zeros give 0); its derivative is 0 almost everywhere.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return -1, 0 or 1.
template <math::math_argument X>
constexpr math::real_t<X> sign(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    return v > T(0) ? T(1) : v < T(0) ? T(-1) : T(0);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_activations_hpp
