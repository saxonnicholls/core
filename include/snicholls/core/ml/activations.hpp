//
//  activations.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` activations: sigmoid, softplus, logit, ReLU, ReLU², SiLU, GELU (exact and tanh) and the logit soft cap, each with its derivative, all `constexpr`.
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
/// - `gelu(x) = x Φ(x)` uses `erfc` for the normal CDF, accurate in the left tail; `gelu_tanh` (GPT-2's
///   approximation) is computed as `x · sigmoid(2u)`, which is identical to `0.5 x (1 + tanh u)` and does not
///   cancel for large negative `x`.
///
/// \snippet ml/activations_demo.cpp activations
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
/// **Known issues.** `relu` and its derivative at exactly 0 take 0, as PyTorch does.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_ml_activations_hpp
#define snicholls_core_ml_activations_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <limits>
#include <type_traits>

namespace snicholls::ml
{

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

/// \brief The derivative of `sigmoid` at `x`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return `d sigmoid / dx`.
template <math::math_argument X>
constexpr math::real_t<X> sigmoid_derivative(X x)
{
    return sigmoid_derivative_from_output(sigmoid(x));
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

/// \brief The derivative of `silu`: `s (1 + x (1 - s))` with `s = sigmoid(x)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> silu_derivative(X x)
{
    using T = math::real_t<X>;
    const T v = static_cast<T>(x);
    const T s = sigmoid(v);
    return s * (T(1) + v * (T(1) - s));
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

/// \brief The derivative of `gelu_tanh`: `s + 2 x s (1 - s) √(2/π) (1 + 3 · 0.044715 x²)` with `s = sigmoid(2u)`.
/// \tparam X An integer or a `math::math_number`.
/// \param x The argument.
/// \return The result.
template <math::math_argument X>
constexpr math::real_t<X> gelu_tanh_derivative(X x)
{
    using T = math::real_t<X>;
    using L = math::detail::limits<T>;
    const T v = static_cast<T>(x);
    const T s = sigmoid(detail::activations::gelu_tanh_2u(v));
    const T du = T(2) / T(L::sqrt_2pi) * (T(1) + T(3) * T(0.044715) * v * v);
    return s + T(2) * v * s * (T(1) - s) * du;
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

} // namespace snicholls::ml

#endif // snicholls_core_ml_activations_hpp
