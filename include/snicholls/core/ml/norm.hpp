//
//  norm.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` normalisation: `layer_norm` and `rms_norm` over one row, forward and backward, with or without learned parameters.
/// \ingroup core
///
/// **Overview.** The per-token normalisations of a transformer. Each works on one row of `C` channels; a batch is
/// a loop over rows, which a caller spreads across threads (each row is independent).
///
/// - `layer_norm(x, weight, bias, out, eps)`: `out = (x - mean) · rstd · weight + bias`, with
///   `rstd = 1 / sqrt(var + eps)` and the biased variance, as GPT-2 and PyTorch compute it. The mean is subtracted
///   before squaring (two passes), so a row with a large mean keeps its variance's digits. Returns the mean and
///   `rstd`, which the backward pass needs.
/// - `rms_norm(x, weight, out, eps)`: `out = x · rstd · weight`, `rstd = 1 / sqrt(mean(x²) + eps)` (LLaMA,
///   nanochat). Returns `rstd`.
/// - Empty `weight` and `bias` mean none: nanochat's RMSNorm has no parameters.
/// - The backward passes write `dx` and add into `dweight` and `dbias`, which sum over the rows of a batch.
///
/// \snippet ml/norm_demo.cpp norm
///
/// **Relationships.** `ml/row.hpp` (the rows), `ml/activations.hpp`. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation, linear in the row. `out` and `dx` may be the input row.
/// `norm_stats` is trivially copyable.
///
/// **Errors.** Preconditions, asserted: every row has `x`'s size, or (`weight`, `bias`, `dweight`, `dbias`) is
/// empty. An empty `x` is a no-op that returns `rstd` 1/sqrt(eps).
///
/// **Performance.** Benchmarked against the same loops written by hand (`benchmarks/ml/norm_benchmark.cpp`).
///
/// **Tests and demos.** `tests/ml/norm_test.cpp`, `demos/ml/norm_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<concepts>`, `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`.
///
/// **Known issues.** Sums accumulate in the row's type, as llm.c does in `float`.
///
/// **Status.** Experimental, new in 0.6.0.

#ifndef snicholls_core_ml_norm_hpp
#define snicholls_core_ml_norm_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief What `layer_norm` returns for its backward pass: the row's mean and `1 / sqrt(var + eps)`.
/// \ingroup core
/// \tparam T The number type.
template <typename T>
struct norm_stats
{
    T mean;   ///< The mean of the row.
    T rstd;   ///< `1 / sqrt(variance + eps)`.
};

namespace detail::norm
{
    template <typename T>
    constexpr T param(std::span<const T> p, std::size_t i, T otherwise) { return p.empty() ? otherwise : p[i]; }

    template <typename T>
    constexpr bool fits(std::size_t size, std::size_t n) { return size == 0 || size == n; }
} // namespace detail::norm

/// \brief LayerNorm of one row: `out = (x - mean) · rstd · weight + bias`.
/// \tparam X A `row`.
/// \tparam W A `row` of the same number type (may be empty: no weight).
/// \tparam B A `row` of the same number type (may be empty: no bias).
/// \tparam Out A `mutable_row` of the same number type.
/// \param x The row.
/// \param weight The learned scale, one per channel, or empty.
/// \param bias The learned shift, one per channel, or empty.
/// \param out The normalised row: `x`'s size; may be `x`.
/// \param eps Added to the variance: 1e-5 by default, as GPT-2 and PyTorch.
/// \return The mean and `rstd`, for `layer_norm_backward`.
template <row X, row W, row B, mutable_row Out>
    requires std::same_as<row_value_t<X>, row_value_t<W>> && std::same_as<row_value_t<X>, row_value_t<B>>
             && std::same_as<row_value_t<X>, row_value_t<Out>>
constexpr norm_stats<row_value_t<X>> layer_norm(const X& x, const W& weight, const B& bias, Out&& out,
                                                std::type_identity_t<row_value_t<X>> eps = 1e-5)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto w = as_row(weight);
    const auto b = as_row(bias);
    const auto y = as_mutable_row(out);
    const std::size_t n = in.size();
    assert(y.size() == n && detail::norm::fits<T>(w.size(), n) && detail::norm::fits<T>(b.size(), n));
    T mean = T(0);
    for (const T v : in)
        mean += v;
    mean = n == 0 ? T(0) : mean / T(n);
    T var = T(0);
    for (const T v : in)
        var += (v - mean) * (v - mean);
    var = n == 0 ? T(0) : var / T(n);
    const T rstd = T(1) / math::sqrt(var + eps);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = (in[i] - mean) * rstd * detail::norm::param(w, i, T(1)) + detail::norm::param(b, i, T(0));
    return {mean, rstd};
}

/// \brief The backward pass of `layer_norm`: writes `dx`, adds into `dweight` and `dbias`.
///
/// With `n = (x - mean) · rstd` and `g = weight · dout`: `dx = rstd · (g - mean(g) - n · mean(g · n))`,
/// `dweight += dout · n`, `dbias += dout`.
/// \param dout The gradient in the output.
/// \param x The forward pass's input.
/// \param weight The forward pass's weight, or empty.
/// \param stats What the forward pass returned.
/// \param dx The gradient in the input: written; `x`'s size; may be `dout`.
/// \param dweight Accumulated into; `x`'s size, or empty.
/// \param dbias Accumulated into; `x`'s size, or empty.
template <row D, row X, row W, mutable_row DX, mutable_row DW, mutable_row DB>
    requires std::same_as<row_value_t<X>, row_value_t<D>> && std::same_as<row_value_t<X>, row_value_t<W>>
             && std::same_as<row_value_t<X>, row_value_t<DX>> && std::same_as<row_value_t<X>, row_value_t<DW>>
             && std::same_as<row_value_t<X>, row_value_t<DB>>
constexpr void layer_norm_backward(const D& dout, const X& x, const W& weight, norm_stats<row_value_t<X>> stats, DX&& dx,
                                   DW&& dweight, DB&& dbias)
{
    using T = row_value_t<X>;
    const auto d = as_row(dout);
    const auto in = as_row(x);
    const auto w = as_row(weight);
    const auto gx = as_mutable_row(dx);
    const auto gw = as_mutable_row(dweight);
    const auto gb = as_mutable_row(dbias);
    const std::size_t n = in.size();
    assert(d.size() == n && gx.size() == n && detail::norm::fits<T>(w.size(), n) && detail::norm::fits<T>(gw.size(), n)
           && detail::norm::fits<T>(gb.size(), n));
    if (n == 0)
        return;
    T mean_g = T(0), mean_gn = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        const T norm = (in[i] - stats.mean) * stats.rstd;
        const T g = detail::norm::param(w, i, T(1)) * d[i];
        mean_g += g;
        mean_gn += g * norm;
    }
    mean_g /= T(n);
    mean_gn /= T(n);
    for (std::size_t i = 0; i < n; ++i) {
        const T norm = (in[i] - stats.mean) * stats.rstd;
        const T g = detail::norm::param(w, i, T(1)) * d[i];
        if (!gw.empty())
            gw[i] += d[i] * norm;
        if (!gb.empty())
            gb[i] += d[i];
        gx[i] = stats.rstd * (g - mean_g - norm * mean_gn);
    }
}

/// \brief RMSNorm of one row: `out = x · rstd · weight`, `rstd = 1 / sqrt(mean(x²) + eps)`.
/// \tparam X A `row`.
/// \tparam W A `row` of the same number type (may be empty: no weight).
/// \tparam Out A `mutable_row` of the same number type.
/// \param x The row.
/// \param weight The learned scale, one per channel, or empty.
/// \param out The normalised row: `x`'s size; may be `x`.
/// \param eps Added to the mean square: the type's epsilon by default, as PyTorch's `rms_norm`.
/// \return `rstd`, for `rms_norm_backward`.
template <row X, row W, mutable_row Out>
    requires std::same_as<row_value_t<X>, row_value_t<W>> && std::same_as<row_value_t<X>, row_value_t<Out>>
constexpr row_value_t<X> rms_norm(const X& x, const W& weight, Out&& out,
                                  std::type_identity_t<row_value_t<X>> eps = std::numeric_limits<row_value_t<X>>::epsilon())
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto w = as_row(weight);
    const auto y = as_mutable_row(out);
    const std::size_t n = in.size();
    assert(y.size() == n && detail::norm::fits<T>(w.size(), n));
    T ms = T(0);
    for (const T v : in)
        ms += v * v;
    ms = n == 0 ? T(0) : ms / T(n);
    const T rstd = T(1) / math::sqrt(ms + eps);
    for (std::size_t i = 0; i < n; ++i)
        y[i] = in[i] * rstd * detail::norm::param(w, i, T(1));
    return rstd;
}

/// \brief The backward pass of `rms_norm`: writes `dx`, adds into `dweight`.
///
/// With `n = x · rstd` and `g = weight · dout`: `dx = rstd · (g - n · mean(g · n))`, `dweight += dout · n`.
/// \param dout The gradient in the output.
/// \param x The forward pass's input.
/// \param weight The forward pass's weight, or empty.
/// \param rstd What the forward pass returned.
/// \param dx The gradient in the input: written; `x`'s size; may be `dout`.
/// \param dweight Accumulated into; `x`'s size, or empty.
template <row D, row X, row W, mutable_row DX, mutable_row DW>
    requires std::same_as<row_value_t<X>, row_value_t<D>> && std::same_as<row_value_t<X>, row_value_t<W>>
             && std::same_as<row_value_t<X>, row_value_t<DX>> && std::same_as<row_value_t<X>, row_value_t<DW>>
constexpr void rms_norm_backward(const D& dout, const X& x, const W& weight, std::type_identity_t<row_value_t<X>> rstd, DX&& dx,
                                 DW&& dweight)
{
    using T = row_value_t<X>;
    const auto d = as_row(dout);
    const auto in = as_row(x);
    const auto w = as_row(weight);
    const auto gx = as_mutable_row(dx);
    const auto gw = as_mutable_row(dweight);
    const std::size_t n = in.size();
    assert(d.size() == n && gx.size() == n && detail::norm::fits<T>(w.size(), n) && detail::norm::fits<T>(gw.size(), n));
    if (n == 0)
        return;
    T mean_gn = T(0);
    for (std::size_t i = 0; i < n; ++i)
        mean_gn += detail::norm::param(w, i, T(1)) * d[i] * in[i] * rstd;
    mean_gn /= T(n);
    for (std::size_t i = 0; i < n; ++i) {
        const T norm = in[i] * rstd;
        const T g = detail::norm::param(w, i, T(1)) * d[i];
        if (!gw.empty())
            gw[i] += d[i] * norm;
        gx[i] = rstd * (g - norm * mean_gn);
    }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_norm_hpp
