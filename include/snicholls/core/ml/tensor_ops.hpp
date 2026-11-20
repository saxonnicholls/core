//
//  tensor_ops.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` small tensor operations: the sinusoidal timestep embedding, `arange`, `cumsum` and its backward, a stable `argsort`, the top-k mask, triangular fill and the lower-triangular solve with its backward, all `constexpr` over the caller's spans.
/// \ingroup core
///
/// **Overview.** The building blocks that diffusion models, state-space and linear-attention layers and samplers ask
/// of a tensor library, as ggml defines them:
///
/// - `timestep_embedding(t, dim, out, max_period)`: each timestep's row `[cos(t f_0) ... cos(t f_{h-1}), sin(t f_0) ...
///   sin(t f_{h-1})]` with `h = dim / 2`, `f_j = exp(-ln(max_period) · j / h)`, cosines first and a trailing 0 for an
///   odd `dim` (ggml's `ggml_timestep_embedding`, the diffusion models' `timestep_embedding`).
/// - `arange_count(start, stop, step)` and `arange(start, step, out)`: `start + step · i`, `ceil((stop - start) /
///   step)` values (ggml's `ggml_arange`).
/// - `cumsum(x, y)`: the running sum `y_i = y_{i-1} + x_i` (ggml's `ggml_cumsum`, in place allowed); `cumsum_backward`
///   is the reverse running sum.
/// - `argsort(x, indices, order)`: the indices that sort a row ascending or descending, equal values in index order
///   (stable), so the result is the same everywhere; `top_k_mask(x, k, y, scratch)` keeps the `k` largest and sets the
///   rest to -infinity, ties to the lower index.
/// - `tri(kind, rows, cols, x, y, fill)`: keep the lower or upper triangle of each matrix, with or without the
///   diagonal, `fill` elsewhere (ggml's `ggml_tri` and its `ggml_tri_type`, which fills with 0).
/// - `solve_tri(a, b, x, n, k)`: `A X = B` for a lower-triangular `n x n` `A` and `k` right-hand sides, by forward
///   substitution in ggml's order (`ggml_solve_tri`); `solve_tri_backward` gives `dB = A⁻ᵀ dX` and adds
///   `dA = -dB Xᵀ` on the lower triangle.
///
/// \snippet ml/tensor_ops_demo.cpp tensor_ops
///
/// **Relationships.** `ml/sampling.hpp` (a top-k of its own, with values), `ml/row.hpp`. Group: \ref core.
///
/// **Implementations.** One each.
///
/// **Guarantees.** `constexpr`, no allocation, no exceptions. Each multiply and add is its own statement. `argsort` and
/// `top_k_mask` are O(n log n) through `std::sort` with the index as the tie-break; `solve_tri` O(n² k).
///
/// **Errors.** Preconditions, asserted: spans of the sizes the arguments imply; a nonzero step; `k` at most the row;
/// a nonzero diagonal for `solve_tri`.
///
/// **Performance.** Not benchmarked: one pass, or a sort.
///
/// **Tests and demos.** `tests/ml/tensor_ops_test.cpp`, `demos/ml/tensor_ops_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<limits>`, `<span>`, `<type_traits>`,
/// `snicholls/core/math/constexpr_math.hpp`, `ml/row.hpp`.
///
/// **Known issues.** ggml's `argsort` uses `std::sort`, which does not keep equal values in order; here they are, so
/// rows with ties can order differently from ggml's. `arange_count` rounds `(stop - start) / step` up in the row's
/// type, as ggml does in `float`: a step that does not divide the range exactly may give one value more than in
/// exact arithmetic.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_tensor_ops_hpp
#define snicholls_core_ml_tensor_ops_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/row.hpp>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief The sinusoidal timestep embedding: `out[i] = [cos(t_i f_j) for j < h, sin(t_i f_j) for j < h, 0 if dim is odd]`
/// with `h = dim / 2` and `f_j = exp(-ln(max_period) · j / h)` (ggml's `ggml_timestep_embedding`).
/// \param timesteps The timesteps.
/// \param dim The width of each row.
/// \param out Receives `timesteps.size() · dim`.
/// \param max_period The longest period, 10,000 by default.
template <row X, mutable_row Out>
    requires std::same_as<row_value_t<X>, row_value_t<Out>>
constexpr void timestep_embedding(const X& timesteps, std::size_t dim, Out&& out, std::type_identity_t<row_value_t<X>> max_period = 10000)
{
    using T = row_value_t<X>;
    const auto t = as_row(timesteps);
    const auto o = as_mutable_row(out);
    assert(o.size() == t.size() * dim);
    const std::size_t half = dim / 2;
    const T log_period = math::log(max_period);
    for (std::size_t i = 0; i < t.size(); ++i) {
        const auto row_out = o.subspan(i * dim, dim);
        for (std::size_t j = 0; j < half; ++j) {
            const T freq = math::exp(-log_period * T(j) / T(half));
            const T arg = t[i] * freq;
            row_out[j] = math::cos(arg);
            row_out[j + half] = math::sin(arg);
        }
        if (dim % 2 != 0)
            row_out[2 * half] = T(0);
    }
}

/// \brief How many values `arange` gives from `start` towards `stop`: `ceil((stop - start) / step)`, 0 if none.
/// \tparam T A `math::math_number`.
/// \param start The first value.
/// \param stop The bound, not reached.
/// \param step The step, not 0.
/// \return The count.
template <math::math_number T>
[[nodiscard]] constexpr std::size_t arange_count(T start, T stop, T step)
{
    assert(step != T(0));
    const T n = -math::floor(-((stop - start) / step));
    return n > T(0) ? static_cast<std::size_t>(n) : 0;
}

/// \brief `out[i] = start + step · i` (ggml's `ggml_arange`).
/// \param start The first value.
/// \param step The step.
/// \param out Receives the values, `arange_count(start, stop, step)` of them for a bound `stop`.
template <mutable_row Out>
constexpr void arange(std::type_identity_t<row_value_t<Out>> start, std::type_identity_t<row_value_t<Out>> step, Out&& out)
{
    using T = row_value_t<Out>;
    const auto o = as_mutable_row(out);
    for (std::size_t i = 0; i < o.size(); ++i) {
        const T term = step * T(i);
        o[i] = start + term;
    }
}

/// \brief The running sum, `y_0 = x_0`, `y_i = y_{i-1} + x_i` (ggml's `ggml_cumsum`); `y` may be `x`.
/// \param x The row.
/// \param y Receives the sums, `x`'s size.
template <row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void cumsum(const X& x, Y&& y)
{
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    assert(out.size() == in.size());
    for (std::size_t i = 0; i < in.size(); ++i)
        out[i] = i == 0 ? in[0] : out[i - 1] + in[i];
}

/// \brief The backward pass of `cumsum`: the reverse running sum, `dx_i = Σ_{j >= i} dy_j`, from the end; `dx` may be `dy`.
/// \param dy The gradient in the sums.
/// \param dx Receives the gradient in the row, `dy`'s size.
template <row DY, mutable_row DX>
    requires std::same_as<row_value_t<DY>, row_value_t<DX>>
constexpr void cumsum_backward(const DY& dy, DX&& dx)
{
    const auto g = as_row(dy);
    const auto gx = as_mutable_row(dx);
    assert(gx.size() == g.size());
    for (std::size_t i = g.size(); i-- > 0;)
        gx[i] = i + 1 == g.size() ? g[i] : gx[i + 1] + g[i];
}

/// \brief The order `argsort` sorts in.
enum class sort_order
{
    ascending,    ///< Smallest first.
    descending,   ///< Largest first.
};

/// \brief The indices that sort a row, equal values in index order.
/// \param x The row.
/// \param indices Receives `x.size()` indices.
/// \param order Ascending or descending.
template <row X>
constexpr void argsort(const X& x, std::span<std::size_t> indices, sort_order order = sort_order::ascending)
{
    const auto v = as_row(x);
    assert(indices.size() == v.size());
    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;
    if (order == sort_order::ascending)
        std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b] || (!(v[b] < v[a]) && a < b); });
    else
        std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) { return v[a] > v[b] || (!(v[b] > v[a]) && a < b); });
}

/// \brief Keeps the `k` largest values of a row, the rest -infinity: equal values kept in index order.
/// \param x The row.
/// \param k The values kept, at most the row's size.
/// \param y Receives the masked row, `x`'s size; may be `x`.
/// \param scratch The row's size of indices: overwritten.
template <row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void top_k_mask(const X& x, std::size_t k, Y&& y, std::span<std::size_t> scratch)
{
    using T = row_value_t<X>;
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    assert(out.size() == in.size() && k <= in.size());
    argsort(in, scratch, sort_order::descending);
    for (std::size_t r = k; r < scratch.size(); ++r)
        out[scratch[r]] = -std::numeric_limits<T>::infinity();
    for (std::size_t r = 0; r < k; ++r)
        out[scratch[r]] = in[scratch[r]];
}

/// \brief Which triangle `tri` keeps, as ggml's `ggml_tri_type` (column `j`, row `i`).
enum class tri_kind
{
    upper_diag,   ///< `j >= i`.
    upper,        ///< `j > i`.
    lower_diag,   ///< `j <= i`.
    lower,        ///< `j < i`.
};

/// \brief Whether column `j` of row `i` is in the triangle.
/// \param kind The triangle.
/// \param i The row.
/// \param j The column.
/// \return Whether `tri` keeps it.
[[nodiscard]] constexpr bool in_triangle(tri_kind kind, std::size_t i, std::size_t j) noexcept
{
    switch (kind) {
    case tri_kind::upper_diag: return j >= i;
    case tri_kind::upper: return j > i;
    case tri_kind::lower_diag: return j <= i;
    case tri_kind::lower: return j < i;
    }
    return false;
}

/// \brief Keeps one triangle of each `rows x cols` matrix, `fill` elsewhere (ggml's `ggml_tri`, whose fill is 0).
/// \param kind The triangle.
/// \param rows The rows of each matrix.
/// \param cols The columns of each matrix.
/// \param x The matrices, row-major, a multiple of `rows · cols`.
/// \param y Receives them, `x`'s size; may be `x`.
/// \param fill The value outside the triangle: 0, or -infinity for a causal mask.
template <row X, mutable_row Y>
    requires std::same_as<row_value_t<X>, row_value_t<Y>>
constexpr void tri(tri_kind kind, std::size_t rows, std::size_t cols, const X& x, Y&& y, std::type_identity_t<row_value_t<X>> fill = 0)
{
    const auto in = as_row(x);
    const auto out = as_mutable_row(y);
    assert(out.size() == in.size() && rows * cols > 0 && in.size() % (rows * cols) == 0);
    for (std::size_t e = 0; e < in.size(); ++e) {
        const std::size_t i = (e / cols) % rows, j = e % cols;
        out[e] = in_triangle(kind, i, j) ? in[e] : fill;
    }
}

/// \brief Solves `A X = B` for a lower-triangular `A` by forward substitution, as ggml's `ggml_solve_tri`:
/// `X[i][c] = (B[i][c] - Σ_{t < i} A[i][t] X[t][c]) / A[i][i]`, the sum in `t` order.
/// \param a `A`, `n x n` row-major; only the lower triangle is read.
/// \param b `B`, `n x k` row-major.
/// \param x Receives `X`, `n x k`; may be `b`.
/// \param n The order of `A`.
/// \param k The right-hand sides.
template <row A, row B, mutable_row X>
    requires std::same_as<row_value_t<A>, row_value_t<B>> && std::same_as<row_value_t<A>, row_value_t<X>>
constexpr void solve_tri(const A& a, const B& b, X&& x, std::size_t n, std::size_t k)
{
    using T = row_value_t<A>;
    const auto as = as_row(a);
    const auto bs = as_row(b);
    const auto xs = as_mutable_row(x);
    assert(as.size() == n * n && bs.size() == n * k && xs.size() == n * k);
    for (std::size_t c = 0; c < k; ++c)
        for (std::size_t i = 0; i < n; ++i) {
            T sum = T(0);
            for (std::size_t t = 0; t < i; ++t) {
                const T term = as[i * n + t] * xs[t * k + c];
                sum += term;
            }
            assert(as[i * n + i] != T(0));
            xs[i * k + c] = (bs[i * k + c] - sum) / as[i * n + i];
        }
}

/// \brief The backward pass of `solve_tri`: `dB = A⁻ᵀ dX` by back substitution (written), and `dA -= dB Xᵀ` on the
/// lower triangle (added).
/// \param a `A`, `n x n`.
/// \param x The solution `X`, `n x k`.
/// \param dx The gradient in `X`, `n x k`.
/// \param da Accumulates the gradient in `A`, `n x n` (the upper triangle untouched), or empty.
/// \param db Receives the gradient in `B`, `n x k`.
/// \param n The order of `A`.
/// \param k The right-hand sides.
template <row A, row X, row DX, mutable_row DA, mutable_row DB>
    requires std::same_as<row_value_t<A>, row_value_t<X>> && std::same_as<row_value_t<A>, row_value_t<DX>>
             && std::same_as<row_value_t<A>, row_value_t<DA>> && std::same_as<row_value_t<A>, row_value_t<DB>>
constexpr void solve_tri_backward(const A& a, const X& x, const DX& dx, DA&& da, DB&& db, std::size_t n, std::size_t k)
{
    using T = row_value_t<A>;
    const auto as = as_row(a);
    const auto xs = as_row(x);
    const auto g = as_row(dx);
    const auto ga = as_mutable_row(da);
    const auto gb = as_mutable_row(db);
    assert(as.size() == n * n && xs.size() == n * k && g.size() == n * k && gb.size() == n * k && (ga.empty() || ga.size() == n * n));
    for (std::size_t c = 0; c < k; ++c)
        for (std::size_t i = n; i-- > 0;) {
            T sum = T(0);
            for (std::size_t t = i + 1; t < n; ++t) {
                const T term = as[t * n + i] * gb[t * k + c];
                sum += term;
            }
            gb[i * k + c] = (g[i * k + c] - sum) / as[i * n + i];
        }
    if (ga.empty())
        return;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            T sum = T(0);
            for (std::size_t c = 0; c < k; ++c) {
                const T term = gb[i * k + c] * xs[j * k + c];
                sum += term;
            }
            ga[i * n + j] -= sum;
        }
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_tensor_ops_hpp
