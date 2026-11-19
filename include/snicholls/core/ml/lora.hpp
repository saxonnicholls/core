//
//  lora.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` LoRA: a trainable low-rank update `y = W x + (α/r) B A x` beside a frozen base (dense, bfloat16 or NF4 for QLoRA), forward and backward for one row, a sequence and attention's heads, its initialisation, merge and unmerge, and the optimiser's view of the adapter alone.
/// \ingroup core
///
/// **Overview.** Low-rank adaptation (Hu et al. 2021) fine-tunes a model by freezing each weight `W` (`rows x cols`)
/// and learning `ΔW = s · B A`, `A` of `r x cols`, `B` of `rows x r`, `r` small, `s = α / r` (TRAINING.md T13). Only
/// `A` and `B` have gradients and optimiser state, so a model of billions of weights is tuned with a few million.
/// QLoRA (Dettmers et al. 2023, eq. 5) stores `W` in NF4 (`ml/nf4.hpp`) and dequantises it as it is read.
///
/// - `lora_weights<W, T>{base, a, b, rank, scaling}`: the view; `W` is `dense_weights<S>` (any storage `linear`
///   reads, `float`, `double`, `bfloat16`) or `nf4_weights`.
/// - `lora_linear(w, x, y, u, bias)`: `u = A x` (kept for the backward), `y = W x + b + s · B u`.
/// - `lora_linear_backward(w, x, u, dy, dx, da, db, du)`: `du = s · Bᵀ dy`, `dB += s · dy uᵀ`, `dA += du xᵀ`, `dx +=
///   Wᵀ dy + Aᵀ du`. No gradient reaches `W`.
/// - `lora_sequence` / `lora_sequence_backward`: the same over `n` rows through a `linalg` backend: the base by
///   `linear_sequence` (any storage with an overload), the adapter as three matrix products.
/// - `lora_heads` / `lora_heads_backward`: the outputs head by head over a batch of sequences, as `linear_heads` lays
///   out attention's queries, keys and values; the base dense in the sequence's type.
/// - `lora_initialise(a, b, cols, g, kind)`: `B = 0`, so a new adapter changes nothing, and `A` Kaiming-uniform (the
///   bound `1/√cols` of PyTorch's `nn.Linear`, PEFT's default) or Gaussian with standard deviation `1/r` (PEFT's
///   `"gaussian"`), from the caller's `stats::wyrand`.
/// - `lora_merge` / `lora_unmerge`: `W ± s · B A` into a dense matrix for inference, and back.
/// - `lora_parameters<T>`: the adapter's two tensors and their gradients, as the spans an optimiser steps
///   (`adamw_step` on each), so nothing of the base is touched.
///
/// \snippet ml/lora_demo.cpp lora
///
/// **Relationships.** `ml/linear.hpp` (`linear`, `linear_sequence`, `linear_heads`, the products), `ml/nf4.hpp` (the
/// quantised base), `ml/optim.hpp` (`adamw_step`), `stats/random.hpp` (`wyrand`). Group: \ref core.
///
/// **Implementations.** One. The products go through the backend, so a device library runs them; the base's input
/// gradient is `linear_sequence_backward_input`, overloaded per storage.
///
/// **Guarantees.** `constexpr`, no allocation (the caller's `u` and `du`, `r` a row). Linear in `rows · cols` for the
/// base plus `r · (rows + cols)` for the adapter. Gradients `da`, `db`, `dx` are added into; `u` and `du` written.
/// With `B = 0` the output is the base's to the bit (the adapter adds `0`). Merge then unmerge returns each weight
/// within `2 ε (|w| + |s · (BA)_ij|)` of itself (each is one rounding of a sum and one of a difference; the product
/// `(BA)_ij` is computed identically both times), and exactly when the update is exact.
///
/// **Errors.** Preconditions, asserted: `a` has `rank · cols`, `b` `rows · rank`, the rows and scratch their sizes.
///
/// **Definitions.** The update, `s = α / r`, zero `B`: from the LoRA paper (recalled, not re-read) and QLoRA's eq. 5
/// (checked). The initialisations are PEFT's, recalled.
///
/// **Performance.** `lora_linear` and `lora_linear_backward` (rank 16 on a 1024 x 1024 base, 8 rows) against the same
/// loops by hand, after a bit-for-bit check (`benchmarks/ml/lora_benchmark.cpp`): 10.4 ms against 10.2 ms, medians of
/// three, level (Xeon W-3245, -O2, Apple clang 17, 2026-11-19, one thread, CPU time, load average about 100: a record, not a quiet-machine figure). The adapter adds `2r / rows` of the base's work forward and about as much backward.
///
/// **Tests and demos.** `tests/ml/lora_test.cpp`, `demos/ml/lora_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `math/constexpr_math.hpp`, `ml/linear.hpp`, `ml/nf4.hpp`, `ml/row.hpp`, `stats/random.hpp`.
///
/// **Known issues.** DoRA, rsLoRA's `α/√r` and other variants are not here (`scaling` is the caller's, so rsLoRA is
/// one argument). Merging into `bfloat16` or NF4 storage is the caller's (merge in float, then round).
///
/// **Status.** Experimental, new in Unreleased.

#ifndef snicholls_core_ml_lora_hpp
#define snicholls_core_ml_lora_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/nf4.hpp>
#include <snicholls/core/ml/row.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A frozen base and its trainable low-rank adapter: `y = W x + scaling · B A x`.
/// \ingroup core
/// \tparam W The base: `dense_weights<S>` or `nf4_weights`.
/// \tparam T The adapter's and the activations' number type.
template <typename W, math::math_number T>
struct lora_weights
{
    W base{};                   ///< The frozen weight, `rows x cols`.
    std::span<const T> a{};     ///< `A`, `rank x cols`, row by row.
    std::span<const T> b{};     ///< `B`, `rows x rank`, row by row.
    std::size_t rank = 0;       ///< `r`.
    T scaling = T(1);           ///< `s = α / r`.

    /// \brief The outputs. \return The base's rows.
    [[nodiscard]] constexpr std::size_t rows() const noexcept { return base.rows; }
    /// \brief The inputs. \return The base's columns.
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return base.cols; }
};

/// \brief LoRA's scaling `α / r`. \param alpha `α`. \param rank `r`, at least 1. \return The scaling.
template <math::math_number T>
constexpr T lora_scaling(T alpha, std::size_t rank) noexcept(std::is_floating_point_v<T>)
{
    assert(rank > 0);
    return alpha / T(rank);
}

/// \brief The input's gradient through a frozen dense weight: `dx += Wᵀ dy`, for each output `i` in order.
/// \param w The matrix. \param dy `rows`. \param dx Added into, `cols`.
template <typename S, math::math_number T>
constexpr void linear_backward_input(const dense_weights<S>& w, std::span<const T> dy, std::type_identity_t<std::span<T>> dx) noexcept(std::is_floating_point_v<T>)
{
    assert(dy.size() == w.rows && dx.size() == w.cols);
    for (std::size_t i = 0; i < w.rows; ++i) {
        const auto r = w.row(i);
        for (std::size_t j = 0; j < w.cols; ++j) {
            const T p = detail::linear::widen<T>(r[j]) * dy[i];
            dx[j] += p;
        }
    }
}

/// \brief `dX += dY W` over `n` rows through a frozen dense weight, one matrix product on the backend.
/// \param backend A `linalg::blas_backend`. \param w The matrix. \param dys `n · rows`. \param dxs Added into, `n · cols`.
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void linear_sequence_backward_input(const B& backend, const dense_weights<T>& w, std::type_identity_t<std::span<const T>> dys,
                                              std::type_identity_t<std::span<T>> dxs)
{
    assert(w.rows > 0 && dys.size() % w.rows == 0 && dxs.size() == dys.size() / w.rows * w.cols);
    const std::size_t n = dys.size() / w.rows;
    if (n == 0 || w.cols == 0)
        return;
    using detail::linear::view;
    detail::linear::product(backend, T(1), view(dys, n, w.rows), view(w.data, w.rows, w.cols), T(1), view(dxs, n, w.cols));
}

/// \brief One row through the base and the adapter: `u = A x`, `y = W x + b + s · B u`.
/// \param w The weights. \param x The input, `cols`. \param y The output, `rows`. \param u Written, `rank`: `A x`,
/// for the backward. \param bias `rows`, or empty.
template <typename W, math::math_number T>
constexpr void lora_linear(const lora_weights<W, T>& w, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<T>> y,
                           std::type_identity_t<std::span<T>> u, std::type_identity_t<std::span<const T>> bias = {}) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t rows = w.rows(), cols = w.cols(), r = w.rank;
    assert(x.size() == cols && y.size() == rows && u.size() == r && w.a.size() == r * cols && w.b.size() == rows * r);
    linear(w.base, x, y, bias);
    for (std::size_t k = 0; k < r; ++k) {
        T acc = T(0);
        for (std::size_t j = 0; j < cols; ++j)
            acc += w.a[k * cols + j] * x[j];
        u[k] = acc;
    }
    for (std::size_t i = 0; i < rows; ++i) {
        T acc = T(0);
        for (std::size_t k = 0; k < r; ++k)
            acc += w.b[i * r + k] * u[k];
        y[i] += w.scaling * acc;
    }
}

/// \brief The backward pass of `lora_linear`: `du = s · Bᵀ dy`, `dB += s · dy uᵀ`, `dA += du xᵀ`, `dx += Wᵀ dy + Aᵀ du`.
/// \param w The weights. \param x The forward's input. \param u The forward's `A x`. \param dy The output's gradient.
/// \param dx Added into, `cols`, or empty. \param da Added into, `rank · cols`. \param db Added into, `rows · rank`.
/// \param du Written, `rank`.
template <typename W, math::math_number T>
constexpr void lora_linear_backward(const lora_weights<W, T>& w, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<const T>> u,
                                    std::type_identity_t<std::span<const T>> dy, std::type_identity_t<std::span<T>> dx,
                                    std::type_identity_t<std::span<T>> da, std::type_identity_t<std::span<T>> db,
                                    std::type_identity_t<std::span<T>> du) noexcept(std::is_floating_point_v<T>)
{
    const std::size_t rows = w.rows(), cols = w.cols(), r = w.rank;
    assert(x.size() == cols && u.size() == r && dy.size() == rows && (dx.empty() || dx.size() == cols) && da.size() == r * cols
           && db.size() == rows * r && du.size() == r);
    for (std::size_t k = 0; k < r; ++k)
        du[k] = T(0);
    for (std::size_t i = 0; i < rows; ++i) {
        const T g = w.scaling * dy[i];
        for (std::size_t k = 0; k < r; ++k) {
            du[k] += w.b[i * r + k] * g;
            db[i * r + k] += g * u[k];
        }
    }
    for (std::size_t k = 0; k < r; ++k)
        for (std::size_t j = 0; j < cols; ++j)
            da[k * cols + j] += du[k] * x[j];
    if (dx.empty())
        return;
    linear_backward_input(w.base, dy, dx);
    for (std::size_t k = 0; k < r; ++k)
        for (std::size_t j = 0; j < cols; ++j)
            dx[j] += w.a[k * cols + j] * du[k];
}

/// \brief `n` rows through the base and the adapter on the backend: `Y = X Wᵀ + b` (`linear_sequence` for the
/// base's storage), `U = X Aᵀ`, `Y += s · U Bᵀ`.
/// \param backend A `linalg::blas_backend`. \param w The weights. \param xs `n · cols`. \param ys Written, `n · rows`.
/// \param us Written, `n · rank`. \param bias `rows`, or empty.
template <typename W, math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void lora_sequence(const B& backend, const lora_weights<W, T>& w, std::type_identity_t<std::span<const T>> xs,
                             std::type_identity_t<std::span<T>> ys, std::type_identity_t<std::span<T>> us,
                             std::type_identity_t<std::span<const T>> bias = {})
{
    const std::size_t rows = w.rows(), cols = w.cols(), r = w.rank;
    assert(cols > 0 && xs.size() % cols == 0);
    const std::size_t n = xs.size() / cols;
    assert(ys.size() == n * rows && us.size() == n * r && w.a.size() == r * cols && w.b.size() == rows * r);
    linear_sequence(backend, w.base, xs, ys, bias);
    if (n == 0 || r == 0)
        return;
    linear_sequence(backend, dense_weights<T>{w.a, r, cols}, xs, us);
    using detail::linear::view;
    detail::linear::product(backend, w.scaling, view(std::span<const T>(us), n, r), linalg::transposed(view(w.b, rows, r)), T(1),
                            view(ys, n, rows));
}

/// \brief The backward pass of `lora_sequence`: `dU = s · dY B`, `dB += s · dYᵀ U`, `dA += dUᵀ X`, `dX += dY W + dU A`.
/// \param backend A `linalg::blas_backend`. \param w The weights. \param xs The forward's `n` inputs. \param us The
/// forward's `U`. \param dys `n · rows`. \param dxs Added into, `n · cols`, or empty. \param da Added into.
/// \param db Added into. \param dus Written, `n · rank`.
template <typename W, math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void lora_sequence_backward(const B& backend, const lora_weights<W, T>& w, std::type_identity_t<std::span<const T>> xs,
                                      std::type_identity_t<std::span<const T>> us, std::type_identity_t<std::span<const T>> dys,
                                      std::type_identity_t<std::span<T>> dxs, std::type_identity_t<std::span<T>> da,
                                      std::type_identity_t<std::span<T>> db, std::type_identity_t<std::span<T>> dus)
{
    const std::size_t rows = w.rows(), cols = w.cols(), r = w.rank;
    assert(cols > 0 && xs.size() % cols == 0);
    const std::size_t n = xs.size() / cols;
    assert(us.size() == n * r && dys.size() == n * rows && (dxs.empty() || dxs.size() == xs.size()) && da.size() == r * cols
           && db.size() == rows * r && dus.size() == n * r);
    if (n == 0)
        return;
    using detail::linear::view;
    const auto dy = view(dys, n, rows);
    if (r > 0) {
        detail::linear::product(backend, w.scaling, dy, view(w.b, rows, r), T(0), view(dus, n, r));
        detail::linear::product(backend, w.scaling, linalg::transposed(dy), view(us, n, r), T(1), view(db, rows, r));
        detail::linear::product(backend, T(1), linalg::transposed(view(std::span<const T>(dus), n, r)), view(xs, n, cols), T(1),
                                view(da, r, cols));
    }
    if (dxs.empty())
        return;
    linear_sequence_backward_input(backend, w.base, dys, dxs);
    if (r > 0)
        detail::linear::product(backend, T(1), view(std::span<const T>(dus), n, r), view(w.a, r, cols), T(1), view(dxs, n, cols));
}

/// \brief `lora_sequence` with the outputs head by head, over `sequences` sequences end to end, as `linear_heads`
/// lays them out: `linear_heads` for the base, then for each sequence and head `Y_sh += s · U_s B_hᵀ`, `B_h` the
/// head's `rows / heads` rows of `B`.
/// \param backend A `linalg::blas_backend`. \param w The weights, the base dense in `T`. \param xs `sequences · n ·
/// cols`. \param out Written, `sequences · heads · n · d`. \param us Written, `sequences · n · rank`. \param bias
/// `rows`, or empty. \param heads The heads. \param sequences The sequences.
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void lora_heads(const B& backend, const lora_weights<dense_weights<T>, T>& w, std::type_identity_t<std::span<const T>> xs,
                          std::type_identity_t<std::span<T>> out, std::type_identity_t<std::span<T>> us,
                          std::type_identity_t<std::span<const T>> bias, std::size_t heads, std::size_t sequences = 1)
{
    const std::size_t rows = w.rows(), cols = w.cols(), r = w.rank;
    assert(heads > 0 && sequences > 0 && rows % heads == 0 && cols > 0 && xs.size() % (cols * sequences) == 0);
    const std::size_t d = rows / heads, n = xs.size() / cols / sequences;
    assert(us.size() == sequences * n * r && out.size() == sequences * n * rows);
    linear_heads(backend, w.base, xs, out, bias, heads, sequences);
    if (n == 0 || r == 0)
        return;
    linear_sequence(backend, dense_weights<T>{w.a, r, cols}, xs, us);
    using detail::linear::view;
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t h = 0; h < heads; ++h)
            detail::linear::product(backend, w.scaling, view(std::span<const T>(us).subspan(s * n * r, n * r), n, r),
                                    linalg::transposed(view(w.b.subspan(h * d * r, d * r), d, r)), T(1),
                                    view(out.subspan((s * heads + h) * n * d, n * d), n, d));
}

/// \brief The backward pass of `lora_heads`: for each sequence and head, `dU_s += s · dY_sh B_h`, `dB_h += s · dY_shᵀ
/// U_s`, `dX_s += dY_sh W_h`; then `dA += dUᵀ X`, `dX += dU A`.
/// \param backend A `linalg::blas_backend`. \param w The weights. \param xs The forward's inputs. \param us The
/// forward's `U`. \param douts Laid out as `lora_heads` writes. \param dxs Added into, or empty. \param da Added into.
/// \param db Added into. \param dus Written, `sequences · n · rank`. \param heads The heads. \param sequences The sequences.
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void lora_heads_backward(const B& backend, const lora_weights<dense_weights<T>, T>& w, std::type_identity_t<std::span<const T>> xs,
                                   std::type_identity_t<std::span<const T>> us, std::type_identity_t<std::span<const T>> douts,
                                   std::type_identity_t<std::span<T>> dxs, std::type_identity_t<std::span<T>> da,
                                   std::type_identity_t<std::span<T>> db, std::type_identity_t<std::span<T>> dus, std::size_t heads,
                                   std::size_t sequences = 1)
{
    const std::size_t rows = w.rows(), cols = w.cols(), r = w.rank;
    assert(heads > 0 && sequences > 0 && rows % heads == 0 && cols > 0 && xs.size() % (cols * sequences) == 0);
    const std::size_t d = rows / heads, n = xs.size() / cols / sequences;
    assert(us.size() == sequences * n * r && dus.size() == us.size() && douts.size() == sequences * n * rows
           && (dxs.empty() || dxs.size() == xs.size()) && da.size() == r * cols && db.size() == rows * r);
    if (n == 0)
        return;
    using detail::linear::view;
    for (T& v : dus)
        v = T(0);
    for (std::size_t s = 0; s < sequences; ++s)
        for (std::size_t h = 0; h < heads; ++h) {
            const auto dy = view(douts.subspan((s * heads + h) * n * d, n * d), n, d);
            if (r > 0) {
                detail::linear::product(backend, w.scaling, dy, view(w.b.subspan(h * d * r, d * r), d, r), T(1),
                                        view(dus.subspan(s * n * r, n * r), n, r));
                detail::linear::product(backend, w.scaling, linalg::transposed(dy), view(us.subspan(s * n * r, n * r), n, r), T(1),
                                        view(db.subspan(h * d * r, d * r), d, r));
            }
            if (!dxs.empty())
                detail::linear::product(backend, T(1), dy, view(w.base.data.subspan(h * d * cols, d * cols), d, cols), T(1),
                                        view(dxs.subspan(s * n * cols, n * cols), n, cols));
        }
    if (r == 0)
        return;
    const std::size_t m = sequences * n;
    detail::linear::product(backend, T(1), linalg::transposed(view(std::span<const T>(dus), m, r)), view(xs, m, cols), T(1), view(da, r, cols));
    if (!dxs.empty())
        detail::linear::product(backend, T(1), view(std::span<const T>(dus), m, r), view(w.a, r, cols), T(1), view(dxs, m, cols));
}

/// \brief How `A` starts.
/// \ingroup core
enum class lora_init : std::uint8_t
{
    kaiming_uniform,   ///< Uniform in `±1/√cols`: PyTorch's `nn.Linear` default, PEFT's default.
    gaussian,          ///< Normal with standard deviation `1/r`: PEFT's `"gaussian"`.
};

/// \brief Initialises an adapter: `B = 0`, `A` drawn from the caller's generator.
/// \param a Written, `rank · cols`. \param b Written, `rows · rank` (zeros). \param cols The inputs. \param g The
/// generator. \param kind The distribution of `A`.
template <math::math_number T>
constexpr void lora_initialise(std::span<T> a, std::type_identity_t<std::span<T>> b, std::size_t cols, stats::wyrand& g,
                               lora_init kind = lora_init::kaiming_uniform) noexcept(std::is_floating_point_v<T>)
{
    assert(cols > 0 && a.size() % cols == 0);
    const std::size_t rank = a.size() / cols;
    for (T& v : b)
        v = T(0);
    const double bound = 1.0 / math::sqrt(static_cast<double>(cols));
    for (T& v : a) {
        if (kind == lora_init::kaiming_uniform) {
            v = T((2.0 * stats::uniform_real(g) - 1.0) * bound);
        } else {
            const double u = (static_cast<double>(g() >> 11) + 0.5) * 0x1.0p-53;   // in (0, 1): a finite quantile
            v = T(math::normal_quantile(u) / static_cast<double>(rank));
        }
    }
}

namespace detail::lora
{
    template <typename T>
    constexpr T update(std::span<const T> a, std::span<const T> b, std::size_t rank, std::size_t cols, std::size_t i, std::size_t j)
    {
        T acc = T(0);
        for (std::size_t k = 0; k < rank; ++k)
            acc += b[i * rank + k] * a[k * cols + j];
        return acc;
    }
} // namespace detail::lora

/// \brief Merges the adapter into a dense weight for inference: `W += s · B A`.
/// \param w The weight, `rows · cols`, updated. \param cols The inputs. \param a `A`. \param b `B`. \param rank `r`.
/// \param scaling `s`.
template <math::math_number T>
constexpr void lora_merge(std::span<T> w, std::size_t cols, std::type_identity_t<std::span<const T>> a,
                          std::type_identity_t<std::span<const T>> b, std::size_t rank, std::type_identity_t<T> scaling) noexcept(std::is_floating_point_v<T>)
{
    assert(cols > 0 && w.size() % cols == 0 && a.size() == rank * cols && b.size() == w.size() / cols * rank);
    for (std::size_t i = 0; i < w.size() / cols; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            w[i * cols + j] += scaling * detail::lora::update(a, b, rank, cols, i, j);
}

/// \brief Removes a merged adapter: `W -= s · B A`, the update computed as `lora_merge` computed it.
/// \param w The weight, updated. \param cols The inputs. \param a `A`. \param b `B`. \param rank `r`. \param scaling `s`.
template <math::math_number T>
constexpr void lora_unmerge(std::span<T> w, std::size_t cols, std::type_identity_t<std::span<const T>> a,
                            std::type_identity_t<std::span<const T>> b, std::size_t rank, std::type_identity_t<T> scaling) noexcept(std::is_floating_point_v<T>)
{
    assert(cols > 0 && w.size() % cols == 0 && a.size() == rank * cols && b.size() == w.size() / cols * rank);
    for (std::size_t i = 0; i < w.size() / cols; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            w[i * cols + j] -= scaling * detail::lora::update(a, b, rank, cols, i, j);
}

/// \brief The adapter as an optimiser sees it: `A` and `B` and their gradients, nothing of the base.
/// \ingroup core
template <math::math_number T>
struct lora_parameters
{
    std::span<T> a{};    ///< `A`.
    std::span<T> b{};    ///< `B`.
    std::span<T> da{};   ///< `A`'s gradient.
    std::span<T> db{};   ///< `B`'s gradient.

    /// \brief The trainable tensors, `A` then `B`. \return Two spans.
    [[nodiscard]] constexpr std::array<std::span<T>, 2> parameters() const noexcept { return {a, b}; }
    /// \brief Their gradients, in the same order. \return Two spans.
    [[nodiscard]] constexpr std::array<std::span<T>, 2> gradients() const noexcept { return {da, db}; }
    /// \brief The trainable numbers. \return `|A| + |B|`.
    [[nodiscard]] constexpr std::size_t size() const noexcept { return a.size() + b.size(); }
};

} // namespace snicholls::ml

#endif // snicholls_core_ml_lora_hpp
