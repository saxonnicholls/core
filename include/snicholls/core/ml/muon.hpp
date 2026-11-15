//
//  muon.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::muon_step`: the Muon optimiser for a weight matrix, Newton-Schulz orthogonalisation of its momentum through a `linalg` backend.
/// \ingroup core
///
/// **Overview.** Muon (Jordan et al., 2024; nanochat's optimiser for the matrices of its blocks) replaces a matrix
/// parameter's update by its nearest semi-orthogonal matrix, approximately: the momentum `M` is divided by its
/// Frobenius norm and iterated `X ← a X + (b A + c A²) X` with `A = X Xᵀ`, the quintic of `muon_newton_schulz`
/// (`ml/optim.hpp`), which moves every singular value toward 1 (into roughly `[0.7, 1.2]`). The products run on any
/// `linalg::blas_backend`: `linalg::reference` here and in tests, a BLAS or device backend for speed.
///
/// - `muon_orthogonalise(backend, g, rows, cols, out, work)`: the iteration on an `rows x cols` matrix, on its
///   transpose when it has more rows than columns, so `A` is the smaller square.
/// - `muon_step(backend, w, g, momentum, rows, cols, config, work)`: one update. `momentum ← μ momentum + g`, then
///   `g + μ momentum` (Nesterov) or `momentum` is orthogonalised and `w ← w − lr · sqrt(max(1, rows / cols)) · X`,
///   nanochat's scaling.
/// - `muon_work_size(rows, cols)`: the working space both need, so nothing is allocated.
///
/// Muon is for the 2-D weights inside the blocks; embeddings, the head, norms and biases use `adamw_step`.
///
/// \snippet ml/muon_demo.cpp muon
///
/// A tiny GPT-2 trained end to end with both: token and position embeddings, two blocks
/// (`block_forward_sequence`, `block_backward`), the final norm, a tied head (`lm_head_loss`), Muon on the blocks'
/// matrices and AdamW on the rest. It learns a 17-token sequence (loss 2.81, a uniform guess, to 0.01 in 30 steps),
/// with the same bits on a second run:
///
/// \snippet ml/muon_demo.cpp train_gpt2
///
/// **Relationships.** `ml/optim.hpp` (`muon_newton_schulz`, `adamw_step`), `linalg/backend.hpp` (the backend
/// concept), `linalg/reference.hpp` (the reference backend). Group: \ref core.
///
/// **Implementations.** One, over the backend's `gemm`.
///
/// **Guarantees.** `constexpr` with a `constexpr` backend, no allocation (the caller's `muon_work_size` elements),
/// `steps · (2 r² s + r³)` multiply-adds for `r = min(rows, cols)`, `s = max(rows, cols)`. A zero gradient gives a
/// zero update.
///
/// **Errors.** Preconditions, asserted: `g`, `out`, `w` and `momentum` have `rows · cols`; `work` has
/// `muon_work_size(rows, cols)`.
///
/// **Performance.** Not benchmarked: three `gemm`s per step, the backend's.
///
/// **Tests and demos.** `tests/ml/muon_test.cpp`, `demos/ml/muon_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<cstddef>`, `<span>`, `snicholls/core/math/constexpr_math.hpp`,
/// `linalg/backend.hpp`, `ml/optim.hpp`.
///
/// **Known issues.** nanochat runs the iteration in bfloat16; here it runs in `T`.
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_core_ml_muon_hpp
#define snicholls_core_ml_muon_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/optim.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>

namespace snicholls::ml
{

/// \brief Muon's settings for one matrix.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct muon_config
{
    T lr = T(0.02);          ///< The learning rate.
    T momentum = T(0.95);    ///< `μ`.
    bool nesterov = true;    ///< Orthogonalise `g + μ momentum` rather than `momentum`.
};

/// \brief The working space `muon_orthogonalise` and `muon_step` need, in elements.
/// \param rows The matrix's rows.
/// \param cols Its columns.
/// \return `4 · rows · cols + 2 · min(rows, cols)²`: the iteration's two copies of `X` and its two squares, and
/// `muon_step`'s update and result.
[[nodiscard]] constexpr std::size_t muon_work_size(std::size_t rows, std::size_t cols) noexcept
{
    const std::size_t r = std::min(rows, cols);
    return 4 * rows * cols + 2 * r * r;
}

/// \brief Muon's approximate orthogonalisation of a matrix: `steps` Newton-Schulz iterations from `g / |g|_F`.
/// \param backend A `linalg::blas_backend`.
/// \param g The matrix, `rows x cols`, row-major.
/// \param rows Its rows.
/// \param cols Its columns.
/// \param out Receives the result, `rows x cols`, row-major; may not alias `g`.
/// \param work At least `2 · rows · cols + 2 · min(rows, cols)²` elements (`muon_work_size` is enough).
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void muon_orthogonalise(const B& backend, std::span<const T> g, std::size_t rows, std::size_t cols, std::span<T> out,
                                  std::span<T> work)
{
    using right = linalg::layout_right;
    assert(g.size() == rows * cols && out.size() == g.size() && work.size() >= 2 * rows * cols + 2 * std::min(rows, cols) * std::min(rows, cols));
    const bool tall = rows > cols;
    const std::size_t r = tall ? cols : rows, s = tall ? rows : cols;   // X is r x s, r <= s
    const auto x = work.first(r * s), x2 = work.subspan(r * s, r * s), a = work.subspan(2 * r * s, r * r),
               bm = work.subspan(2 * r * s + r * r, r * r);
    T norm2 = T(0);
    for (const T v : g)
        norm2 += v * v;
    const T scale = T(1) / (math::sqrt(norm2) + T(1e-7));
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            (tall ? x[j * s + i] : x[i * s + j]) = g[i * cols + j] * scale;
    using ns = muon_newton_schulz<T>;
    const linalg::matrix_view<T, right> x2v(x2.data(), {r, s}), av(a.data(), {r, r}), bv(bm.data(), {r, r});
    const linalg::matrix_view<const T, right> xc(x.data(), {r, s}), ac(a.data(), {r, r}), bc(bm.data(), {r, r});
    // The shapes are right by construction and C never overlaps A or B, so every product succeeds
    const auto product = [&](T alpha, const auto& lhs, const auto& rhs, T beta, const auto& c) {
        [[maybe_unused]] const auto done = linalg::gemm(backend, alpha, lhs, rhs, beta, c);
        assert(done.is_ok());
    };
    for (int step = 0; step < ns::steps; ++step) {
        product(T(1), xc, linalg::transposed(xc), T(0), av);   // A = X Xᵀ
        for (std::size_t i = 0; i < r * r; ++i)
            bm[i] = ns::b * a[i];
        product(ns::c, ac, ac, T(1), bv);                       // B = b A + c A²
        for (std::size_t i = 0; i < r * s; ++i)
            x2[i] = ns::a * x[i];
        product(T(1), bc, xc, T(1), x2v);                       // X ← a X + B X
        std::copy(x2.begin(), x2.end(), x.begin());
    }
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            out[i * cols + j] = tall ? x[j * s + i] : x[i * s + j];
}

/// \brief One Muon update of a weight matrix.
/// \param backend A `linalg::blas_backend`.
/// \param w The weights, `rows x cols`, row-major: updated.
/// \param g Their gradient.
/// \param momentum The momentum buffer, zero before the first step: updated.
/// \param rows The rows.
/// \param cols The columns.
/// \param config The settings.
/// \param work `muon_work_size(rows, cols)` elements.
template <math::math_number T, typename B>
    requires linalg::blas_backend<B, T>
constexpr void muon_step(const B& backend, std::span<T> w, std::span<const T> g, std::span<T> momentum, std::size_t rows,
                         std::size_t cols, const muon_config<T>& config, std::span<T> work)
{
    const std::size_t n = rows * cols;
    assert(w.size() == n && g.size() == n && momentum.size() == n && work.size() >= muon_work_size(rows, cols));
    const auto update = work.first(n), rest = work.subspan(n);
    for (std::size_t i = 0; i < n; ++i) {
        momentum[i] = config.momentum * momentum[i] + g[i];
        update[i] = config.nesterov ? g[i] + config.momentum * momentum[i] : momentum[i];
    }
    const auto x = rest.first(n);
    muon_orthogonalise(backend, std::span<const T>(update), rows, cols, x, rest.subspan(n));
    const T step = config.lr * math::sqrt(std::max(T(1), T(rows) / T(cols)));
    for (std::size_t i = 0; i < n; ++i)
        w[i] -= step * x[i];
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_muon_hpp
