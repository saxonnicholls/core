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
/// (`ml/optim.hpp`), which moves every singular value toward 1: five steps carry any singular value of at least `1.55e-3` (of the
/// normalised matrix) into about `[0.682, 1.2024]`, and a smaller one `s` only to about `a⁵ s ≈ 488 s`, below 0.7. The products run on any
/// `linalg::blas_backend`: `linalg::reference` here and in tests, a BLAS or device backend for speed.
///
/// - `muon_orthogonalise(backend, g, rows, cols, out, work)`: the iteration on an `rows x cols` matrix, on its
///   transpose when it has more rows than columns, so `A` is the smaller square.
/// - `muon_step(backend, w, g, momentum, rows, cols, config, work)`: one update. `momentum ← μ momentum + g`, then
///   `g + μ momentum` (Nesterov) or `momentum` is orthogonalised and `w ← w − lr · sqrt(max(1, rows / cols)) · X`,
///   nanochat's scaling.
/// - `muon_work_size(rows, cols)`: the working space both need, so nothing is allocated.
///
/// Every step is a call on the backend (TRAINING.md T9): the products are its `gemm`, and the rest are `ml/linear.hpp`'s
/// elementwise steps (`copy_rows`, `scale_rows`, `add_rows`, and `sum_squares` for the norm), whose host overloads take
/// any backend. A device library overloads them for its own backend (snicholls-gpu's `cuda_training_backend`), so a
/// Muon step runs on the device over device memory, as `adamw_rows` does; through core's reference backend the result
/// is the elementwise loops' to the bit.
///
/// \snippet ml/muon_demo.cpp muon_backend
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
/// **Relationships.** `ml/optim.hpp` (`muon_newton_schulz`, `adamw_step`, `adamw_rows`), `ml/linear.hpp` (the
/// elementwise steps on the backend), `linalg/backend.hpp` (the backend concept), `linalg/reference.hpp` (the reference backend). Group: \ref core.
///
/// **Implementations.** One, over the backend's `gemm` and elementwise steps. A tall matrix is iterated with
/// `A = Xᵀ X` and `X ← a X + X B` rather than transposed, the same arithmetic term for term.
///
/// **Guarantees.** `constexpr` with a `constexpr` backend, no allocation (the caller's `muon_work_size` elements),
/// `steps · (2 r² s + r³)` multiply-adds for `r = min(rows, cols)`, `s = max(rows, cols)`. A zero gradient gives a
/// zero update. The norm is one `sum_squares`, so on a device the norm's value comes back to the host once per
/// orthogonalisation (a synchronisation).
///
/// **Errors.** Preconditions, asserted: `g`, `out`, `w` and `momentum` have `rows · cols`; `work` has
/// `muon_work_size(rows, cols)`.
///
/// **Performance.** Not benchmarked here: three `gemm`s and two copies per iteration, the backend's. On a GPU see
/// snicholls-gpu's `cuda_ops_benchmark.cpp` (a GPT-2-small Muon step against AdamW on the same matrices).
///
/// **Tests and demos.** `tests/ml/muon_test.cpp`, `demos/ml/muon_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<cstddef>`, `<span>`, `snicholls/core/math/constexpr_math.hpp`,
/// `<utility>`, `linalg/backend.hpp`, `ml/linear.hpp`, `ml/optim.hpp`.
///
/// **Known issues.** nanochat runs the iteration in bfloat16; here it runs in `T`, or in whatever the backend's
/// products use (snicholls-gpu's `cuda_math::bf16` rounds their operands to bfloat16).
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_core_ml_muon_hpp
#define snicholls_core_ml_muon_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/optim.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <utility>

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
///
/// Every step is a call on the backend (TRAINING.md T9): `sum_squares` for the norm, `copy_rows` and `scale_rows` for
/// `X = g / |g|`, three `gemm`s per iteration with the `b A` and `a X` terms as the products' `beta`, and `copy_rows`
/// for the result, so a device library's overloads keep the matrix on the device. A tall matrix is iterated as it is,
/// with `A = Xᵀ X` and `X ← a X + X B`, which is the transpose's iteration term for term (`A` and `B` are symmetric
/// to the bit), so nothing is transposed.
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
    const std::size_t n = rows * cols, r = std::min(rows, cols);
    assert(g.size() == n && out.size() == n && work.size() >= 2 * n + 2 * r * r);
    const bool tall = rows > cols;
    // X ping-pongs between out and x2; A and B are r x r
    const auto x2 = work.first(n), a = work.subspan(n, r * r), bm = work.subspan(n + r * r, r * r);
    const T scale = T(1) / (math::sqrt(sum_squares(backend, g)) + T(1e-7));
    copy_rows(backend, g, out);
    scale_rows(backend, out, scale);
    using ns = muon_newton_schulz<T>;
    const linalg::matrix_view<T, right> av(a.data(), {r, r}), bv(bm.data(), {r, r});
    const linalg::matrix_view<const T, right> ac(a.data(), {r, r}), bc(bm.data(), {r, r});
    // The shapes are right by construction and C never overlaps A or B, so every product succeeds
    const auto product = [&](T alpha, const auto& lhs, const auto& rhs, T beta, const auto& c) {
        [[maybe_unused]] const auto done = linalg::gemm(backend, alpha, lhs, rhs, beta, c);
        assert(done.is_ok());
    };
    std::span<T> x = out, next = x2;
    for (int step = 0; step < ns::steps; ++step) {
        const linalg::matrix_view<const T, right> xc(x.data(), {rows, cols});
        const linalg::matrix_view<T, right> nv(next.data(), {rows, cols});
        if (tall)
            product(T(1), linalg::transposed(xc), xc, T(0), av);   // A = Xᵀ X
        else
            product(T(1), xc, linalg::transposed(xc), T(0), av);   // A = X Xᵀ
        copy_rows(backend, std::span<const T>(a), bm);
        product(ns::c, ac, ac, ns::b, bv);                          // B = b A + c A²
        copy_rows(backend, std::span<const T>(x), next);
        if (tall)
            product(T(1), xc, bc, ns::a, nv);                       // X ← a X + X B
        else
            product(T(1), bc, xc, ns::a, nv);                       // X ← a X + B X
        std::swap(x, next);
    }
    if (x.data() != out.data())
        copy_rows(backend, std::span<const T>(x), out);
}

/// \brief One Muon update of a weight matrix, every step on the backend.
///
/// `momentum ← μ momentum + g` and Nesterov's `g + μ momentum` are `scale_rows`, `add_rows` and `copy_rows`; the
/// orthogonalisation is `muon_orthogonalise`; `w ← w − lr · sqrt(max(1, rows / cols)) · X` is `scale_rows` and
/// `add_rows`. With core's reference backend the result is the elementwise loops' to the bit; with a device library's
/// overloads (snicholls-gpu's `cuda_training_backend`) `w`, `g`, `momentum` and `work` stay in device memory.
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
    const auto update = work.first(n), x = work.subspan(n, n), rest = work.subspan(2 * n);
    scale_rows(backend, momentum, config.momentum);   // momentum = g + μ momentum
    add_rows(backend, g, momentum);
    copy_rows(backend, std::span<const T>(momentum), update);
    if (config.nesterov) {   // update = g + μ momentum
        scale_rows(backend, update, config.momentum);
        add_rows(backend, g, update);
    }
    muon_orthogonalise(backend, std::span<const T>(update), rows, cols, x, rest);
    scale_rows(backend, x, -(config.lr * math::sqrt(std::max(T(1), T(rows) / T(cols)))));   // w = -step X + w
    add_rows(backend, std::span<const T>(x), w);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_muon_hpp
