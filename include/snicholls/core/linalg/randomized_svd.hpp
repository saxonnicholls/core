//
//  randomized_svd.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::linalg::truncated_svd`: the leading `k` singular values and vectors of a matrix by randomised range finding (Halko, Martinsson and Tropp), on any `linalg` backend.
/// \ingroup core
///
/// **Overview.** A factor model, a PCA of returns or a low-rank compression wants the largest `k` singular triplets
/// of an `m x n` matrix, not all of them. The exact SVD (`linalg::svd`, one-sided Jacobi) costs about `6 m n^2` per
/// sweep; this costs a few matrix products with `l = k + oversample` columns, `(2q + 2) m n l` for `q` power
/// iterations, plus a QR of `m x l` and the exact SVD of `n x l`:
///
/// 1. `Y = A Ω` for an `n x l` random test matrix `Ω`, filled by the sketch (Gaussian by default, from its seed: the same
///    seed, the same bits);
/// 2. `q` times: `Y = A (Aᵀ Q(Y))`, each product re-orthonormalised by QR, so the leading directions dominate;
/// 3. `Q = Q(Y)`, an orthonormal basis of (nearly) `A`'s leading range;
/// 4. the exact SVD of the small `Aᵀ Q = U_b S V_bᵀ` (`n x l`), so `A ≈ Q Qᵀ A = (Q V_b) S U_bᵀ`;
/// 5. `U = Q V_b`, `V = U_b`, the first `k` columns, `s` descending.
///
/// The error is close to the best rank-`k` error `σ_{k+1}` when the spectrum decays; with a flat spectrum, more power
/// iterations help. When `l` reaches `min(m, n)` the result is the exact SVD's, to rounding.
///
/// - `truncated_svd(backend, a, s, u, v, work, options, sketch)`: `s` (`k`), `u` (`m x k`), `v` (`n x k`); `A` is not
///   changed.
/// - `truncated_svd_work_size(m, n, k, options)`: the working space, in elements, so nothing is allocated.
/// - The randomness is a seam: a **sketch** is anything callable as `sketch(omega, rows, cols)` that fills the
///   `rows x cols` test matrix (row-major, in the caller's `T`). `gaussian_sketch<Normal>{seed}` is the default (the analysis's
///   choice), its Gaussian sampler itself interchangeable (`stats/normal.hpp`); `rademacher_sketch{seed}` draws ±1 (cheaper, as accurate in practice). A sketch of the caller's own (another
///   generator, quasi-random points, a device's generator, a structured transform) drops in without touching the
///   algorithm.
///
/// \snippet linalg/randomized_svd_demo.cpp truncated_svd
///
/// **Relationships.** `linalg/backend.hpp` (`gemm`, `qr_factor`, `qr_q`, `svd`: the operations it is written in),
/// `linalg/reference.hpp` (core's backend), `stats/normal.hpp` (the Gaussian samplers), `stats/random.hpp`
/// (`wyrand`). Group: \ref core.
///
/// **Implementations.** One algorithm, over the backend: a backend with fast products (a BLAS, a GPU) makes it fast;
/// it needs `blas_backend`, `factorisation_backend` and `decomposition_backend`. Two sketches: `gaussian_sketch` (any
/// `stats::normal_sampler` over `wyrand`: the inverse CDF by default, Box-Muller) and `rademacher_sketch` (one bit of `wyrand` per entry).
///
/// **Guarantees.** `constexpr` with a `constexpr` backend, no allocation (the caller's `truncated_svd_work_size`
/// elements), deterministic for a given seed and backend. `U` and `V` have orthonormal columns; `s` is descending and
/// non-negative. A zero matrix gives zero singular values.
///
/// **Errors.** `error::dimension_mismatch` when `k > min(m, n)`, `s` is not `k`, `u` is not `m x k`, `v` is not
/// `n x k`, or `work` is short. When `A` holds a NaN or an infinity, the backend's QR or SVD reports it: core's reference
/// reports `error::singular` (the first QR's pivot) or `error::did_not_converge` (the small SVD).
/// `k = 0` succeeds and writes nothing.
///
/// **Performance.** Against the exact `svd` of the same matrix through the reference backend, `k = 10`, a decaying
/// spectrum (`benchmarks/linalg/randomized_svd_benchmark.cpp`; a different algorithm, so the fairness check is that the
/// leading 10 singular values agree to 1e-9 relative; Release, Apple clang 17, Xeon W-3245, macOS 26, 2026-11-16):
///
/// | Matrix | `truncated_svd` | `svd` |
/// |---|---|---|
/// | 500 x 100 | 7.7 ms | 145 ms |
/// | 2000 x 200 | 55 ms | 3.56 s |
///
/// 19 and 64 times faster; the gap grows with `n`. Nearly all of the time is in `gemm`, so a faster backend (a BLAS,
/// snicholls-gpu's OpenCL backend once it carries the factorisations) speeds it in proportion.
///
/// **Tests and demos.** `tests/linalg/randomized_svd_test.cpp`, `demos/linalg/randomized_svd_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `result.hpp`,
/// `linalg/backend.hpp`, `stats/normal.hpp`, `stats/random.hpp`.
///
/// **Known issues.** A slowly decaying spectrum needs more power iterations (or more oversampling) for the trailing
/// directions of the `k`; the leading ones are accurate first. Sums accumulate in `T`.
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_core_linalg_randomized_svd_hpp
#define snicholls_core_linalg_randomized_svd_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::linalg {

    /// \brief How `truncated_svd` samples.
    /// \ingroup core
    struct truncated_svd_options {
        std::size_t oversample = 10;   ///< Extra columns beyond `k`: `l = min(k + oversample, m, n)`.
        int power_iterations = 2;      ///< `q`: products with `A Aᵀ`, each sharpening the leading directions.
    };

    /// \brief A source of random test matrices for `truncated_svd`: `s(omega, rows, cols)` fills `rows x cols`, row-major.
    /// \ingroup core
    /// \tparam S The sketch.
    /// \tparam T The number type.
    template <typename S, typename T>
    concept sketch = requires(const S& s, std::span<T> omega, std::size_t rows, std::size_t cols) { s(omega, rows, cols); };

    /// \brief Gaussian test matrices, in row-major order, from a `stats::normal_sampler` over `stats::wyrand`.
    /// \ingroup core
    /// \tparam Normal The Gaussian sampler: `stats::normal_inverse_cdf` (the default), `stats::normal_box_muller`, or the
    /// caller's own.
    template <typename Normal = stats::normal_inverse_cdf>
        requires stats::normal_sampler<Normal, stats::wyrand>
    struct gaussian_sketch {
        std::uint64_t seed = 0;   ///< The same seed and sampler, the same matrix.
        Normal normal{};          ///< The sampler, copied per fill so a stateful one starts afresh each time.

        /// \brief Fills the test matrix. \param omega `rows · cols` elements. \param rows Its rows. \param cols Its columns.
        template <scalar T>
        constexpr void operator()(std::span<T> omega, std::size_t rows, std::size_t cols) const
        {
            stats::wyrand g(seed);
            Normal draw = normal;
            for (T& e : omega.first(rows * cols))
                e = static_cast<T>(draw(g));
        }
    };

    /// \brief Rademacher test matrices: each entry +1 or -1 with probability one half, one bit of `stats::wyrand` each.
    /// \ingroup core
    struct rademacher_sketch {
        std::uint64_t seed = 0;   ///< The same seed, the same matrix, on every platform.

        /// \brief Fills the test matrix. \param omega `rows · cols` elements. \param rows Its rows. \param cols Its columns.
        template <scalar T>
        constexpr void operator()(std::span<T> omega, std::size_t rows, std::size_t cols) const
        {
            stats::wyrand g(seed);
            std::uint64_t bits = 0;
            for (std::size_t i = 0; i < rows * cols; ++i) {
                if (i % 64 == 0)
                    bits = g();
                omega[i] = (bits >> (i % 64)) & 1 ? T(1) : T(-1);
            }
        }
    };

    /// \brief The working space `truncated_svd` needs, in elements.
    /// \param m The rows.
    /// \param n The columns.
    /// \param k The triplets wanted.
    /// \param options The sampling (its `oversample`).
    /// \return `2 n l + 2 m l + l² + 2 l` for `l = min(k + oversample, m, n)`; 0 for `k = 0`.
    [[nodiscard]] constexpr std::size_t truncated_svd_work_size(std::size_t m, std::size_t n, std::size_t k,
                                                                const truncated_svd_options& options = {}) noexcept
    {
        const std::size_t l = k == 0 ? 0 : std::min({k + options.oversample, m, n});
        return 2 * n * l + 2 * m * l + l * l + 2 * l;
    }

    namespace detail::randomized {
        // The shapes are right by construction and no output overlaps an input, so a product cannot fail
        template <typename B, typename T, typename X, typename Y, typename C>
        constexpr void product(const B& b, const X& x, const Y& y, const C& c)
        {
            [[maybe_unused]] const auto done = gemm(b, T(1), x, y, T(0), c);
        }

        // Q (m x l, orthonormal columns) of y, which is overwritten by its factors
        template <typename B, typename T>
        constexpr result<void, error> orthonormalise(const B& b, matrix_view<T, layout_right> y, std::span<T> tau,
                                                     matrix_view<T, layout_right> q)
        {
            if (auto f = qr_factor(b, y, tau); f.is_err())
                return f;
            return qr_q(b, matrix_view<const T, layout_right>(y), std::span<const T>(tau), q);
        }
    } // namespace detail::randomized

    /// \brief The leading `k` singular values and vectors of `A`, `A ≈ U diag(s) Vᵀ`, by randomised range finding.
    /// \ingroup core
    /// \tparam B  The backend: a `blas_backend`, `factorisation_backend` and `decomposition_backend` for `T`.
    /// \param b       The backend.
    /// \param a       `m x n`, any layout; not changed.
    /// \param s       Receives the `k` singular values, descending.
    /// \param u       Receives the `k` left singular vectors, `m x k`.
    /// \param v       Receives the `k` right singular vectors, `n x k`.
    /// \param work    At least `truncated_svd_work_size(m, n, k, options)` elements.
    /// \param options The oversampling and the power iterations.
    /// \param sk      The source of the random test matrix: `gaussian_sketch<>{}` by default.
    /// \return Nothing, or `error::dimension_mismatch`, or the QR's or the small SVD's error for a non-finite `A`.
    template <typename B, typename EA, typename LA, scalar T, typename LU, typename LV, typename S = gaussian_sketch<>>
        requires(std::same_as<detail::value_t<EA>, T> && blas_backend<B, T> && factorisation_backend<B, T> && decomposition_backend<B, T>
                 && sketch<S, T>)
    constexpr result<void, error> truncated_svd(const B& b, const md_span<EA, 2, LA>& a, std::span<T> s, const md_span<T, 2, LU>& u,
                                                const md_span<T, 2, LV>& v, std::span<T> work, const truncated_svd_options& options = {},
                                                const S& sk = S{})
    {
        using right = layout_right;
        const std::size_t m = a.extent(0), n = a.extent(1), k = s.size();
        if (k > std::min(m, n) || u.extent(0) != m || u.extent(1) != k || v.extent(0) != n || v.extent(1) != k
            || work.size() < truncated_svd_work_size(m, n, k, options))
            return err(error::dimension_mismatch);
        if (k == 0)
            return ok();
        const std::size_t l = std::min({k + options.oversample, m, n});
        std::size_t at = 0;
        auto take = [&](std::size_t size) {
            const auto part = work.subspan(at, size);
            at += size;
            return part;
        };
        const auto n1 = take(n * l), n2 = take(n * l), y = take(m * l), q = take(m * l), vb = take(l * l), sl = take(l), tau = take(l);
        const matrix_view<const T> av(a);
        const matrix_view<T, right> n1v(n1.data(), {n, l}), n2v(n2.data(), {n, l}), yv(y.data(), {m, l}), qv(q.data(), {m, l}),
            vbv(vb.data(), {l, l});
        const matrix_view<const T, right> n1c(n1.data(), {n, l}), n2c(n2.data(), {n, l}), qc(q.data(), {m, l});
        using detail::randomized::orthonormalise;
        using detail::randomized::product;

        // 1. Y = A Ω, Ω from the sketch
        sk(n1, n, l);
        product<B, T>(b, av, n1c, yv);

        // 2. Power iterations: Y = A Q(Aᵀ Q(Y))
        for (int it = 0; it < options.power_iterations; ++it) {
            if (auto r = orthonormalise(b, yv, tau, qv); r.is_err())
                return r;
            product<B, T>(b, transposed(av), qc, n1v);   // Z = Aᵀ Q, n x l
            if (auto r = orthonormalise(b, n1v, tau, n2v); r.is_err())
                return r;
            product<B, T>(b, av, n2c, yv);               // Y = A Q(Z)
        }

        // 3. Q, then 4. the exact SVD of Aᵀ Q = U_b S V_bᵀ (n x l, n >= l)
        if (auto r = orthonormalise(b, yv, tau, qv); r.is_err())
            return r;
        product<B, T>(b, transposed(av), qc, n1v);
        if (auto r = svd(b, n1v, sl, vbv); r.is_err())
            return r;

        // 5. U = Q V_b and V = U_b, the first k columns; s the first k values
        product<B, T>(b, qc, matrix_view<const T>(vb.data(), {l, k}, {l, 1}), u);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < k; ++j)
                v(i, j) = n1[i * l + j];
        std::copy_n(sl.begin(), k, s.begin());
        return ok();
    }

} // namespace snicholls::linalg

#endif /* snicholls_core_linalg_randomized_svd_hpp */
