//
//  collapse.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml` collapse safeguards for joint-embedding training: `embedding_moments` (mergeable per-dimension standard deviation, covariance and effective rank), VICReg's variance and covariance terms, and LeJEPA's SIGReg (sketched Epps-Pulley), each with its gradient.
/// \ingroup core
///
/// **Overview.** A joint-embedding objective has a trivial minimum: an encoder that maps every input to one point
/// predicts itself perfectly. JEPAs avoid it with an EMA target and a narrow predictor (`ml/jepa.hpp`), and watch for it
/// with diagnostics; other methods add a regulariser that makes collapse expensive. This header has both (TRAINING.md
/// T11).
///
/// **Diagnostics.** `embedding_moments<T>` accumulates rows of embeddings (`add`, `add_rows`) into a count, a mean
/// and the co-moment (Welford's update), over views the caller owns: the full `width x width` co-moment, or only its
/// diagonal for the variances alone. Two accumulators `merge` (Chan, Golub and LeVeque's update), so workers and
/// batches combine. From it: `variance` and `standard_deviation` per dimension (unbiased, `n - 1`), whose collapse to 0
/// is the first sign of trouble (V-JEPA's `pstd`); `covariance`; and `effective_rank` (Roy and Vetterli, 2007:
/// `exp(-Σ p_k log p_k)` with `p_k = σ_k / Σ σ`, `σ_k = sqrt(λ_k)` from the covariance's eigenvalues, the singular
/// values of the centred embeddings up to a factor; RankMe, Garrido et al. 2023, takes the singular values of the
/// embeddings uncentred), 1 for a collapse to a line and `width` for isotropic embeddings.
///
/// **VICReg** (Bardes, Ponce and LeCun, 2022; read from its code 2026-11-19): over a batch `X` of `n` rows of `d`,
/// centred `X̃ = X - mean`,
/// - `vicreg_variance_rows`: `v = (1/d) Σ_j max(0, γ - sqrt(Var(x_j) + ε))`, `Var` unbiased, `γ = 1`, `ε = 1e-4`;
/// - `vicreg_covariance_rows`: `c = (1/d) Σ_{i≠j} C_ij²`, `C = X̃ᵀ X̃ / (n - 1)`;
/// each with its gradient added, times a weight (VICReg's coefficients: 25 on the invariance and variance terms, 1 on
/// the covariance, for a loss over two branches each).
///
/// **SIGReg** (Balestriero and LeCun, 2025, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the
/// Heuristics", and its code, read 2026-11-19): embeddings should be distributed as an isotropic Gaussian, and SIGReg
/// tests that along `M` random unit directions `a` (`sigreg_directions`, redrawn every step from `(seed, step)`, the
/// same on every worker) with the Epps-Pulley statistic of the 1-D projections `z_n = a·x_n`:
/// `EP = N ∫ |φ̂(t) - e^(-t²/2)|² e^(-t²/2) dt`, `φ̂(t) = (1/N) Σ_n e^(i t z_n)`, integrated by the trapezoid rule on
/// `[-t_max, t_max]` through the integrand's symmetry, `K` knots on `[0, t_max]` (`sigreg_quadrature`, 17 knots and
/// `t_max = 3` by default, the code's), and averaged over the directions. The empirical characteristic function is a
/// mean, so the statistic splits into sums that merge across workers (an all-reduce): `sigreg_moments_rows` adds each
/// row's `cos(t_k z_n)` and `sin(t_k z_n)` into `sums`, `sigreg_value` reads the statistic from the merged sums and the
/// global count, and `sigreg_backward_rows` adds its gradient, `∂EP/∂z_n = Σ_k 2 w_k t_k (s̄_k cos(t_k z_n) - (c̄_k -
/// φ_k) sin(t_k z_n))` taken along each direction. LeJEPA's loss is `(1 - λ) · prediction + λ · SIGReg`, `λ = 0.05`,
/// its prediction term the squared distance of each view's embedding to the mean of the global views'
/// (`jepa_loss_rows` with `jepa_loss::l2`).
///
/// \snippet ml/collapse_demo.cpp moments
///
/// \snippet ml/collapse_demo.cpp sigreg
///
/// **Relationships.** `ml/jepa.hpp`, `linalg/backend.hpp` (`symmetric_eigen` for the effective rank),
/// `stats/normal.hpp` (`normal_inverse_cdf` for the directions), `ml/diloco.hpp` (`ordered_sum`, the merge of `sums`
/// across workers). Group: \ref core.
///
/// **Implementations.** One each, on the host for any backend; a device library overloads the `_rows` functions.
///
/// **Guarantees.** `constexpr`, no allocation (the caller's spans). Sums run in row order, then channel order, as
/// written. `merge` of two halves equals one pass up to rounding.
///
/// **Errors.** Preconditions, asserted: the sizes of the spans; at least two rows for a variance. `effective_rank`
/// returns the backend's `linalg::error` if the eigensolver fails.
///
/// **Performance.** `vicreg_covariance_rows` and `sigreg_moments_rows` against the loops by hand
/// (`benchmarks/ml/collapse_benchmark.cpp`), 256 embeddings of 128 floats, SIGReg over 64 directions and 17 knots: the
/// covariance term with its gradient within 10% (4.9 ms), the SIGReg sums level (3.6 ms, its cosines and sines) (Xeon W-3245, Release, Apple clang 17, 2026-11-19, one thread, a machine at load average 60 to 80: differences within about 15% are noise).
/// The covariance term is O(n d²); SIGReg O(n d M + n M K).
///
/// **Tests and demos.** `tests/ml/collapse_test.cpp`, `demos/ml/collapse_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `linalg/backend.hpp`,
/// `snicholls/core/math/constexpr_math.hpp`, `snicholls/core/result.hpp`, `stats/normal.hpp`, `stats/random.hpp`.
///
/// **Known issues.** The LeJEPA paper's text gives the weight as `e^(-t²/σ²)` with `σ = 1`; its code uses `e^(-t²/2)`,
/// as here. Its code multiplies by the world size as well as the local count; `count` here is the global count.
///
/// **Status.** Experimental, new in 0.16.0.

#ifndef snicholls_core_ml_collapse_hpp
#define snicholls_core_ml_collapse_hpp

#include <snicholls/core/linalg/backend.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/stats/normal.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace snicholls::ml
{

/// \brief A mergeable accumulator of embedding rows: their count, mean and co-moment, over views the caller owns.
/// \ingroup core
/// \tparam T The number type.
template <math::math_number T>
struct embedding_moments
{
    std::span<T> mean{};       ///< `width`: the mean of the rows added; zero to start.
    std::span<T> comoment{};   ///< `width · width` (`Σ (x - mean)(x - mean)ᵀ`), or `width` (its diagonal only); zero to start.
    std::size_t count = 0;     ///< The rows added.

    /// \brief Whether the full co-moment is kept. \return `comoment.size() != mean.size()`.
    [[nodiscard]] constexpr bool full() const noexcept { return comoment.size() != mean.size(); }

    /// \brief Adds one row (Welford): `δ = x - mean`, `mean += δ / n`, `M_ij += δ_i (x_j - mean_j)`.
    /// \param x The row, `width`.
    constexpr void add(std::span<const T> x)
    {
        const std::size_t d = mean.size();
        assert(x.size() == d && (comoment.size() == d || comoment.size() == d * d));
        ++count;
        const T n = T(count);
        if (full()) {
            // M_ij += δ_i · (x_j - mean_j) with the new mean; δ is kept in the row's own pass
            for (std::size_t i = 0; i < d; ++i) {
                const T di = x[i] - mean[i];
                for (std::size_t j = 0; j < d; ++j) {
                    const T newj = mean[j] + (x[j] - mean[j]) / n;
                    comoment[i * d + j] += di * (x[j] - newj);
                }
            }
            for (std::size_t j = 0; j < d; ++j)
                mean[j] += (x[j] - mean[j]) / n;
        } else {
            for (std::size_t j = 0; j < d; ++j) {
                const T dj = x[j] - mean[j];
                mean[j] += dj / n;
                comoment[j] += dj * (x[j] - mean[j]);
            }
        }
    }

    /// \brief Adds rows of `width`, in order. \param rows The rows, end to end.
    constexpr void add_rows(std::span<const T> rows)
    {
        const std::size_t d = mean.size();
        assert(d > 0 && rows.size() % d == 0);
        for (std::size_t r = 0; r < rows.size() / d; ++r)
            add(rows.subspan(r * d, d));
    }

    /// \brief Merges another accumulator of the same width and shape (Chan et al.): as if this one had seen its rows.
    /// \param o The other.
    constexpr void merge(const embedding_moments& o)
    {
        const std::size_t d = mean.size();
        assert(o.mean.size() == d && o.comoment.size() == comoment.size());
        if (o.count == 0)
            return;
        const T na = T(count), nb = T(o.count), n = na + nb, f = na * nb / n;
        for (std::size_t i = 0; i < d; ++i) {
            const T di = o.mean[i] - mean[i];
            if (full())
                for (std::size_t j = 0; j < d; ++j)
                    comoment[i * d + j] += o.comoment[i * d + j] + di * (o.mean[j] - mean[j]) * f;
            else
                comoment[i] += o.comoment[i] + di * di * f;
        }
        for (std::size_t i = 0; i < d; ++i)
            mean[i] += (o.mean[i] - mean[i]) * nb / n;
        count += o.count;
    }

    /// \brief The unbiased variance of each dimension, `M_jj / (n - 1)`. \param out `width` values.
    constexpr void variance(std::span<T> out) const
    {
        const std::size_t d = mean.size();
        assert(out.size() == d && count >= 2);
        for (std::size_t j = 0; j < d; ++j)
            out[j] = comoment[full() ? j * d + j : j] / T(count - 1);
    }

    /// \brief The unbiased standard deviation of each dimension. \param out `width` values.
    constexpr void standard_deviation(std::span<T> out) const
    {
        variance(out);
        for (T& v : out)
            v = math::sqrt(v);
    }

    /// \brief The unbiased covariance, `M / (n - 1)` (full co-moment only). \param out `width · width` values.
    constexpr void covariance(std::span<T> out) const
    {
        assert(full() && out.size() == comoment.size() && count >= 2);
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = comoment[i] / T(count - 1);
    }

    /// \brief The effective rank, `exp(-Σ p_k log p_k)` with `p_k = sqrt(λ_k) / Σ sqrt(λ)` over the covariance's
    /// eigenvalues `λ` (negative ones, rounding, taken as 0), through the backend's `symmetric_eigen`.
    /// \param backend A `linalg::decomposition_backend` (`linalg::reference`).
    /// \param work `2 · width² + width` values of working space.
    /// \return The rank in `[1, width]`, 0 if every eigenvalue is 0, or the eigensolver's error.
    template <typename B>
    constexpr result<T, linalg::error> effective_rank(const B& backend, std::span<T> work) const
    {
        const std::size_t d = mean.size();
        assert(work.size() == 2 * d * d + d);
        covariance(work.first(d * d));
        using view = linalg::matrix_view<T, linalg::layout_right>;
        const auto done = linalg::symmetric_eigen(backend, view(work.data(), {d, d}), work.subspan(2 * d * d, d),
                                                  view(work.data() + d * d, {d, d}));
        if (!done)
            return err(done.unwrap_err_unchecked());
        const auto w = work.subspan(2 * d * d, d);
        T total = T(0);
        for (T& l : w) {
            l = l > T(0) ? math::sqrt(l) : T(0);
            total += l;
        }
        if (total == T(0))
            return ok(T(0));
        T h = T(0);
        for (const T s : w)
            if (s > T(0)) {
                const T p = s / total;
                h -= p * math::log(p);
            }
        return ok(math::exp(h));
    }
};

namespace detail::collapse
{
    // Each column's mean of n rows of d, the rows summed in order then divided
    template <typename T>
    constexpr void column_means(std::span<const T> x, std::size_t d, std::span<T> mean)
    {
        const std::size_t n = x.size() / d;
        for (std::size_t j = 0; j < d; ++j)
            mean[j] = T(0);
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t j = 0; j < d; ++j)
                mean[j] += x[r * d + j];
        for (std::size_t j = 0; j < d; ++j)
            mean[j] /= T(n);
    }
} // namespace detail::collapse

/// \brief VICReg's variance term over a batch, its gradient added: `weight · (1/d) Σ_j max(0, γ - sqrt(Var(x_j) + ε))`.
///
/// `Var` is unbiased: `Σ_r (x_rj - mean_j)² / (n - 1)`, rows in order. The gradient of an element is
/// `-weight / d · [γ > std_j] · (x_rj - mean_j) / ((n - 1) std_j)`.
/// \param x `n` rows of `width`, `n >= 2`.
/// \param width The dimensions.
/// \param dx Added into; `x`'s size, or empty.
/// \param work `2 · width` values of working space.
/// \param weight The term's coefficient.
/// \param gamma The target standard deviation, VICReg's 1.
/// \param eps Added to the variance, VICReg's 1e-4.
/// \return The term.
template <typename B, math::math_number T>
constexpr T vicreg_variance_rows(const B&, std::type_identity_t<std::span<const T>> x, std::size_t width, std::span<T> dx,
                                 std::type_identity_t<std::span<T>> work, std::type_identity_t<T> weight = T(1),
                                 std::type_identity_t<T> gamma = T(1), std::type_identity_t<T> eps = T(1e-4))
{
    const std::size_t d = width, n = width == 0 ? 0 : x.size() / width;
    assert(d > 0 && x.size() == n * d && n >= 2 && (dx.empty() || dx.size() == x.size()) && work.size() == 2 * d);
    const auto mean = work.first(d), sd = work.last(d);
    detail::collapse::column_means(x, d, mean);
    for (std::size_t j = 0; j < d; ++j)
        sd[j] = T(0);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t j = 0; j < d; ++j) {
            const T c = x[r * d + j] - mean[j];
            sd[j] += c * c;
        }
    T total = T(0);
    for (std::size_t j = 0; j < d; ++j) {
        sd[j] = math::sqrt(sd[j] / T(n - 1) + eps);
        total += gamma - sd[j] > T(0) ? gamma - sd[j] : T(0);
    }
    if (!dx.empty())
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t j = 0; j < d; ++j)
                if (gamma - sd[j] > T(0))
                    dx[r * d + j] -= weight / T(d) * (x[r * d + j] - mean[j]) / (T(n - 1) * sd[j]);
    return weight * (total / T(d));
}

/// \brief VICReg's covariance term over a batch, its gradient added: `weight · (1/d) Σ_{i≠j} C_ij²`, `C = X̃ᵀ X̃ / (n - 1)`.
///
/// `C` is summed over the rows in order; the term over `i`, then `j`. The gradient of an element is
/// `weight · 4 / (d (n - 1)) · Σ_{j≠b} x̃_aj C_jb`.
/// \param x `n` rows of `width`, `n >= 2`.
/// \param width The dimensions.
/// \param dx Added into; `x`'s size, or empty.
/// \param work `width + width²` values of working space.
/// \param weight The term's coefficient.
/// \return The term.
template <typename B, math::math_number T>
constexpr T vicreg_covariance_rows(const B&, std::type_identity_t<std::span<const T>> x, std::size_t width, std::span<T> dx,
                                   std::type_identity_t<std::span<T>> work, std::type_identity_t<T> weight = T(1))
{
    const std::size_t d = width, n = width == 0 ? 0 : x.size() / width;
    assert(d > 0 && x.size() == n * d && n >= 2 && (dx.empty() || dx.size() == x.size()) && work.size() == d + d * d);
    const auto mean = work.first(d), c = work.subspan(d, d * d);
    detail::collapse::column_means(x, d, mean);
    for (T& v : c)
        v = T(0);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t i = 0; i < d; ++i) {
            const T xi = x[r * d + i] - mean[i];
            for (std::size_t j = 0; j < d; ++j)
                c[i * d + j] += xi * (x[r * d + j] - mean[j]);
        }
    T total = T(0);
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j < d; ++j) {
            c[i * d + j] /= T(n - 1);
            if (i != j)
                total += c[i * d + j] * c[i * d + j];
        }
    if (!dx.empty()) {
        const T g = weight * T(4) / (T(d) * T(n - 1));
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t b = 0; b < d; ++b) {
                T s = T(0);
                for (std::size_t j = 0; j < d; ++j)
                    if (j != b)
                        s += (x[r * d + j] - mean[j]) * c[j * d + b];
                dx[r * d + b] += g * s;
            }
    }
    return weight * (total / T(d));
}

/// \brief SIGReg's quadrature: `K` knots `t_k = k · t_max / (K - 1)` on `[0, t_max]`, the normal's characteristic
/// function `φ_k = e^(-t_k²/2)` there, and the weights `w_k = φ_k · 2 Δt` (`Δt` at the two ends), the trapezoid rule on
/// `[-t_max, t_max]` by symmetry times the Gaussian window.
/// \ingroup core
/// \tparam T The number type. \tparam K The knots, odd, at least 3: 17 in LeJEPA's code.
template <math::math_number T, std::size_t K = 17>
    requires(K >= 3 && K % 2 == 1)
struct sigreg_quadrature
{
    std::array<T, K> t{};        ///< The knots.
    std::array<T, K> phi{};      ///< `e^(-t²/2)` at each.
    std::array<T, K> weight{};   ///< The window times the trapezoid's weight.

    /// \brief Builds the knots. \param t_max The last knot: 3 in LeJEPA's code.
    constexpr explicit sigreg_quadrature(T t_max = T(3))
    {
        const T dt = t_max / T(K - 1);
        for (std::size_t k = 0; k < K; ++k) {
            t[k] = T(k) * dt;
            phi[k] = math::exp(-(t[k] * t[k]) * T(0.5));
            weight[k] = (k == 0 || k == K - 1 ? dt : T(2) * dt) * phi[k];
        }
    }
};

/// \brief SIGReg's `M` random unit directions in `width` dimensions for one step: standard normal components
/// (`stats::normal_inverse_cdf` over `wyrand::stream(seed, step)`, direction by direction), each direction divided by
/// its length.
/// \param seed The run's seed.
/// \param step The step: every worker of the step draws the same directions.
/// \param width The dimensions.
/// \param out Receives `M · width` values, direction by direction.
template <math::math_number T>
constexpr void sigreg_directions(std::uint64_t seed, std::uint64_t step, std::size_t width, std::span<T> out)
{
    assert(width > 0 && out.size() % width == 0);
    stats::wyrand g = stats::wyrand::stream(seed, step);
    const stats::normal_inverse_cdf normal{};
    for (std::size_t a = 0; a < out.size() / width; ++a) {
        const auto v = out.subspan(a * width, width);
        double s = 0;
        for (T& x : v) {
            const double z = normal(g);
            x = static_cast<T>(z);
            s += z * z;
        }
        const double len = math::sqrt(s);
        for (T& x : v)
            x = static_cast<T>(static_cast<double>(x) / len);
    }
}

/// \brief SIGReg's sums for a batch, added into `sums`: for each direction `a` and knot `k`, `Σ_n cos(t_k z_n)` and
/// `Σ_n sin(t_k z_n)`, `z_n = Σ_c x_nc a_c`, rows in order. Sums of different rows (workers) add.
/// \param x `n` rows of `width`.
/// \param width The dimensions.
/// \param directions `M · width`, from `sigreg_directions`.
/// \param q The quadrature.
/// \param sums Added into, `M · K · 2`: direction by direction, knot by knot, the cosine's then the sine's.
template <typename B, math::math_number T, std::size_t K>
constexpr void sigreg_moments_rows(const B&, std::type_identity_t<std::span<const T>> x, std::size_t width,
                                   std::type_identity_t<std::span<const T>> directions, const sigreg_quadrature<T, K>& q,
                                   std::span<T> sums)
{
    assert(width > 0 && x.size() % width == 0 && directions.size() % width == 0 && sums.size() == directions.size() / width * K * 2);
    const std::size_t m = directions.size() / width;
    for (std::size_t r = 0; r < x.size() / width; ++r)
        for (std::size_t a = 0; a < m; ++a) {
            T z = T(0);
            for (std::size_t c = 0; c < width; ++c)
                z += x[r * width + c] * directions[a * width + c];
            for (std::size_t k = 0; k < K; ++k) {
                sums[(a * K + k) * 2] += math::cos(q.t[k] * z);
                sums[(a * K + k) * 2 + 1] += math::sin(q.t[k] * z);
            }
        }
}

/// \brief SIGReg from merged sums: `(1/M) Σ_a N Σ_k w_k ((C_ak/N - φ_k)² + (S_ak/N)²)`.
/// \param sums The sums of every row, `M · K · 2`.
/// \param count `N`, the rows summed (all workers').
/// \param q The quadrature.
/// \return The statistic.
template <math::math_number T, std::size_t K>
[[nodiscard]] constexpr T sigreg_value(std::span<const T> sums, std::size_t count, const sigreg_quadrature<T, K>& q)
{
    assert(count > 0 && sums.size() % (K * 2) == 0);
    const std::size_t m = sums.size() / (K * 2);
    const T n = T(count);
    T total = T(0);
    for (std::size_t a = 0; a < m; ++a) {
        T ep = T(0);
        for (std::size_t k = 0; k < K; ++k) {
            const T c = sums[(a * K + k) * 2] / n - q.phi[k], s = sums[(a * K + k) * 2 + 1] / n;
            ep += q.weight[k] * (c * c + s * s);
        }
        total += n * ep;
    }
    return total / T(m);
}

/// \brief SIGReg's gradient for a batch, added into `dx`: `weight / M · Σ_a a · Σ_k 2 w_k t_k (s̄_ak cos(t_k z) -
/// (c̄_ak - φ_k) sin(t_k z))`, the means from the merged sums.
/// \param x The rows `sigreg_moments_rows` read (this worker's).
/// \param width The dimensions.
/// \param directions The directions.
/// \param q The quadrature.
/// \param sums The merged sums.
/// \param count `N`, the rows of every worker.
/// \param dx Added into, `x`'s size.
/// \param weight SIGReg's coefficient (LeJEPA's `λ`).
template <typename B, math::math_number T, std::size_t K>
constexpr void sigreg_backward_rows(const B&, std::type_identity_t<std::span<const T>> x, std::size_t width,
                                    std::type_identity_t<std::span<const T>> directions, const sigreg_quadrature<T, K>& q,
                                    std::type_identity_t<std::span<const T>> sums, std::size_t count, std::span<T> dx,
                                    std::type_identity_t<T> weight = T(1))
{
    assert(width > 0 && x.size() % width == 0 && dx.size() == x.size() && count > 0);
    const std::size_t m = directions.size() / width;
    assert(sums.size() == m * K * 2);
    const T n = T(count), g = weight / T(m);
    for (std::size_t r = 0; r < x.size() / width; ++r)
        for (std::size_t a = 0; a < m; ++a) {
            T z = T(0);
            for (std::size_t c = 0; c < width; ++c)
                z += x[r * width + c] * directions[a * width + c];
            T dz = T(0);
            for (std::size_t k = 0; k < K; ++k) {
                const T cb = sums[(a * K + k) * 2] / n - q.phi[k], sb = sums[(a * K + k) * 2 + 1] / n;
                dz += T(2) * q.weight[k] * q.t[k] * (sb * math::cos(q.t[k] * z) - cb * math::sin(q.t[k] * z));
            }
            for (std::size_t c = 0; c < width; ++c)
                dx[r * width + c] += g * dz * directions[a * width + c];
        }
}

/// \brief SIGReg on one worker: zeroes `sums`, `sigreg_moments_rows`, `sigreg_value` and, if `dx` is not empty,
/// `sigreg_backward_rows`.
/// \param backend The backend.
/// \param x `n` rows of `width`.
/// \param width The dimensions.
/// \param directions `M · width`.
/// \param q The quadrature.
/// \param sums Working space, `M · K · 2`.
/// \param dx Added into, or empty.
/// \param weight The coefficient, on the value and the gradient.
/// \return `weight ·` the statistic.
template <typename B, math::math_number T, std::size_t K>
constexpr T sigreg_rows(const B& backend, std::type_identity_t<std::span<const T>> x, std::size_t width,
                        std::type_identity_t<std::span<const T>> directions, const sigreg_quadrature<T, K>& q, std::span<T> sums,
                        std::type_identity_t<std::span<T>> dx, std::type_identity_t<T> weight = T(1))
{
    for (T& v : sums)
        v = T(0);
    sigreg_moments_rows(backend, x, width, directions, q, sums);
    const std::size_t n = x.size() / width;
    if (!dx.empty())
        sigreg_backward_rows(backend, x, width, directions, q, std::span<const T>(sums), n, dx, weight);
    return weight * sigreg_value(std::span<const T>(sums), n, q);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_collapse_hpp
