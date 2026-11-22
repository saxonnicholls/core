//
//  discrete.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::tracking::hmm_predict`, `hmm_update` and `hmm_predict_measurement`: the Bayes filter of a hidden Markov model, for a target's class or mode.
/// \ingroup core
///
/// **Overview.** Some of a target's state is a category, not a number: a swimmer above the surface, at it, or
/// not visible; a manoeuvre mode; an object class. Its distribution is a probability vector `p` over `K` hidden
/// categories, propagated by a column-stochastic transition matrix and corrected by an emission matrix:
///
/// - `hmm_predict(p, A)`: `normalise(A p)`, `A[i][j]` the probability of moving from `j` to `i`.
/// - `hmm_predict_measurement(p, E)`: `normalise(E p)`, the predicted distribution of the `L` observable
///   categories, `E[l][k]` the probability of observing `l` in hidden category `k`.
/// - `hmm_update(p, z, E)`: `normalise((Eᵀ z) ⊙ p)`, for an observation `z` that is one-hot or itself a
///   distribution (a classifier's soft output).
/// - `hmm_likelihood(p, z, E)`: `zᵀ E p`, the observation's marginal likelihood, for association.
/// - `normalise_columns(A)`: columns summed to 1, as Stone Soup normalises its matrices.
///
/// Conventions are Stone Soup's (`MarkovianTransitionModel`, `MarkovianMeasurementModel`, `HMMPredictor`,
/// `HMMUpdater`).
///
/// \snippet tracking/discrete_demo.cpp discrete
///
/// **Relationships.** `linalg/fixed.hpp`, `result.hpp`. Combined with a `gaussian` kinematic state in the framework above core.
/// Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` for the standard floating-point types, no allocation. Results are
/// non-negative and sum to 1.
///
/// **Errors.** `hmm_error::zero_probability` when the result would be all zero: the observation is impossible under
/// every hidden category the prior allows.
///
/// **Performance.** Not benchmarked: a `K x K` or `L x K` product.
///
/// **Tests and demos.** `tests/tracking/discrete_test.cpp`, `demos/tracking/discrete_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `linalg/fixed.hpp`, `result.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_tracking_discrete_hpp
#define snicholls_core_tracking_discrete_hpp

#include <snicholls/core/linalg/fixed.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <cstdint>

namespace snicholls::tracking {

    /// \brief Why a discrete update has no answer.
    /// \ingroup core
    enum class hmm_error : std::uint8_t {
        zero_probability = 1,  ///< Every category has probability zero after the step.
    };

    /// \brief A vector scaled to sum to 1. \tparam T The number type. \tparam K The categories.
    /// \ingroup core
    /// \param p Non-negative values. \return `p / Σ p`, or `hmm_error::zero_probability`.
    template <linalg::scalar T, std::size_t K>
    [[nodiscard]] constexpr result<linalg::fixed_vector<T, K>, hmm_error> normalise(const linalg::fixed_vector<T, K>& p) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        T s(0);
        for (std::size_t i = 0; i < K; ++i)
            s += p[i];
        if (!(s > T(0)))
            return err(hmm_error::zero_probability);
        return ok(p / s);
    }

    /// \brief A matrix with each column scaled to sum to 1 (a zero column is left zero). \ingroup core
    /// \tparam T The number type. \tparam R The rows. \tparam C The columns. \param a The matrix. \return It, normalised.
    template <linalg::scalar T, std::size_t R, std::size_t C>
    [[nodiscard]] constexpr linalg::fixed_matrix<T, R, C> normalise_columns(const linalg::fixed_matrix<T, R, C>& a) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        auto m = a;
        for (std::size_t j = 0; j < C; ++j) {
            T s(0);
            for (std::size_t i = 0; i < R; ++i)
                s += a(i, j);
            if (s > T(0))
                for (std::size_t i = 0; i < R; ++i)
                    m(i, j) = a(i, j) / s;
        }
        return m;
    }

    /// \brief The predicted distribution, `normalise(A p)`. \tparam T The number type. \tparam K The categories.
    /// \ingroup core
    /// \param p The prior. \param a The transition matrix, `a(i, j)` = P(j → i). \return The prediction, or `hmm_error::zero_probability`.
    template <linalg::scalar T, std::size_t K>
    [[nodiscard]] constexpr result<linalg::fixed_vector<T, K>, hmm_error> hmm_predict(const linalg::fixed_vector<T, K>& p, const linalg::fixed_matrix<T, K, K>& a) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return normalise(a * p);
    }

    /// \brief The predicted observation distribution, `normalise(E p)`. \tparam T The number type.
    /// \ingroup core
    /// \tparam L The observable categories. \tparam K The hidden categories. \param p The predicted distribution.
    /// \param e The emission matrix, `e(l, k)` = P(observe l | k). \return The distribution, or `hmm_error::zero_probability`.
    template <linalg::scalar T, std::size_t L, std::size_t K>
    [[nodiscard]] constexpr result<linalg::fixed_vector<T, L>, hmm_error> hmm_predict_measurement(const linalg::fixed_vector<T, K>& p, const linalg::fixed_matrix<T, L, K>& e) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return normalise(e * p);
    }

    /// \brief The posterior, `normalise((Eᵀ z) ⊙ p)`. \tparam T The number type.
    /// \ingroup core
    /// \tparam L The observable categories. \tparam K The hidden categories. \param p The predicted distribution.
    /// \param z The observation, one-hot or a distribution. \param e The emission matrix.
    /// \return The posterior, or `hmm_error::zero_probability`.
    template <linalg::scalar T, std::size_t L, std::size_t K>
    [[nodiscard]] constexpr result<linalg::fixed_vector<T, K>, hmm_error> hmm_update(const linalg::fixed_vector<T, K>& p, const linalg::fixed_vector<T, L>& z,
                                                                                   const linalg::fixed_matrix<T, L, K>& e) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        const auto like = linalg::transpose(e) * z;
        linalg::fixed_vector<T, K> q{};
        for (std::size_t k = 0; k < K; ++k)
            q[k] = like[k] * p[k];
        return normalise(q);
    }

    /// \brief The observation's marginal likelihood, `zᵀ E p`. \tparam T The number type.
    /// \ingroup core
    /// \tparam L The observable categories. \tparam K The hidden categories. \param p The predicted distribution.
    /// \param z The observation. \param e The emission matrix. \return The likelihood.
    template <linalg::scalar T, std::size_t L, std::size_t K>
    [[nodiscard]] constexpr T hmm_likelihood(const linalg::fixed_vector<T, K>& p, const linalg::fixed_vector<T, L>& z, const linalg::fixed_matrix<T, L, K>& e) noexcept(linalg::detail::nothrow_scalar<T>)
    {
        return linalg::dot(z, e * p);
    }

} // namespace snicholls::tracking

#endif /* snicholls_core_tracking_discrete_hpp */
