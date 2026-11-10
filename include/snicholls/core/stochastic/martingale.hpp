//
//  martingale.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Martingale tools: the Doob decomposition, the martingale transform, quadratic variation, Markov chains, and tests of the martingale hypothesis.
/// \ingroup core
///
/// **Overview.** A price is a martingale when no information available now predicts its next move. The
/// tools here work on sampled paths, in discrete time:
///
/// - `doob_decomposition`: any adapted path is `X_t = X_0 + M_t + A_t`, a martingale `M` and a predictable
///   part `A` (the compensator), given the conditional expectations a model supplies. `A` is the drift a
///   strategy can harvest; `M` is the noise it cannot. For a supermartingale, `A` decreases: Doob and Meyer.
/// - `martingale_transform`: `(H . X)_t = sum H_k (X_k - X_{k-1})`, the P&L of holding `H_k`, chosen at
///   `k - 1`. Against a martingale it is a martingale: no strategy beats a fair game.
/// - `quadratic_variation`: `[X]_t = sum (X_k - X_{k-1})^2`.
/// - `markov_chain`: finite states and a transition matrix, the setting of `stochastic/snell.hpp`.
/// - `variance_ratio_test` (Lo and MacKinlay, heteroskedasticity-robust) and `ljung_box`: is a series of
///   prices a martingale, or do its increments carry memory?
///
/// \snippet stochastic/martingale_demo.cpp martingale
///
/// **Relationships.** `stochastic/snell.hpp` for optimal stopping; `stats/bootstrap.hpp` for the
/// stationary bootstrap when the increments are dependent. Group: \ref core.
///
/// **Guarantees.** Each function is one pass, O(n); the decomposition, transform and quadratic variation are
/// `constexpr`, usable inside a constant evaluation (`variance_ratio_test`: O(n q)). The variance ratio's z
/// statistic and p-value match Python arch's `VarianceRatio` with its defaults (overlapping, debiased, robust).
///
/// **Errors.** None: sizes that do not match are precondition violations (asserted).
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/stochastic/martingale_test.cpp`, `demos/stochastic/martingale_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cmath>`, `<cstddef>`, `<span>`, `<vector>`, `math/constexpr_math.hpp`.
///
/// **Known issues.** Continuous time is the caller's to discretise.
///
/// **Status.** New in 0.3.0, experimental.

#ifndef snicholls_core_stochastic_martingale_hpp
#define snicholls_core_stochastic_martingale_hpp

#include <snicholls/core/math/constexpr_math.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace snicholls::stochastic {

    /// \brief A path split as `X_t = X_0 + M_t + A_t`: the martingale part and the predictable compensator, both 0 at time 0.
    /// \ingroup core
    struct doob_parts {
        std::vector<double> martingale;    ///< `M_t`, the unpredictable part.
        std::vector<double> compensator;   ///< `A_t`, the sum of predicted moves.
    };

    /// \brief The Doob decomposition of a path, given what was predicted for each step.
    /// \ingroup core
    /// \param x         The path `X_0 .. X_T`.
    /// \param predicted `E[X_t | F_{t-1}]` for `t = 1 .. T`: one fewer than `x`.
    /// \return The parts, each the length of `x`.
    /// \throws std::bad_alloc.
    [[nodiscard]] constexpr doob_parts doob_decomposition(std::span<const double> x, std::span<const double> predicted)
    {
        assert(!x.empty() && predicted.size() + 1 == x.size());
        doob_parts out;
        out.martingale.assign(x.size(), 0.0);
        out.compensator.assign(x.size(), 0.0);
        for (std::size_t t = 1; t < x.size(); ++t) {
            out.compensator[t] = out.compensator[t - 1] + (predicted[t - 1] - x[t - 1]);
            out.martingale[t] = out.martingale[t - 1] + (x[t] - predicted[t - 1]);
        }
        return out;
    }

    /// \brief The martingale transform: the P&L of holding `h[k]` over the move from `x[k-1]` to `x[k]`.
    /// \ingroup core
    /// \param h The holdings: `h[k]` for `k = 1 .. T`, decided at `k - 1`; `h[0]` is ignored.
    /// \param x The path `X_0 .. X_T`, the same length.
    /// \return `(H . X)_t` for each `t`, 0 at time 0.
    /// \throws std::bad_alloc.
    [[nodiscard]] constexpr std::vector<double> martingale_transform(std::span<const double> h, std::span<const double> x)
    {
        assert(h.size() == x.size());
        std::vector<double> out(x.size(), 0.0);
        for (std::size_t t = 1; t < x.size(); ++t)
            out[t] = out[t - 1] + h[t] * (x[t] - x[t - 1]);
        return out;
    }

    /// \brief The quadratic variation of a path: the running sum of squared moves.
    /// \ingroup core
    /// \param x The path.
    /// \return `[X]_t` for each `t`, 0 at time 0.
    /// \throws std::bad_alloc.
    [[nodiscard]] constexpr std::vector<double> quadratic_variation(std::span<const double> x)
    {
        std::vector<double> out(x.size(), 0.0);
        for (std::size_t t = 1; t < x.size(); ++t)
            out[t] = out[t - 1] + (x[t] - x[t - 1]) * (x[t] - x[t - 1]);
        return out;
    }

    /// \brief A time-homogeneous Markov chain on states `0 .. n-1`: `p(i, j)` is the probability of moving from `i` to `j`.
    /// \ingroup core
    class markov_chain {
    public:
        /// \brief A chain from a row-major transition matrix.
        /// \pre Every entry is in [0, 1] and every row sums to 1 (to 1e-9).
        /// \param n           The number of states.
        /// \param transitions `n x n` probabilities, row-major.
        /// \throws std::bad_alloc.
        markov_chain(std::size_t n, std::vector<double> transitions) : n_(n), p_(std::move(transitions))
        {
            assert(p_.size() == n * n);
            for (std::size_t i = 0; i < n; ++i) {
                double row = 0;
                for (std::size_t j = 0; j < n; ++j) {
                    assert(p_[i * n + j] >= 0.0 && p_[i * n + j] <= 1.0);
                    row += p_[i * n + j];
                }
                assert(std::abs(row - 1.0) < 1e-9 && "snicholls::stochastic::markov_chain: a row does not sum to 1");
                static_cast<void>(row);
            }
        }

        /// \brief The number of states.
        /// \return n.
        [[nodiscard]] std::size_t size() const noexcept { return n_; }

        /// \brief The probability of moving from `i` to `j`.
        /// \param i From.
        /// \param j To.
        /// \return It.
        [[nodiscard]] double p(std::size_t i, std::size_t j) const noexcept { return p_[i * n_ + j]; }

        /// \brief `E[f(X_{t+1}) | X_t = i]`: the expectation of a function of the next state.
        /// \param i From.
        /// \param f The values of the function on every state.
        /// \return The expectation.
        [[nodiscard]] double expect(std::size_t i, std::span<const double> f) const noexcept
        {
            double e = 0;
            for (std::size_t j = 0; j < n_; ++j)
                e += p_[i * n_ + j] * f[j];
            return e;
        }

        /// \brief The Doob decomposition of `f` along a path of states: what the chain predicted against what happened.
        /// \param states The path of states.
        /// \param f      The values of the function on every state.
        /// \return The parts of `f(states[t])`.
        /// \throws std::bad_alloc.
        [[nodiscard]] doob_parts doob(std::span<const std::size_t> states, std::span<const double> f) const
        {
            std::vector<double> x(states.size()), predicted(states.empty() ? 0 : states.size() - 1);
            for (std::size_t t = 0; t < states.size(); ++t) {
                x[t] = f[states[t]];
                if (t + 1 < states.size())
                    predicted[t] = expect(states[t], f);
            }
            return doob_decomposition(x, predicted);
        }

    private:
        std::size_t n_;
        std::vector<double> p_;
    };

    /// \brief A test statistic and its p-value.
    /// \ingroup core
    struct test_result {
        double statistic = 0;   ///< The statistic.
        double p_value = 1;     ///< The probability of one at least as extreme under the null hypothesis.
    };

    /// \brief Lo and MacKinlay's variance ratio test of the random walk (martingale) hypothesis, robust to heteroskedasticity.
    ///
    /// Under a martingale the variance of q-step moves is q times that of one-step moves. The statistic is the
    /// standardised `VR(q) - 1` from overlapping q-step moves, debiased; the p-value is two-sided. The same as Python
    /// arch's `VarianceRatio(y, lags=q)`.
    /// \ingroup core
    /// \param y The levels (log prices), at least `q + 2`.
    /// \param q The horizon, at least 2.
    /// \return The z statistic and its p-value.
    [[nodiscard]] inline test_result variance_ratio_test(std::span<const double> y, std::size_t q)
    {
        assert(q >= 2 && y.size() >= q + 2);
        const std::size_t nq = y.size() - 1;
        const double n = static_cast<double>(nq), qd = static_cast<double>(q);
        const double mu = (y.back() - y.front()) / n;
        double sigma_a = 0;
        for (std::size_t t = 1; t < y.size(); ++t)
            sigma_a += (y[t] - y[t - 1] - mu) * (y[t] - y[t - 1] - mu);
        sigma_a /= n - 1;
        double sigma_b = 0;
        for (std::size_t t = q; t < y.size(); ++t)
            sigma_b += (y[t] - y[t - q] - qd * mu) * (y[t] - y[t - q] - qd * mu);
        const double m = qd * (n - qd + 1) * (1 - qd / n);
        sigma_b /= m;
        const double vr = sigma_b / sigma_a;
        double denominator = 0;
        for (std::size_t t = 1; t < y.size(); ++t)
            denominator += (y[t] - y[t - 1] - mu) * (y[t] - y[t - 1] - mu);
        double theta = 0;
        for (std::size_t j = 1; j < q; ++j) {
            double delta = 0;
            for (std::size_t t = j + 1; t < y.size(); ++t) {
                const double a = y[t] - y[t - 1] - mu, b = y[t - j] - y[t - j - 1] - mu;
                delta += a * a * b * b;
            }
            delta *= n / (denominator * denominator);
            const double w = 2 * (qd - static_cast<double>(j)) / qd;
            theta += w * w * delta;
        }
        test_result out;
        out.statistic = std::sqrt(n) * (vr - 1) / std::sqrt(theta);   // Lo and MacKinlay's z*: sqrt(nq) M / sqrt(theta)
        out.p_value = 2 * (1 - math::normal_cdf(std::abs(out.statistic)));
        return out;
    }

    /// \brief The Ljung-Box test that a series' first `lags` autocorrelations are all zero: increments of a martingale should pass.
    /// \ingroup core
    /// \param x    The series (increments, returns), longer than `lags`.
    /// \param lags The number of autocorrelations, at least 1.
    /// \return `Q = n (n + 2) sum rho_k^2 / (n - k)` and its chi-square(`lags`) p-value.
    [[nodiscard]] inline test_result ljung_box(std::span<const double> x, std::size_t lags)
    {
        assert(lags >= 1 && x.size() > lags);
        const double n = static_cast<double>(x.size());
        double mean = 0;
        for (const double v : x)
            mean += v;
        mean /= n;
        double c0 = 0;
        for (const double v : x)
            c0 += (v - mean) * (v - mean);
        double q = 0;
        for (std::size_t k = 1; k <= lags; ++k) {
            double ck = 0;
            for (std::size_t t = k; t < x.size(); ++t)
                ck += (x[t] - mean) * (x[t - k] - mean);
            const double rho = ck / c0;
            q += rho * rho / (n - static_cast<double>(k));
        }
        test_result out;
        out.statistic = n * (n + 2) * q;
        out.p_value = 1 - math::chi_square_cdf(out.statistic, static_cast<double>(lags));
        return out;
    }

} // namespace snicholls::stochastic

#endif /* snicholls_core_stochastic_martingale_hpp */
