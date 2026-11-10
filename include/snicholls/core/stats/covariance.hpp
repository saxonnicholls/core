//
//  covariance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Covariance matrices from a history of returns: the sample estimate and Ledoit and Wolf's shrinkage.
/// \ingroup core
///
/// **Overview.** With many assets and a short history the sample covariance is noisy and badly
/// conditioned: its smallest eigenvalues are too small, so an optimiser leans on them and risk
/// contributions swing from day to day. Ledoit and Wolf (2004) shrink it towards a multiple of the
/// identity by the amount that minimises the expected error, estimated from the data:
///
/// \snippet stats/covariance_demo.cpp covariance
///
/// The data are rows of returns, one row per date, each a random-access range of the same length (one
/// column per asset). `sample_covariance` is the maximum-likelihood estimate (divided by T, about each
/// column's mean), the one Ledoit and Wolf shrink; `ledoit_wolf` returns the shrunk matrix and the
/// shrinkage, the same as scikit-learn's `LedoitWolf`.
///
/// **Relationships.** `finance/attribution.hpp`'s `risk_contributions` takes the result;
/// `stats/accumulators.hpp`'s `covariance` for one pair, streaming. Group: \ref core.
///
/// **Guarantees.** The result is symmetric. The shrinkage is in [0, 1]; the shrunk matrix is positive
/// definite whenever any column varies. O(T N^2) time, O(N^2) memory.
///
/// **Errors.** None: rows of different lengths are a precondition violation (asserted); no rows give an
/// empty matrix.
///
/// **Performance.** Not benchmarked; two passes over the data.
///
/// **Tests and demos.** `tests/stats/covariance_test.cpp`, against scikit-learn;
/// `demos/stats/covariance_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<ranges>`, `<vector>`.
///
/// **Known issues.** The target is a scaled identity; Ledoit and Wolf's constant-correlation and
/// single-index targets are not provided.
///
/// **Status.** New in 0.3.0.

#ifndef snicholls_core_stats_covariance_hpp
#define snicholls_core_stats_covariance_hpp

#include <cassert>
#include <cstddef>
#include <ranges>
#include <vector>

namespace snicholls::stats {

    /// \brief A covariance matrix, row-major, with the shrinkage applied to it (0 for the sample estimate).
    /// \ingroup core
    struct covariance_matrix {
        std::size_t n = 0;              ///< The number of assets.
        std::vector<double> values;     ///< `n x n`, row-major.
        double shrinkage = 0;           ///< The weight on the target: 0 for the sample estimate.
        double target = 0;              ///< The target's diagonal: the mean of the sample variances.

        /// \brief The covariance of assets `i` and `j`.
        /// \param i A row.
        /// \param j A column.
        /// \return The entry.
        [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept { return values[i * n + j]; }
    };

    namespace detail {

        // The data about each column's mean, as one contiguous T x N block.
        template <std::ranges::input_range Rows>
        std::vector<double> centred(const Rows& rows, std::size_t& t, std::size_t& n)
        {
            std::vector<double> x;
            t = 0;
            n = 0;
            for (const auto& row : rows) {
                const auto width = static_cast<std::size_t>(std::ranges::size(row));
                assert(t == 0 || width == n);
                n = width;
                for (const auto& v : row)
                    x.push_back(static_cast<double>(v));
                ++t;
            }
            for (std::size_t j = 0; j < n; ++j) {
                double mean = 0;
                for (std::size_t r = 0; r < t; ++r)
                    mean += x[r * n + j];
                mean /= static_cast<double>(t);
                for (std::size_t r = 0; r < t; ++r)
                    x[r * n + j] -= mean;
            }
            return x;
        }

        inline std::vector<double> cross(const std::vector<double>& x, std::size_t t, std::size_t n)
        {
            std::vector<double> s(n * n);
            for (std::size_t r = 0; r < t; ++r)
                for (std::size_t i = 0; i < n; ++i) {
                    const double xi = x[r * n + i];
                    for (std::size_t j = i; j < n; ++j)
                        s[i * n + j] += xi * x[r * n + j];
                }
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    s[i * n + j] = s[j * n + i];
            return s;
        }

    } // namespace detail

    /// \brief The sample covariance (maximum likelihood: divided by T, about each column's mean).
    /// \ingroup core
    /// \tparam Rows A range of rows, each a sized range of numbers.
    /// \param rows The returns, one row per date.
    /// \return The matrix.
    /// \throws std::bad_alloc.
    template <std::ranges::input_range Rows>
    [[nodiscard]] covariance_matrix sample_covariance(const Rows& rows)
    {
        std::size_t t = 0, n = 0;
        const std::vector<double> x = detail::centred(rows, t, n);
        covariance_matrix out;
        out.n = n;
        out.values = detail::cross(x, t, n);
        for (double& v : out.values)
            v /= static_cast<double>(t);
        return out;
    }

    /// \brief Ledoit and Wolf's shrunk covariance: `(1 - s) S + s mu I`, with `s` estimated from the data.
    /// \ingroup core
    /// \tparam Rows A range of rows, each a sized range of numbers.
    /// \param rows The returns, one row per date.
    /// \return The matrix, its shrinkage `s` and target `mu`.
    /// \throws std::bad_alloc.
    template <std::ranges::input_range Rows>
    [[nodiscard]] covariance_matrix ledoit_wolf(const Rows& rows)
    {
        std::size_t t = 0, n = 0;
        const std::vector<double> x = detail::centred(rows, t, n);
        covariance_matrix out;
        out.n = n;
        if (t == 0 || n == 0)
            return out;
        const std::vector<double> xtx = detail::cross(x, t, n);
        std::vector<double> x2(x.size());
        for (std::size_t k = 0; k < x.size(); ++k)
            x2[k] = x[k] * x[k];
        const auto td = static_cast<double>(t), nd = static_cast<double>(n);
        double trace = 0;
        for (std::size_t i = 0; i < n; ++i)
            trace += xtx[i * n + i] / td;
        const double mu = trace / nd;
        // beta_: the sum of the entries of X2' X2; delta_: the sum of the squared entries of X' X, over T^2
        double beta_ = 0;
        for (std::size_t r = 0; r < t; ++r) {
            double row_sum = 0;
            for (std::size_t j = 0; j < n; ++j)
                row_sum += x2[r * n + j];
            beta_ += row_sum * row_sum;
        }
        double delta_ = 0;
        for (const double v : xtx)
            delta_ += v * v;
        delta_ /= td * td;
        double beta = (beta_ / td - delta_) / (nd * td);
        const double delta = (delta_ - 2.0 * mu * trace + nd * mu * mu) / nd;
        beta = beta < delta ? beta : delta;
        out.shrinkage = beta == 0 ? 0.0 : beta / delta;
        out.target = mu;
        out.values.resize(n * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                out.values[i * n + j] = (1 - out.shrinkage) * xtx[i * n + j] / td + (i == j ? out.shrinkage * mu : 0.0);
        return out;
    }

} // namespace snicholls::stats

#endif /* snicholls_core_stats_covariance_hpp */
