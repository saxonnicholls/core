//
//  ransac.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::stats::ransac` and `ransac_iterations`: a robust fit of the caller's model, ignoring the data that do not fit it.
/// \ingroup core
///
/// **Overview.** Least squares is ruined by a few wrong correspondences (two tracks associated in error, a
/// mismatched landmark). RANSAC fits the model to many random minimal samples, scores each fit by how many data
/// lie within a threshold of it, and keeps the best. The model is the caller's: `ransac<S>` needs
///
/// - `fit(std::span<const std::size_t> sample)`, which returns something that tests as a model and dereferences to
///   one (`std::optional`, `result`): the fit to `S` data, or nothing for a degenerate sample;
/// - `error(model, i)`, the residual of datum `i` (in the threshold's units).
///
/// It returns the best model, its inlier count and its inliers' indices. A final least-squares fit on the inliers
/// is the caller's call (`align_2d` on them, say). The number of tries adapts: after each improvement it drops to
/// `ransac_iterations(confidence, inlier fraction, S)`, never above `max_iterations`.
///
/// \snippet stats/ransac_demo.cpp ransac
///
/// **Relationships.** `stats/random.hpp` (`uniform_index`), `geometry/alignment.hpp` (a typical model),
/// `math/constexpr_math.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, no allocation (the caller passes the inlier buffer). `noexcept` exactly when the
/// generator, `fit` and `error` are. Reproducible for a given generator state. Ties in the inlier count go to the
/// lower sum of inlier residuals.
///
/// **Errors.** `ransac_error::too_few_data`: fewer than `S` data, or an inlier buffer shorter than the data.
/// `ransac_error::no_model`: every sample was degenerate, or no model had `S` inliers.
///
/// **Performance.** Not benchmarked: the caller's `fit` and `error` dominate.
///
/// **Tests and demos.** `tests/stats/ransac_test.cpp`, `demos/stats/ransac_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<span>`, `<type_traits>`, `<utility>`,
/// `math/constexpr_math.hpp`, `result.hpp`, `stats/random.hpp`.
///
/// **Known issues.** Samples are drawn with replacement across tries; within a try the `S` indices are distinct.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_stats_ransac_hpp
#define snicholls_core_stats_ransac_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::stats {

    /// \brief Why RANSAC found no model.
    /// \ingroup core
    enum class ransac_error : std::uint8_t {
        too_few_data = 1,  ///< Fewer data than a sample needs, or an inlier buffer shorter than the data.
        no_model = 2,      ///< No sample gave a model with a full sample's worth of inliers.
    };

    /// \brief The best model found.
    /// \ingroup core
    /// \tparam Model The model type.
    template <typename Model>
    struct ransac_result {
        Model model;               ///< The model.
        std::size_t inliers = 0;   ///< How many data are within the threshold; their indices lead the caller's buffer.
        std::size_t tries = 0;     ///< How many samples were drawn.
    };

    /// \brief The tries needed to draw an all-inlier sample of `s` data with probability `confidence`, when a
    /// fraction `inlier_fraction` of the data are inliers: `⌈log(1 - confidence) / log(1 - w^s)⌉`.
    /// \ingroup core
    /// \param confidence The probability wanted, in (0, 1).
    /// \param inlier_fraction The fraction of inliers, in (0, 1].
    /// \param s The sample size.
    /// \return The tries: 1 when every datum is an inlier, the largest `std::size_t` when none is.
    [[nodiscard]] constexpr std::size_t ransac_iterations(double confidence, double inlier_fraction, std::size_t s) noexcept
    {
        if (inlier_fraction >= 1.0)
            return 1;
        const double ws = math::pow(inlier_fraction, static_cast<double>(s));
        if (!(ws > 0.0) || !(confidence > 0.0))
            return ws > 0.0 ? 1 : static_cast<std::size_t>(-1);
        const double n = math::log(1.0 - (confidence < 1.0 ? confidence : 1.0 - 1e-16)) / math::log1p(-ws);   // log1p: 1 - ws rounds to 1 for ws below 2^-53
        if (!(n < 1e18))
            return static_cast<std::size_t>(-1);
        const auto k = static_cast<std::size_t>(n);
        return static_cast<double>(k) < n ? k + 1 : (k == 0 ? 1 : k);
    }

    /// \brief RANSAC over `n` data with minimal samples of `S`.
    /// \ingroup core
    /// \tparam S The minimal sample size.
    /// \tparam Fit The fitter: `fit(std::span<const std::size_t>)` returns an optional-like model.
    /// \tparam Error The residual: `error(model, i)` returns a number.
    /// \tparam G A `full_range_generator`.
    /// \param n The data.
    /// \param fit The fitter.
    /// \param error The residual of one datum under a model.
    /// \param threshold The largest residual of an inlier.
    /// \param inliers Receives the best model's inlier indices (first `inliers` entries); at least `n` long.
    /// \param g The generator.
    /// \param max_iterations The most samples to draw.
    /// \param confidence The probability of drawing at least one all-inlier sample, for the adaptive stop.
    /// \return The best model, or a `ransac_error`.
    template <std::size_t S, typename Fit, typename Error, full_range_generator G>
    [[nodiscard]] constexpr auto ransac(std::size_t n, Fit&& fit, Error&& error, double threshold, std::span<std::size_t> inliers, G& g,
                                        std::size_t max_iterations = 1000, double confidence = 0.99)
        noexcept(noexcept(g()) && noexcept(fit(std::declval<std::span<const std::size_t>>()))
                 && noexcept(error(*fit(std::declval<std::span<const std::size_t>>()), std::size_t{}))
                 && std::is_nothrow_copy_constructible_v<std::remove_cvref_t<decltype(*fit(std::declval<std::span<const std::size_t>>()))>>
                 && std::is_nothrow_copy_assignable_v<std::remove_cvref_t<decltype(*fit(std::declval<std::span<const std::size_t>>()))>>)
        -> result<ransac_result<std::remove_cvref_t<decltype(*fit(std::declval<std::span<const std::size_t>>()))>>, ransac_error>
    {
        using model_t = std::remove_cvref_t<decltype(*fit(std::declval<std::span<const std::size_t>>()))>;
        static_assert(S > 0, "snicholls::stats::ransac: a sample has at least one datum");
        if (n < S || inliers.size() < n)
            return err(ransac_error::too_few_data);
        bool have = false;
        model_t best{};
        std::size_t best_count = 0;
        double best_sum = 0;
        std::size_t needed = max_iterations;
        std::size_t tries = 0;
        std::array<std::size_t, S> sample{};
        for (; tries < needed && tries < max_iterations; ++tries) {
            for (std::size_t k = 0; k < S; ++k) {
                bool fresh = false;
                while (!fresh) {
                    sample[k] = static_cast<std::size_t>(uniform_index(g, n));
                    fresh = true;
                    for (std::size_t q = 0; q < k; ++q)
                        fresh = fresh && sample[q] != sample[k];
                }
            }
            auto m = fit(std::span<const std::size_t>(sample));
            if (!m)
                continue;
            std::size_t count = 0;
            double sum = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double e = static_cast<double>(error(*m, i));
                if (e <= threshold) {
                    ++count;
                    sum += e;
                }
            }
            if (count > best_count || (count == best_count && have && sum < best_sum)) {
                best = *m;
                best_count = count;
                best_sum = sum;
                have = true;
                const std::size_t want = ransac_iterations(confidence, static_cast<double>(count) / static_cast<double>(n), S);
                if (want < needed)
                    needed = want;
            }
        }
        if (!have || best_count < S)
            return err(ransac_error::no_model);
        std::size_t k = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (static_cast<double>(error(best, i)) <= threshold)
                inliers[k++] = i;
        return ok(ransac_result<model_t>{best, k, tries});
    }

} // namespace snicholls::stats

#endif /* snicholls_core_stats_ransac_hpp */
