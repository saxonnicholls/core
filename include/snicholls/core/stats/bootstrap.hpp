//
//  bootstrap.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The bootstrap, fast: resampling by indices, by multinomial or Poisson counts, balanced, or in blocks for time series.
/// \ingroup core
///
/// **Overview.** A bootstrap draws `B` resamples of `n` observations with replacement and recomputes a
/// statistic on each. The usual implementation (`std::mt19937_64`, `std::uniform_int_distribution`, a
/// gathered copy of each resample) spends its time drawing indices and copying data. Here:
///
/// - indices come from `wyrand` and Lemire's unbiased draw, two indices per 64-bit draw;
/// - a statistic that is a weighted sum (a mean, a variance, a Sharpe ratio, any moment) need not see a
///   resample at all, only how many times each observation was drawn: `for_each_resample_counts`
///   gives those multinomial counts, and `bootstrap_moments` turns them into weighted moments in one
///   sequential pass the compiler vectorises;
/// - the Poisson bootstrap (`for_each_poisson_resample`) draws each count independently as Poisson(1),
///   two per 64-bit draw: about 1.4 times the cost of multinomial counts on one core, but independent,
///   so it suits streaming data, an `n` not known in advance, and workers that must not coordinate; its
///   resamples vary slightly in size, which for moments is immaterial;
/// - the balanced bootstrap uses every observation exactly `B` times in total, which removes the
///   first-order simulation error in the bootstrap mean;
/// - returns are autocorrelated, so `for_each_stationary_resample` (Politis and Romano) and
///   `for_each_block_resample` keep runs of consecutive observations together.
///
/// \snippet stats/bootstrap_demo.cpp bootstrap
///
/// A statistic that is not a weighted sum (a median, a maximum drawdown) uses `bootstrap`, which gathers
/// each resample into one reused buffer. The draws are specified to the bit, so a seed reproduces a
/// bootstrap on every platform.
///
/// When the same `n` is bootstrapped many times (every strategy over the same window), draw the resamples
/// once into a `resample_table` and reuse it: reading an index is about twice as fast as drawing one, and
/// every series then sees the same resamples (common random numbers). Built at run time it is as fast as a
/// table built at compile time, without megabytes in the binary. Quasi-random indices do not help much,
/// because a bootstrap's dimension is `n`.
///
/// **Relationships.** `stats/random.hpp`; `stats/low_discrepancy.hpp` for Monte Carlo integration.
/// Group: \ref core.
///
/// **Guarantees.** The `for_each_*` functions allocate once per call (the index or count buffer) and call
/// `f` once per resample with a view valid for that call only. Results are in resample order. `n` is below
/// 2^32. Not thread-safe per generator; give each thread its own (`wyrand::stream`).
///
/// **Errors.** None of their own; `n` of zero and a mean block length below 1 are precondition violations
/// (asserted). `std::bad_alloc`.
///
/// **Performance.** `benchmarks/stats/bootstrap_benchmark.cpp`: the bootstrap distribution of the mean of
/// 1,024 observations, 2,000 resamples, against `std::mt19937_64` with a gathered copy, after a fairness
/// check. Release, Apple clang 17, Xeon W-3245, 2026-11-07: 10.4 ms for the standard way;
/// 3.7 ms by `for_each_resample` (2.8 times faster; 4.4 ms for the same draws written by hand); 3.7 ms by
/// multinomial counts; 5.4 ms by Poisson counts; 2.1 ms reusing a `resample_table` (4.9 times faster). All
/// of these are single-threaded.
///
/// **Tests and demos.** `tests/stats/bootstrap_test.cpp` (BCa against SciPy, block lengths against arch), `demos/stats/bootstrap_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cassert>`, `<cmath>`, `<cstddef>`, `<cstdint>`, `<numeric>`, `<ranges>`,
/// `<span>`, `<utility>`, `<vector>`, `math/constexpr_math.hpp`, `stats/random.hpp`.
///
/// **Known issues.** Poisson counts above 12 are capped (probability below 1e-10). The jackknife for BCa
/// recomputes the statistic `n` times; for a statistic of moments it could be done in one pass, and is not yet.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_stats_bootstrap_hpp
#define snicholls_core_stats_bootstrap_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace snicholls::stats {

    namespace detail {

        // Indices in [0, n) from 32-bit halves of 64-bit draws: two per call of the generator.
        template <full_range_generator G>
        class index_draws {
        public:
            constexpr index_draws(G& g, std::uint32_t n) noexcept : g_(&g), n_(n) {}

            constexpr std::uint32_t next() noexcept(noexcept(std::declval<G&>()()))
            {
                return uniform_index32(bits32(), n_, [this] { return bits32(); });
            }

            // A whole resample with the state in locals: stores to `out` cannot then alias it.
            constexpr void fill(std::span<std::uint32_t> out) noexcept(noexcept(std::declval<G&>()()))
            {
                G& g = *g_;
                const std::uint32_t n = n_;
                std::uint64_t bits = bits_;
                bool have = have_;
                const auto next32 = [&]() -> std::uint32_t {
                    if (have) {
                        have = false;
                        return static_cast<std::uint32_t>(bits >> 32);
                    }
                    bits = random_u64(g);
                    have = true;
                    return static_cast<std::uint32_t>(bits);
                };
                for (std::uint32_t& i : out)
                    i = uniform_index32(next32(), n, next32);
                bits_ = bits;
                have_ = have;
            }

            constexpr std::uint32_t bits32() noexcept(noexcept(std::declval<G&>()()))
            {
                if (have_) {
                    have_ = false;
                    return static_cast<std::uint32_t>(bits_ >> 32);
                }
                bits_ = random_u64(*g_);
                have_ = true;
                return static_cast<std::uint32_t>(bits_);
            }

        private:
            G* g_;
            std::uint32_t n_;
            std::uint64_t bits_ = 0;
            bool have_ = false;
        };

        // Poisson(1) by inversion on 32 bits: thresholds floor(F(k) 2^32), the last a sentinel.
        consteval std::array<std::uint64_t, 13> poisson1_thresholds() noexcept
        {
            double e_inverse = 0, term = 1;   // e^-1 = sum (-1)^j / j!
            for (int j = 0; j < 25; ++j) {
                e_inverse += (j % 2 == 0 ? term : -term);
                term /= j + 1;
            }
            std::array<std::uint64_t, 13> t{};
            double p = e_inverse, cdf = 0;
            for (std::size_t k = 0; k < 12; ++k) {
                cdf += p;
                p /= static_cast<double>(k + 1);
                t[k] = static_cast<std::uint64_t>(cdf * 4294967296.0);
            }
            t[12] = std::uint64_t{1} << 32;
            return t;
        }
        inline constexpr std::array<std::uint64_t, 13> poisson1 = poisson1_thresholds();

        // Branch-free to a count of 5, which is 99.94% of draws; above that a loop, so rare that it is
        // always predicted not taken.
        constexpr std::uint32_t poisson1_from(std::uint32_t bits) noexcept
        {
            const std::uint64_t b = bits;
            std::uint32_t k = 0;
            for (std::size_t j = 0; j < 5; ++j)
                k += static_cast<std::uint32_t>(b >= poisson1[j]);
            if (b >= poisson1[4]) [[unlikely]]
                while (b >= poisson1[k])
                    ++k;
            return k;
        }

    } // namespace detail

    /// \brief Call `f(indices)` for each of `resamples` resamples of `n` observations, drawn with replacement.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \tparam F The callable's type.
    /// \param n         The number of observations, 1 to 2^32 - 1.
    /// \param resamples How many resamples.
    /// \param g         The generator.
    /// \param f         Called as `f(std::span<const std::uint32_t>)` with `n` indices; the view is reused.
    /// \throws std::bad_alloc; whatever `f` throws.
    template <full_range_generator G, typename F>
    void for_each_resample(std::size_t n, std::size_t resamples, G& g, F&& f)
    {
        assert(n > 0 && n <= 0xFFFFFFFFu);
        std::vector<std::uint32_t> indices(n);
        detail::index_draws<G> draws(g, static_cast<std::uint32_t>(n));
        for (std::size_t b = 0; b < resamples; ++b) {
            draws.fill(std::span<std::uint32_t>(indices));
            f(std::span<const std::uint32_t>(indices));
        }
    }

    /// \brief Resamples drawn once and kept: reused across many series and statistics, they cost a read instead of a draw.
    ///
    /// Reading a table of indices is about twice as fast as drawing them (`BENCHMARKS.md`), and every series
    /// bootstrapped through one table sees the same resamples: common random numbers, so the difference between two
    /// strategies is estimated with less noise than from independent resamples. It holds `n x resamples` 32-bit
    /// indices: 8 MB for 1,024 observations and 2,000 resamples. Built at run time, it is as fast as one built at
    /// compile time and costs the binary nothing.
    /// \ingroup core
    class resample_table {
    public:
        /// \brief Draw the resamples.
        /// \tparam G The generator's type.
        /// \param n         The number of observations, 1 to 2^32 - 1.
        /// \param resamples How many resamples.
        /// \param g         The generator; the same state gives the same table on every platform.
        /// \throws std::bad_alloc.
        template <full_range_generator G>
        resample_table(std::size_t n, std::size_t resamples, G& g) : n_(n), resamples_(resamples), indices_(n * resamples)
        {
            assert(n > 0 && n <= 0xFFFFFFFFu);
            detail::index_draws<G> draws(g, static_cast<std::uint32_t>(n));
            for (std::size_t b = 0; b < resamples; ++b)
                draws.fill(std::span<std::uint32_t>(indices_.data() + b * n, n));
        }

        /// \brief The number of observations in a resample.
        /// \return n.
        [[nodiscard]] std::size_t observations() const noexcept { return n_; }

        /// \brief The number of resamples.
        /// \return B.
        [[nodiscard]] std::size_t size() const noexcept { return resamples_; }

        /// \brief Resample `b`'s indices.
        /// \param b Below `size()`.
        /// \return The view, valid as long as the table.
        [[nodiscard]] std::span<const std::uint32_t> operator[](std::size_t b) const noexcept
        {
            assert(b < resamples_);
            return {indices_.data() + b * n_, n_};
        }

        /// \brief Call `f(indices)` for each resample, in order.
        /// \tparam F The callable's type.
        /// \param f Called as `f(std::span<const std::uint32_t>)`.
        /// \throws Whatever `f` throws.
        template <typename F>
        void for_each(F&& f) const
        {
            for (std::size_t b = 0; b < resamples_; ++b)
                f((*this)[b]);
        }

    private:
        std::size_t n_;
        std::size_t resamples_;
        std::vector<std::uint32_t> indices_;
    };

    /// \brief Call `f(counts)` for each resample: how many times each observation was drawn, `n` draws in all.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \tparam F The callable's type.
    /// \param n         The number of observations, 1 to 2^32 - 1.
    /// \param resamples How many resamples.
    /// \param g         The generator.
    /// \param f         Called as `f(std::span<const std::uint32_t>)` with `n` counts summing to `n`; the view is reused.
    /// \throws std::bad_alloc; whatever `f` throws.
    template <full_range_generator G, typename F>
    void for_each_resample_counts(std::size_t n, std::size_t resamples, G& g, F&& f)
    {
        assert(n > 0 && n <= 0xFFFFFFFFu);
        std::vector<std::uint32_t> counts(n);
        detail::index_draws<G> draws(g, static_cast<std::uint32_t>(n));
        for (std::size_t b = 0; b < resamples; ++b) {
            std::fill(counts.begin(), counts.end(), 0u);
            for (std::size_t k = 0; k < n; ++k)
                ++counts[draws.next()];
            f(std::span<const std::uint32_t>(counts));
        }
    }

    /// \brief Call `f(counts)` for each Poisson resample: each observation's count drawn independently as Poisson(1).
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \tparam F The callable's type.
    /// \param n         The number of observations.
    /// \param resamples How many resamples.
    /// \param g         The generator.
    /// \param f         Called as `f(std::span<const std::uint32_t>)` with `n` counts; their sum is about `n`, not exactly.
    /// \throws std::bad_alloc; whatever `f` throws.
    template <full_range_generator G, typename F>
    void for_each_poisson_resample(std::size_t n, std::size_t resamples, G& g, F&& f)
    {
        assert(n > 0);
        std::vector<std::uint32_t> counts(n);
        for (std::size_t b = 0; b < resamples; ++b) {
            std::size_t k = 0;
            for (; k + 1 < n; k += 2) {
                const std::uint64_t bits = random_u64(g);
                counts[k] = detail::poisson1_from(static_cast<std::uint32_t>(bits));
                counts[k + 1] = detail::poisson1_from(static_cast<std::uint32_t>(bits >> 32));
            }
            if (k < n)
                counts[k] = detail::poisson1_from(static_cast<std::uint32_t>(random_u64(g)));
            f(std::span<const std::uint32_t>(counts));
        }
    }

    /// \brief Call `f(indices)` for each resample of the balanced bootstrap: over all resamples, each observation appears exactly `resamples` times.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \tparam F The callable's type.
    /// \param n         The number of observations, 1 to 2^32 - 1.
    /// \param resamples How many resamples.
    /// \param g         The generator.
    /// \param f         Called as `f(std::span<const std::uint32_t>)` with `n` indices.
    /// \throws std::bad_alloc (it holds `n x resamples` indices); whatever `f` throws.
    template <full_range_generator G, typename F>
    void for_each_balanced_resample(std::size_t n, std::size_t resamples, G& g, F&& f)
    {
        assert(n > 0 && n <= 0xFFFFFFFFu);
        std::vector<std::uint32_t> pool(n * resamples);
        for (std::size_t i = 0; i < pool.size(); ++i)
            pool[i] = static_cast<std::uint32_t>(i % n);
        shuffle(std::span<std::uint32_t>(pool), g);
        for (std::size_t b = 0; b < resamples; ++b)
            f(std::span<const std::uint32_t>(pool.data() + b * n, n));
    }

    /// \brief Call `f(indices)` for each resample of the stationary bootstrap (Politis and Romano), for time series.
    ///
    /// Each resample is a run of blocks: a block starts at a random observation and continues to the next
    /// one (wrapping round) with probability `1 - 1/mean_block_length`, so block lengths are geometric with
    /// that mean and the resampled series is stationary. Autocorrelation within a block is kept.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \tparam F The callable's type.
    /// \param n                 The number of observations, 1 to 2^32 - 1.
    /// \param mean_block_length The mean block length, at least 1; 1 is the ordinary bootstrap.
    /// \param resamples         How many resamples.
    /// \param g                 The generator.
    /// \param f                 Called as `f(std::span<const std::uint32_t>)` with `n` indices.
    /// \throws std::bad_alloc; whatever `f` throws.
    template <full_range_generator G, typename F>
    void for_each_stationary_resample(std::size_t n, double mean_block_length, std::size_t resamples, G& g, F&& f)
    {
        assert(n > 0 && n <= 0xFFFFFFFFu && mean_block_length >= 1.0);
        std::vector<std::uint32_t> indices(n);
        detail::index_draws<G> draws(g, static_cast<std::uint32_t>(n));
        const auto restart = static_cast<std::uint64_t>((1.0 / mean_block_length) * 4294967296.0);
        for (std::size_t b = 0; b < resamples; ++b) {
            std::uint32_t current = draws.next();
            indices[0] = current;
            for (std::size_t k = 1; k < n; ++k) {
                if (draws.bits32() < restart)
                    current = draws.next();
                else
                    current = current + 1 == n ? 0 : current + 1;
                indices[k] = current;
            }
            f(std::span<const std::uint32_t>(indices));
        }
    }

    /// \brief Call `f(indices)` for each resample of the moving block bootstrap: blocks of `block_length` consecutive observations.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \tparam F The callable's type.
    /// \param n            The number of observations, 1 to 2^32 - 1.
    /// \param block_length The block length, 1 to `n`.
    /// \param resamples    How many resamples.
    /// \param g            The generator.
    /// \param f            Called as `f(std::span<const std::uint32_t>)` with `n` indices; the last block is cut short.
    /// \throws std::bad_alloc; whatever `f` throws.
    template <full_range_generator G, typename F>
    void for_each_block_resample(std::size_t n, std::size_t block_length, std::size_t resamples, G& g, F&& f)
    {
        assert(n > 0 && n <= 0xFFFFFFFFu && block_length >= 1 && block_length <= n);
        std::vector<std::uint32_t> indices(n);
        detail::index_draws<G> draws(g, static_cast<std::uint32_t>(n - block_length + 1));
        for (std::size_t b = 0; b < resamples; ++b) {
            for (std::size_t k = 0; k < n;) {
                const std::uint32_t start = draws.next();
                for (std::size_t j = 0; j < block_length && k < n; ++j, ++k)
                    indices[k] = static_cast<std::uint32_t>(start + j);
            }
            f(std::span<const std::uint32_t>(indices));
        }
    }

    /// \brief The block lengths Politis and White (2004, corrected by Patton, Politis and White 2009) recommend for a series.
    /// \ingroup core
    struct block_lengths {
        double stationary = 1;   ///< The mean block length for `for_each_stationary_resample`.
        double circular = 1;     ///< The block length for a circular (or moving) block bootstrap.
    };

    /// \brief Estimate the block lengths a dependent series needs, from its autocorrelations (the same as Python arch's `optimal_block_length`).
    ///
    /// Autocorrelations are taken until `max(5, log10 n)` in a row are insignificant; a flat-top kernel over twice
    /// that many lags estimates the long-run variance and its derivative, whose ratio sets the length. An
    /// independent series gives about 1; a persistent one, tens. Capped at `min(3 sqrt(n), n / 3)`.
    /// \ingroup core
    /// \param x The series, at least 8 observations.
    /// \return The two lengths.
    /// \throws std::bad_alloc.
    [[nodiscard]] inline block_lengths optimal_block_length(std::span<const double> x)
    {
        const std::size_t n = x.size();
        assert(n >= 8);
        double mean = 0;
        for (const double v : x)
            mean += v;
        mean /= static_cast<double>(n);
        std::vector<double> eps(n);
        for (std::size_t i = 0; i < n; ++i)
            eps[i] = x[i] - mean;
        const double nd = static_cast<double>(n);
        const double b_max = std::ceil(std::min(3 * std::sqrt(nd), nd / 3));
        const std::size_t kn = std::max<std::size_t>(5, static_cast<std::size_t>(std::log10(nd)));
        const std::size_t m_max = static_cast<std::size_t>(std::ceil(std::sqrt(nd))) + kn;
        const double cv = 2 * std::sqrt(std::log10(nd) / nd);
        std::vector<double> acv(m_max + 1), abs_acorr(m_max + 1);
        std::optional<std::size_t> opt_m;
        const auto dot = [&](std::size_t a0, std::size_t b0, std::size_t len) {
            double s = 0;
            for (std::size_t k = 0; k < len; ++k)
                s += eps[a0 + k] * eps[b0 + k];
            return s;
        };
        for (std::size_t i = 0; i <= m_max && i + 1 < n; ++i) {
            const double v1 = dot(i + 1, i + 1, n - i - 1);
            const double v2 = dot(0, 0, n - i - 1);
            const double cross_prod = dot(i, 0, n - i);
            acv[i] = cross_prod / nd;
            abs_acorr[i] = std::abs(cross_prod) / std::sqrt(v1 * v2);
            if (i >= kn && !opt_m) {
                bool insignificant = true;
                for (std::size_t k = i - kn; k < i; ++k)
                    insignificant = insignificant && abs_acorr[k] < cv;
                if (insignificant)
                    opt_m = i - kn;
            }
        }
        std::size_t m = opt_m ? 2 * std::max<std::size_t>(*opt_m, 1) : m_max;
        m = std::min(m, m_max);
        double g = 0, lr_acv = acv[0];
        for (std::size_t k = 1; k <= m; ++k) {
            const double ratio = static_cast<double>(k) / static_cast<double>(m);
            const double lambda = ratio <= 0.5 ? 1.0 : 2 * (1 - ratio);
            g += 2 * lambda * static_cast<double>(k) * acv[k];
            lr_acv += 2 * lambda * acv[k];
        }
        const double d_sb = 2 * lr_acv * lr_acv;
        const double d_cb = 4.0 / 3.0 * lr_acv * lr_acv;
        block_lengths out;
        out.stationary = std::min(std::cbrt(2 * g * g / d_sb) * std::cbrt(nd), b_max);
        out.circular = std::min(std::cbrt(2 * g * g / d_cb) * std::cbrt(nd), b_max);
        return out;
    }

    // ---- summaries -----------------------------------------------------------------------------------------------

    /// \brief A bootstrap distribution: the statistic on the data and on each resample, and what follows from them.
    /// \ingroup core
    struct bootstrap_distribution {
        double estimate = 0;               ///< The statistic on the original data.
        std::vector<double> replicates;    ///< The statistic on each resample, in resample order.

        /// \brief The mean of the replicates.
        /// \return It; NaN with none.
        [[nodiscard]] double mean() const noexcept
        {
            if (replicates.empty())
                return std::nan("");
            return std::accumulate(replicates.begin(), replicates.end(), 0.0) / static_cast<double>(replicates.size());
        }

        /// \brief The bootstrap standard error: the replicates' standard deviation.
        /// \return It; NaN with fewer than two.
        [[nodiscard]] double standard_error() const noexcept
        {
            if (replicates.size() < 2)
                return std::nan("");
            const double m = mean();
            double sum = 0;
            for (const double r : replicates)
                sum += (r - m) * (r - m);
            return std::sqrt(sum / static_cast<double>(replicates.size() - 1));
        }

        /// \brief The bootstrap estimate of bias: the replicates' mean less the estimate.
        /// \return It.
        [[nodiscard]] double bias() const noexcept { return mean() - estimate; }

        /// \brief The `q` quantile of the replicates, interpolated linearly (R's type 7).
        /// \param q In `[0, 1]`.
        /// \return It; NaN with no replicates.
        /// \throws std::bad_alloc.
        [[nodiscard]] double quantile(double q) const
        {
            if (replicates.empty())
                return std::nan("");
            std::vector<double> sorted = replicates;
            std::sort(sorted.begin(), sorted.end());
            const double h = q * static_cast<double>(sorted.size() - 1);
            const auto lo = static_cast<std::size_t>(std::floor(h));
            const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
            return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
        }

        /// \brief The percentile interval: the replicates' `(1 - c)/2` and `(1 + c)/2` quantiles.
        /// \param confidence `c`, for example 0.95.
        /// \return The interval.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::pair<double, double> percentile_interval(double confidence) const
        {
            return {quantile((1 - confidence) / 2), quantile((1 + confidence) / 2)};
        }

        /// \brief The BCa interval (Efron 1987): percentiles corrected for the bootstrap distribution's bias and skew.
        ///
        /// More accurate than the percentile interval for a skewed statistic (a variance, a Sharpe ratio, a
        /// maximum drawdown). The bias correction comes from the share of replicates below the estimate, the
        /// acceleration from the jackknife (`stats::jackknife`). The same formulas as SciPy's `bootstrap(method="BCa")`.
        /// \param confidence `c`, for example 0.95.
        /// \param jackknife  The statistic on the data with each observation left out in turn.
        /// \return The interval; NaN ends if every replicate is on one side of the estimate.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::pair<double, double> bca_interval(double confidence, std::span<const double> jackknife) const
        {
            double below = 0;   // a replicate equal to the estimate counts half, as in SciPy
            for (const double r : replicates)
                below += r < estimate ? 1.0 : (r == estimate ? 0.5 : 0.0);
            const double z0 = math::normal_quantile(below / static_cast<double>(replicates.size()));
            double jmean = 0;
            for (const double j : jackknife)
                jmean += j;
            jmean /= static_cast<double>(jackknife.size());
            double num = 0, den = 0;
            for (const double j : jackknife) {
                const double d = jmean - j;
                num += d * d * d;
                den += d * d;
            }
            const double a = den > 0 ? num / (6 * std::pow(den, 1.5)) : 0.0;
            const auto adjusted = [&](double alpha) {
                const double z = math::normal_quantile(alpha);
                return math::normal_cdf(z0 + (z0 + z) / (1 - a * (z0 + z)));
            };
            const double lo = adjusted((1 - confidence) / 2), hi = adjusted((1 + confidence) / 2);
            if (std::isnan(lo) || std::isnan(hi))
                return {std::nan(""), std::nan("")};
            return {quantile(lo), quantile(hi)};
        }

        /// \brief The basic (reverse percentile) interval: `2 estimate - ` the percentile interval's ends.
        /// \param confidence `c`, for example 0.95.
        /// \return The interval.
        /// \throws std::bad_alloc.
        [[nodiscard]] std::pair<double, double> basic_interval(double confidence) const
        {
            const auto [lo, hi] = percentile_interval(confidence);
            return {2 * estimate - hi, 2 * estimate - lo};
        }
    };

    /// \brief Weighted moments of data under resampling counts: total weight, mean and unbiased variance.
    /// \ingroup core
    struct weighted_moments {
        double weight = 0;     ///< The sum of the counts: the resample's size.
        double mean = 0;       ///< The weighted mean.
        double variance = 0;   ///< The weighted sample variance, divided by `weight - 1`; 0 when `weight <= 1`.

        /// \brief The standard deviation.
        /// \return `sqrt(variance)`.
        [[nodiscard]] double standard_deviation() const noexcept { return std::sqrt(variance); }
    };

    /// \brief The weighted moments of `data` under `counts`, in one sequential pass, shifted by `centre` for accuracy.
    /// \ingroup core
    /// \param data   The observations.
    /// \param counts How many times each was drawn.
    /// \param centre A value near the data's mean (its mean, computed once): the sums are taken about it.
    /// \return The moments.
    [[nodiscard]] inline weighted_moments moments_of(std::span<const double> data, std::span<const std::uint32_t> counts, double centre = 0) noexcept
    {
        assert(data.size() == counts.size());
        double w = 0, s1 = 0, s2 = 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            const double c = static_cast<double>(counts[i]);
            const double x = data[i] - centre;
            w += c;
            s1 += c * x;
            s2 += c * x * x;
        }
        weighted_moments m;
        m.weight = w;
        m.mean = w > 0 ? centre + s1 / w : std::nan("");
        m.variance = w > 1 ? (s2 - s1 * s1 / w) / (w - 1) : 0.0;
        return m;
    }

    /// \brief How `bootstrap_moments` resamples.
    /// \ingroup core
    enum class resampling {
        multinomial,   ///< Efron's: `n` draws with replacement, as counts. The faster on one core.
        poisson,       ///< Independent Poisson(1) counts: resample sizes vary about `n`; for streams and parallel workers.
    };

    /// \brief The bootstrap of any statistic of the weighted moments (a mean, a volatility, a Sharpe ratio), without gathering a resample.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \tparam F The statistic's type.
    /// \param data      The observations.
    /// \param resamples How many resamples.
    /// \param g         The generator.
    /// \param statistic Called as `statistic(const weighted_moments&)`, returning a `double`.
    /// \param how       Multinomial (the default) or Poisson counts.
    /// \return The distribution; `estimate` is the statistic on the data with every count 1.
    /// \throws std::bad_alloc; whatever the statistic throws.
    template <full_range_generator G, typename F>
    [[nodiscard]] bootstrap_distribution bootstrap_moments(std::span<const double> data, std::size_t resamples, G& g, F&& statistic,
                                                           resampling how = resampling::multinomial)
    {
        bootstrap_distribution out;
        const std::vector<std::uint32_t> ones(data.size(), 1u);
        const double centre = data.empty() ? 0.0 : std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
        out.estimate = statistic(moments_of(data, ones, centre));
        out.replicates.reserve(resamples);
        const auto collect = [&](std::span<const std::uint32_t> counts) { out.replicates.push_back(statistic(moments_of(data, counts, centre))); };
        if (how == resampling::poisson)
            for_each_poisson_resample(data.size(), resamples, g, collect);
        else
            for_each_resample_counts(data.size(), resamples, g, collect);
        return out;
    }

    /// \brief The statistic on the data with each observation left out in turn: the jackknife, for `bca_interval`.
    /// \ingroup core
    /// \tparam R         The data's type: a random-access range.
    /// \tparam Statistic The statistic's type.
    /// \param data      The observations, at least two.
    /// \param statistic Called as `statistic(std::span<const T>)` with `n - 1` observations, returning a `double`.
    /// \return `n` values, the i-th without observation i.
    /// \throws std::bad_alloc; whatever the statistic throws.
    template <std::ranges::random_access_range R, typename Statistic>
    [[nodiscard]] std::vector<double> jackknife(const R& data, Statistic&& statistic)
    {
        using T = std::ranges::range_value_t<R>;
        const std::vector<T> all(std::ranges::begin(data), std::ranges::end(data));
        std::vector<T> without(all.size() - 1);
        std::vector<double> out;
        out.reserve(all.size());
        for (std::size_t i = 0; i < all.size(); ++i) {
            std::copy(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(i), without.begin());
            std::copy(all.begin() + static_cast<std::ptrdiff_t>(i) + 1, all.end(), without.begin() + static_cast<std::ptrdiff_t>(i));
            out.push_back(statistic(std::span<const T>(without)));
        }
        return out;
    }

    /// \brief The bootstrap of any statistic: each resample gathered into one reused buffer and passed to it.
    /// \ingroup core
    /// \tparam R         The data's type: a random-access range.
    /// \tparam G         The generator's type.
    /// \tparam Statistic The statistic's type.
    /// \param data      The observations.
    /// \param resamples How many resamples.
    /// \param g         The generator.
    /// \param statistic Called as `statistic(std::span<const T>)`, returning a `double`.
    /// \return The distribution.
    /// \throws std::bad_alloc; whatever the statistic throws.
    template <std::ranges::random_access_range R, full_range_generator G, typename Statistic>
    [[nodiscard]] bootstrap_distribution bootstrap(const R& data, std::size_t resamples, G& g, Statistic&& statistic)
    {
        using T = std::ranges::range_value_t<R>;
        const std::vector<T> original(std::ranges::begin(data), std::ranges::end(data));
        bootstrap_distribution out;
        out.estimate = statistic(std::span<const T>(original));
        out.replicates.reserve(resamples);
        std::vector<T> resample(original.size());
        for_each_resample(original.size(), resamples, g, [&](std::span<const std::uint32_t> indices) {
            for (std::size_t k = 0; k < indices.size(); ++k)
                resample[k] = original[indices[k]];
            out.replicates.push_back(statistic(std::span<const T>(resample)));
        });
        return out;
    }

} // namespace snicholls::stats

#endif /* snicholls_core_stats_bootstrap_hpp */
