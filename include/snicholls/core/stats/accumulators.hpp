//
//  accumulators.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Each is constant time per observation and mergeable, so partial results from several workers combine
//  exactly.
//

/// \file
/// \brief Streaming accumulators: count, sum, min, max, mean, variance, covariance, exponentially weighted moments, histogram.
/// \ingroup core
///
/// **Overview.** Live risk needs statistics that move with every tick without recomputing from
/// history. Each accumulator here takes one observation in constant time (`add`), combines with another
/// built from other observations (`merge`, so worker threads each build one and the results are merged),
/// and reports (`value`). The variance is Welford's, merged by Chan, Golub and LeVeque's formula, so it
/// does not lose precision the way a sum of squares does.
///
/// \snippet stats/accumulators_demo.cpp accumulators
///
/// | accumulator              | `value()`                               | merge             |
/// | ------------------------ | --------------------------------------- | ----------------- |
/// | `count`                  | observations                            | exact             |
/// | `sum<T>`                 | their sum (`compensated_sum<T>`: Kahan) | exact / Neumaier  |
/// | `minimum<T>`, `maximum<T>` | the extreme, `std::nullopt` if none   | exact             |
/// | `mean<T>`                | the mean                                | Chan              |
/// | `variance<T>`            | the sample variance (`mean()`, `standard_deviation()` too) | Chan |
/// | `covariance<T>`          | the sample covariance of pairs (`correlation()` too) | Chan |
/// | `ewma<T>`                | exponentially weighted mean and variance | not mergeable    |
/// | `histogram<T, Bins>`     | counts in fixed bins, with under and overflow | exact       |
/// | `collect<T>`             | every value, for any statistic afterwards | append           |
/// | `bootstrapped<Acc, B>`   | Acc's value and `B` Poisson-bootstrap replicates of it | per replicate |
///
/// Every accumulator is a `snicholls::stats::accumulator`, which `stats/pivot.hpp` uses for its cells.
///
/// **Relationships.** `stats/pivot.hpp`; `stats/bootstrap.hpp` for the uncertainty of a statistic.
/// Group: \ref core.
///
/// **Guarantees.** All but `collect` and `bootstrapped`'s `value()` are `constexpr`, `noexcept`, allocation-free and
/// trivially copyable. `add` is O(1); merging two
/// accumulators gives what one accumulator would give on both sets of observations, exactly for
/// `count`, `minimum`, `maximum` and `histogram`, and up to rounding for the moments.
///
/// **Errors.** None: an empty accumulator's mean and variance are NaN, its minimum is `std::nullopt`.
///
/// **Performance.** `benchmarks/stats/accumulators_benchmark.cpp`: 1,000,000 observations into `variance` and
/// `covariance`, Release, Apple clang 17, Xeon W-3245, 2026-11-08: 6.39 ms, against 6.05 ms for the same updates
/// written by hand. The inner loops compile to identical instructions; two identical hand-written copies
/// measure 6.07 and 6.36 ms, so the difference is code placement, not cost.
///
/// **Tests and demos.** `tests/stats/accumulators_test.cpp`, `demos/stats/accumulators_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cmath>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<optional>`, `<span>`,
/// `<vector>`, `stats/bootstrap.hpp`, `stats/random.hpp`.
///
/// **Known issues.** `ewma` depends on the order of observations and cannot be merged. `histogram`'s bins
/// are fixed at construction.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_stats_accumulators_hpp
#define snicholls_core_stats_accumulators_hpp

#include <snicholls/core/stats/bootstrap.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace snicholls::stats {

    /// \brief A streaming accumulator of `T`: `add(t)`, `value()`, and `merge` with another of its kind.
    /// \ingroup core
    template <typename A, typename T>
    concept accumulator = std::copyable<A> && requires(A a, const A b, const T t) {
        a.add(t);
        b.value();
    };

    /// \brief An accumulator whose partial results can be combined: parallel workers each build one.
    /// \ingroup core
    template <typename A>
    concept mergeable = requires(A a, const A b) { a.merge(b); };

    /// \brief The number of observations; takes any value.
    /// \ingroup core
    struct count {
        std::uint64_t n = 0;   ///< The count.

        /// \brief Count one observation; its value does not matter.
        /// \tparam T Any type.
        template <typename T>
        constexpr void add(const T&) noexcept
        {
            ++n;
        }
        /// \brief Add another's count.
        /// \param other The other.
        constexpr void merge(const count& other) noexcept { n += other.n; }
        /// \brief The count.
        /// \return It.
        [[nodiscard]] constexpr std::uint64_t value() const noexcept { return n; }
    };

    /// \brief The sum.
    /// \ingroup core
    /// \tparam T The value type.
    template <typename T>
    struct sum {
        T total{};   ///< The sum so far.

        /// \brief Add one value.
        /// \param x The value.
        constexpr void add(const T& x) noexcept { total += x; }
        /// \brief Add another's sum.
        /// \param other The other.
        constexpr void merge(const sum& other) noexcept { total += other.total; }
        /// \brief The sum.
        /// \return It.
        [[nodiscard]] constexpr T value() const noexcept { return total; }
    };

    /// \brief A sum with Neumaier's compensation: the rounding error of each addition is carried, so long sums stay exact to the last bit or so.
    /// \ingroup core
    /// \tparam T A floating-point type.
    template <std::floating_point T>
    struct compensated_sum {
        T total{};         ///< The running sum.
        T compensation{};  ///< The rounding error not yet in it.

        /// \brief Add one value.
        /// \param x The value.
        constexpr void add(T x) noexcept
        {
            // Neumaier: the low part lost when the smaller of the two is added to the larger
            const T t = total + x;
            const T abs_total = total < 0 ? -total : total;
            const T abs_x = x < 0 ? -x : x;
            compensation += abs_total >= abs_x ? (total - t) + x : (x - t) + total;
            total = t;
        }
        /// \brief Add another's sum.
        /// \param other The other.
        constexpr void merge(const compensated_sum& other) noexcept
        {
            add(other.total);
            compensation += other.compensation;
        }
        /// \brief The sum.
        /// \return It, compensated.
        [[nodiscard]] constexpr T value() const noexcept { return total + compensation; }
    };

    /// \brief The smallest value.
    /// \ingroup core
    /// \tparam T The value type, ordered.
    template <std::totally_ordered T>
    struct minimum {
        std::optional<T> best;   ///< The smallest so far.

        /// \brief Consider one value.
        /// \param x The value.
        constexpr void add(const T& x) noexcept(std::is_nothrow_copy_assignable_v<T>)
        {
            if (!best || x < *best)
                best = x;
        }
        /// \brief Consider another's.
        /// \param other The other.
        constexpr void merge(const minimum& other) noexcept(std::is_nothrow_copy_assignable_v<T>)
        {
            if (other.best)
                add(*other.best);
        }
        /// \brief The smallest value, or none.
        /// \return It.
        [[nodiscard]] constexpr std::optional<T> value() const noexcept(std::is_nothrow_copy_constructible_v<T>) { return best; }
    };

    /// \brief The largest value.
    /// \ingroup core
    /// \tparam T The value type, ordered.
    template <std::totally_ordered T>
    struct maximum {
        std::optional<T> best;   ///< The largest so far.

        /// \brief Consider one value.
        /// \param x The value.
        constexpr void add(const T& x) noexcept(std::is_nothrow_copy_assignable_v<T>)
        {
            if (!best || *best < x)
                best = x;
        }
        /// \brief Consider another's.
        /// \param other The other.
        constexpr void merge(const maximum& other) noexcept(std::is_nothrow_copy_assignable_v<T>)
        {
            if (other.best)
                add(*other.best);
        }
        /// \brief The largest value, or none.
        /// \return It.
        [[nodiscard]] constexpr std::optional<T> value() const noexcept(std::is_nothrow_copy_constructible_v<T>) { return best; }
    };

    /// \brief The mean, by Welford's update.
    /// \ingroup core
    /// \tparam T A floating-point type.
    template <std::floating_point T>
    struct mean {
        std::uint64_t n = 0;   ///< Observations.
        T m{};                 ///< Their mean.

        /// \brief Add one value.
        /// \param x The value.
        constexpr void add(T x) noexcept
        {
            ++n;
            m += (x - m) / static_cast<T>(n);
        }
        /// \brief Combine with another (Chan et al.).
        /// \param other The other.
        constexpr void merge(const mean& other) noexcept
        {
            if (other.n == 0)
                return;
            const std::uint64_t total = n + other.n;
            m += (other.m - m) * static_cast<T>(other.n) / static_cast<T>(total);
            n = total;
        }
        /// \brief The mean; NaN with no observations.
        /// \return It.
        [[nodiscard]] constexpr T value() const noexcept { return n == 0 ? std::numeric_limits<T>::quiet_NaN() : m; }
    };

    /// \brief The mean and sample variance, by Welford's update; merged by Chan, Golub and LeVeque.
    /// \ingroup core
    /// \tparam T A floating-point type.
    template <std::floating_point T>
    struct variance {
        std::uint64_t n = 0;   ///< Observations.
        T m{};                 ///< Their mean.
        T m2{};                ///< The sum of squared deviations from the mean.

        /// \brief Add one value.
        /// \param x The value.
        constexpr void add(T x) noexcept
        {
            ++n;
            const T delta = x - m;
            m += delta / static_cast<T>(n);
            m2 += delta * (x - m);
        }
        /// \brief Combine with another.
        /// \param other The other.
        constexpr void merge(const variance& other) noexcept
        {
            if (other.n == 0)
                return;
            const std::uint64_t total = n + other.n;
            const T delta = other.m - m;
            m += delta * static_cast<T>(other.n) / static_cast<T>(total);
            m2 += other.m2 + delta * delta * static_cast<T>(n) * static_cast<T>(other.n) / static_cast<T>(total);
            n = total;
        }
        /// \brief The sample variance (divided by n - 1); NaN with fewer than two observations.
        /// \return It.
        [[nodiscard]] constexpr T value() const noexcept { return n < 2 ? std::numeric_limits<T>::quiet_NaN() : m2 / static_cast<T>(n - 1); }
        /// \brief The mean.
        /// \return It; NaN with none.
        [[nodiscard]] constexpr T mean() const noexcept { return n == 0 ? std::numeric_limits<T>::quiet_NaN() : m; }
        /// \brief The sample standard deviation.
        /// \return It.
        [[nodiscard]] T standard_deviation() const noexcept { return std::sqrt(value()); }
        /// \brief The population variance (divided by n).
        /// \return It; NaN with none.
        [[nodiscard]] constexpr T population_variance() const noexcept { return n == 0 ? std::numeric_limits<T>::quiet_NaN() : m2 / static_cast<T>(n); }
    };

    /// \brief The sample covariance of pairs `(x, y)`, with both variances; merged by Chan's formula.
    /// \ingroup core
    /// \tparam T A floating-point type.
    template <std::floating_point T>
    struct covariance {
        std::uint64_t n = 0;   ///< Pairs.
        T mx{};                ///< The mean of x.
        T my{};                ///< The mean of y.
        T cxy{};               ///< The sum of products of deviations.
        T m2x{};               ///< The sum of squared deviations of x.
        T m2y{};               ///< The sum of squared deviations of y.

        /// \brief Add one pair.
        /// \param x The first value.
        /// \param y The second.
        constexpr void add(T x, T y) noexcept
        {
            ++n;
            const T dx = x - mx;
            mx += dx / static_cast<T>(n);
            const T dy = y - my;
            my += dy / static_cast<T>(n);
            cxy += dx * (y - my);
            m2x += dx * (x - mx);
            m2y += dy * (y - my);
        }
        /// \brief Add one pair given as anything with `first` and `second` (`std::pair`).
        /// \tparam P The pair's type.
        /// \param p The pair.
        template <typename P>
            requires requires(const P& q) {
                T(q.first);
                T(q.second);
            }
        constexpr void add(const P& p) noexcept
        {
            add(static_cast<T>(p.first), static_cast<T>(p.second));
        }
        /// \brief Combine with another.
        /// \param other The other.
        constexpr void merge(const covariance& other) noexcept
        {
            if (other.n == 0)
                return;
            const std::uint64_t total = n + other.n;
            const T dx = other.mx - mx, dy = other.my - my;
            const T w = static_cast<T>(n) * static_cast<T>(other.n) / static_cast<T>(total);
            cxy += other.cxy + dx * dy * w;
            m2x += other.m2x + dx * dx * w;
            m2y += other.m2y + dy * dy * w;
            mx += dx * static_cast<T>(other.n) / static_cast<T>(total);
            my += dy * static_cast<T>(other.n) / static_cast<T>(total);
            n = total;
        }
        /// \brief The sample covariance; NaN with fewer than two pairs.
        /// \return It.
        [[nodiscard]] constexpr T value() const noexcept { return n < 2 ? std::numeric_limits<T>::quiet_NaN() : cxy / static_cast<T>(n - 1); }
        /// \brief The correlation; NaN when either variance is zero or there are fewer than two pairs.
        /// \return It.
        [[nodiscard]] T correlation() const noexcept
        {
            return (n < 2 || m2x <= 0 || m2y <= 0) ? std::numeric_limits<T>::quiet_NaN() : cxy / std::sqrt(m2x * m2y);
        }
        /// \brief The regression slope of y on x: `cov / var(x)`. For a portfolio's beta to an index, add `(index, portfolio)`.
        /// \return It; NaN when x does not vary.
        [[nodiscard]] constexpr T beta() const noexcept { return (n < 2 || m2x <= 0) ? std::numeric_limits<T>::quiet_NaN() : cxy / m2x; }
    };

    /// \brief The exponentially weighted mean and variance (RiskMetrics style): each observation weighted `alpha`, the past `1 - alpha`.
    /// \ingroup core
    /// \tparam T A floating-point type.
    template <std::floating_point T>
    struct ewma {
        T alpha = T(0.06);   ///< The weight of a new observation; 0.06 is RiskMetrics' daily lambda of 0.94.
        T m{};               ///< The weighted mean.
        T var{};             ///< The weighted variance about it.
        bool started = false;   ///< Whether any observation has arrived.

        /// \brief Add one value.
        /// \param x The value.
        constexpr void add(T x) noexcept
        {
            if (!started) {
                m = x;
                started = true;
                return;
            }
            const T delta = x - m;
            m += alpha * delta;
            var = (1 - alpha) * (var + alpha * delta * delta);
        }
        /// \brief The weighted mean; NaN before any observation.
        /// \return It.
        [[nodiscard]] constexpr T value() const noexcept { return started ? m : std::numeric_limits<T>::quiet_NaN(); }
        /// \brief The weighted variance.
        /// \return It; 0 after one observation, NaN before any.
        [[nodiscard]] constexpr T variance() const noexcept { return started ? var : std::numeric_limits<T>::quiet_NaN(); }
        /// \brief The weighted volatility.
        /// \return `sqrt(variance())`.
        [[nodiscard]] T volatility() const noexcept { return std::sqrt(variance()); }
    };

    /// \brief Counts in `Bins` equal bins over `[lo, hi)`, with counts below and above.
    /// \ingroup core
    /// \tparam T    A floating-point type.
    /// \tparam Bins The number of bins, at least 1.
    template <std::floating_point T, std::size_t Bins>
    struct histogram {
        static_assert(Bins >= 1, "snicholls::stats::histogram: at least one bin");
        T lo = 0;                                 ///< The lower edge of the first bin.
        T hi = 1;                                 ///< The upper edge of the last bin.
        std::array<std::uint64_t, Bins> bins{};   ///< The counts.
        std::uint64_t below = 0;                  ///< Values below `lo` (and NaN).
        std::uint64_t above = 0;                  ///< Values at or above `hi`.

        /// \brief Count one value.
        /// \param x The value.
        constexpr void add(T x) noexcept
        {
            if (!(x >= lo)) {
                ++below;
                return;
            }
            if (x >= hi) {
                ++above;
                return;
            }
            auto i = static_cast<std::size_t>((x - lo) / (hi - lo) * static_cast<T>(Bins));
            ++bins[i < Bins ? i : Bins - 1];
        }
        /// \brief Add another's counts; the edges must be the same.
        /// \param other The other.
        constexpr void merge(const histogram& other) noexcept
        {
            for (std::size_t i = 0; i < Bins; ++i)
                bins[i] += other.bins[i];
            below += other.below;
            above += other.above;
        }
        /// \brief The counts.
        /// \return The bins.
        [[nodiscard]] constexpr const std::array<std::uint64_t, Bins>& value() const noexcept { return bins; }
        /// \brief The lower edge of bin `i`.
        /// \param i The bin.
        /// \return The edge.
        [[nodiscard]] constexpr T edge(std::size_t i) const noexcept { return lo + (hi - lo) * static_cast<T>(i) / static_cast<T>(Bins); }
    };

    /// \brief Every value, kept: the cell of a pivot partitioned rather than aggregated (a raw pivot table),
    /// so that any statistic, a median, a drawdown or a full bootstrap, can be computed per cell afterwards.
    /// \ingroup core
    /// \tparam T The value type.
    template <typename T>
    struct collect {
        std::vector<T> values;   ///< The values, in the order added.

        /// \brief Keep one value.
        /// \param x The value.
        /// \throws std::bad_alloc.
        void add(const T& x) { values.push_back(x); }
        /// \brief Append another's values.
        /// \param other The other.
        /// \throws std::bad_alloc.
        void merge(const collect& other) { values.insert(values.end(), other.values.begin(), other.values.end()); }
        /// \brief The values.
        /// \return A view, valid until the next `add`.
        [[nodiscard]] std::span<const T> value() const noexcept { return values; }
    };

    /// \brief An accumulator with its own online Poisson bootstrap: `B` replicate accumulators, each row added to each a
    /// Poisson(1) number of times, so a statistic and its uncertainty come out of the same single pass.
    ///
    /// Merging merges replicate by replicate, which the Poisson bootstrap allows. A pivot reseeds every cell's copy
    /// when the cell is first used, so cells draw independently.
    /// \ingroup core
    /// \tparam Acc The accumulator bootstrapped (`mean<double>`, `variance<double>`, `sum<double>`, ...), whose `value()` is a number.
    /// \tparam B   The number of replicates.
    template <typename Acc, std::size_t B = 200>
    struct bootstrapped {
        static_assert(B >= 2, "snicholls::stats::bootstrapped: at least two replicates");
        Acc estimate{};                     ///< The accumulator on every row once.
        std::array<Acc, B> replicates{};    ///< The replicates.
        wyrand rng{0x5eed};                 ///< Where the Poisson weights come from.

        /// \brief Use a different stream of weights: a pivot calls this for each new cell.
        /// \param seed The seed.
        constexpr void reseed(std::uint64_t seed) noexcept { rng = wyrand(seed); }

        /// \brief Add one value: once to the estimate, a Poisson(1) number of times to each replicate.
        /// \tparam T The value's type.
        /// \param x The value.
        template <typename T>
        constexpr void add(const T& x)
        {
            estimate.add(x);
            for (std::size_t b = 0; b < B; b += 2) {
                const std::uint64_t bits = rng();
                for (std::uint32_t k = detail::poisson1_from(static_cast<std::uint32_t>(bits)); k > 0; --k)
                    replicates[b].add(x);
                if (b + 1 < B)
                    for (std::uint32_t k = detail::poisson1_from(static_cast<std::uint32_t>(bits >> 32)); k > 0; --k)
                        replicates[b + 1].add(x);
            }
        }

        /// \brief Merge another, replicate by replicate.
        /// \param other The other.
        constexpr void merge(const bootstrapped& other)
            requires mergeable<Acc>
        {
            estimate.merge(other.estimate);
            for (std::size_t b = 0; b < B; ++b)
                replicates[b].merge(other.replicates[b]);
        }

        /// \brief The bootstrap distribution: the estimate and each replicate's value.
        /// \return It; NaN replicates (an empty one, for a mean) are kept.
        /// \throws std::bad_alloc.
        [[nodiscard]] bootstrap_distribution value() const
        {
            bootstrap_distribution out;
            out.estimate = static_cast<double>(estimate.value());
            out.replicates.reserve(B);
            for (const Acc& r : replicates)
                out.replicates.push_back(static_cast<double>(r.value()));
            return out;
        }
    };

} // namespace snicholls::stats

#endif /* snicholls_core_stats_accumulators_hpp */
