//
//  resample.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::stats::systematic_resample`, `stratified_resample`, `multinomial_resample`, `residual_resample`, `effective_sample_size` and log-weight normalisation: the resampling step of particle filters.
/// \ingroup core
///
/// **Overview.** A particle filter's weights degenerate: after a few updates most particles carry almost no weight.
/// Resampling draws `M` indices with probabilities proportional to the weights, so the heavy particles are copied
/// and the light ones are dropped. The schemes differ in their variance and in how many random numbers they use:
///
/// | Scheme | Uniforms | Variance of the copy counts |
/// |---|---|---|
/// | `systematic_resample` | 1: `uᵢ = (i + U) / M` | lowest in practice; the default |
/// | `stratified_resample` | M: `uᵢ = (i + Uᵢ) / M` | below multinomial's |
/// | `multinomial_resample` | M + 1 (sorted by exponential spacings, no sort) | the reference |
/// | `residual_resample` | `⌊M wᵢ⌋` copies deterministically, the remainder by one of the above | low |
///
/// Weights need not be normalised: each function divides by their sum. `effective_sample_size(w)` is
/// `(Σ w)² / Σ w²`; `needs_resampling(w, fraction)` says whether it has fallen below `fraction · N` (Stone Soup's
/// `ESSResampler`). `log_sum_exp` and `normalise_log_weights` keep weights in log space without underflow.
///
/// The randomness is the caller's `full_range_generator` (`wyrand` and its `stream(seed, k)`), so a run reproduces
/// on every platform, and a device thread can own a stream.
///
/// \snippet stats/resample_demo.cpp resample
///
/// **Relationships.** `stats/random.hpp` (`uniform_real`), `math/constexpr_math.hpp` (`log`, `exp`). The
/// particle filters above core (and their device forms) use these. Group: \ref core.
///
/// **Implementations.** Each scheme walks the cumulative weights once against its increasing sequence of uniforms,
/// O(N + M), with no sort and no allocation.
///
/// **Guarantees.** `constexpr`, `noexcept` when the generator is, no allocation. The indices come out in
/// non-decreasing order (for `residual_resample`, the deterministic copies and the remainder are each in order). A particle of zero weight is never drawn.
///
/// **Errors.** None. With a zero, infinite or empty total weight, or any negative or NaN weight, no draw is
/// possible: the functions return `false` and leave `indices` unspecified.
///
/// **Performance.** Systematic resampling of 2¹⁶ particles against the same walk written by hand, after checking
/// every index agrees (`benchmarks/stats/resample_benchmark.cpp`; Release, Apple clang 17, Xeon W-3245, 2026-11-23):
/// 77.0 million particles a second (by hand 79.3).
///
/// **Tests and demos.** `tests/stats/resample_test.cpp`, `demos/stats/resample_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<limits>`, `<span>`, `math/constexpr_math.hpp`, `stats/random.hpp`.
///
/// **Known issues.** Weights are summed in their own type; for millions of `float` weights, normalise in `double`.
///
/// **Status.** Experimental, new in 0.19.0.

#ifndef snicholls_core_stats_resample_hpp
#define snicholls_core_stats_resample_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/random.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace snicholls::stats {

    /// \brief How `residual_resample` draws the remainder.
    /// \ingroup core
    enum class residual_method : std::uint8_t {
        systematic = 0,   ///< One uniform.
        stratified = 1,   ///< One uniform per draw, each in its stratum.
        multinomial = 2,  ///< Independent draws.
    };

    namespace detail::resample {
        // The sum of the weights; NaN if any weight is negative or NaN, so that usable() refuses it
        template <typename T>
        constexpr T total(std::span<const T> w) noexcept
        {
            T s(0);
            for (const T x : w) {
                if (!(x >= T(0)))
                    return std::numeric_limits<T>::quiet_NaN();
                s += x;
            }
            return s;
        }

        template <typename T>
        constexpr bool usable(T s) noexcept
        {
            return s > T(0) && s < std::numeric_limits<T>::infinity();
        }

        // Walks the cumulative weights against the increasing u(i) in [0, 1), writing M indices
        template <typename T, typename U>
        constexpr void walk(std::span<const T> w, T sum, std::span<std::size_t> out, U&& u) noexcept(noexcept(u(std::size_t{})))
        {
            std::size_t j = 0;
            T cum = w.empty() ? T(0) : w[0] / sum;
            const std::size_t last = w.size() - 1;
            for (std::size_t i = 0; i < out.size(); ++i) {
                const T ui = u(i);
                while (ui >= cum && j < last)
                    cum += w[++j] / sum;
                // Never land on a particle of zero weight (rounding at the end of the cumulative sum)
                while (w[j] == T(0) && j > 0)
                    --j;
                out[i] = j;
            }
        }
    } // namespace detail::resample

    /// \brief The effective sample size, `(Σ w)² / Σ w²`.
    /// \ingroup core
    /// \tparam T The weight type.
    /// \param weights The weights, not necessarily normalised.
    /// \return Between 1 (one particle has all the weight) and N (equal weights); 0 for no weight.
    template <typename T>
    [[nodiscard]] constexpr T effective_sample_size(std::span<const T> weights) noexcept
    {
        T s(0), s2(0);
        for (const T x : weights) {
            s += x;
            s2 += x * x;
        }
        return s2 > T(0) ? s * s / s2 : T(0);
    }

    /// \brief Whether the effective sample size is below `fraction` of the particle count (Stone Soup's `ESSResampler`).
    /// \tparam T The weight type. \param weights The weights. \param fraction The threshold, such as 0.5.
    /// \ingroup core
    /// \return True if the particles should be resampled.
    template <typename T>
    [[nodiscard]] constexpr bool needs_resampling(std::span<const T> weights, T fraction) noexcept
    {
        return effective_sample_size(weights) < fraction * static_cast<T>(weights.size());
    }

    /// \brief `log Σ exp(xᵢ)`, computed without overflow or underflow.
    /// \tparam T The number type. \param x The log values.
    /// \ingroup core
    /// \return It; minus infinity for no values or all minus infinity.
    template <typename T>
    [[nodiscard]] constexpr T log_sum_exp(std::span<const T> x) noexcept
    {
        T m = -std::numeric_limits<T>::infinity();
        for (const T v : x)
            if (v > m)
                m = v;
        if (m == -std::numeric_limits<T>::infinity() || m == std::numeric_limits<T>::infinity())
            return m;
        T s(0);
        for (const T v : x)
            s += static_cast<T>(math::exp(v - m));
        return m + static_cast<T>(math::log(s));
    }

    /// \brief Normalises log weights in place so that their exponentials sum to 1.
    /// \tparam T The number type. \param log_weights The log weights, changed.
    /// \ingroup core
    /// \return The log of the normalising constant (the log marginal likelihood of an update).
    template <typename T>
    constexpr T normalise_log_weights(std::span<T> log_weights) noexcept
    {
        const T z = log_sum_exp(std::span<const T>(log_weights));
        if (z == -std::numeric_limits<T>::infinity() || z == std::numeric_limits<T>::infinity())
            return z;
        for (T& v : log_weights)
            v -= z;
        return z;
    }

    /// \brief Systematic resampling: one uniform `U`, draws at `(i + U) / M`.
    /// \ingroup core
    /// \tparam T The weight type.
    /// \tparam G A `full_range_generator`.
    /// \param weights The N weights, not necessarily normalised.
    /// \param indices Receives M indices into the weights, non-decreasing.
    /// \param g The generator.
    /// \return Whether the weights allowed a draw.
    template <typename T, full_range_generator G>
    constexpr bool systematic_resample(std::span<const T> weights, std::span<std::size_t> indices, G& g) noexcept(noexcept(g()))
    {
        const T sum = detail::resample::total(weights);
        if (!detail::resample::usable(sum) || indices.empty())
            return detail::resample::usable(sum);
        const T m = static_cast<T>(indices.size());
        const T u0 = static_cast<T>(uniform_real(g));
        detail::resample::walk(weights, sum, indices, [&](std::size_t i) noexcept { return (static_cast<T>(i) + u0) / m; });
        return true;
    }

    /// \brief Stratified resampling: one uniform per stratum, draws at `(i + Uᵢ) / M`.
    /// \ingroup core
    /// \tparam T The weight type.
    /// \tparam G A `full_range_generator`.
    /// \param weights The N weights, not necessarily normalised.
    /// \param indices Receives M indices, non-decreasing.
    /// \param g The generator.
    /// \return Whether the weights allowed a draw.
    template <typename T, full_range_generator G>
    constexpr bool stratified_resample(std::span<const T> weights, std::span<std::size_t> indices, G& g) noexcept(noexcept(g()))
    {
        const T sum = detail::resample::total(weights);
        if (!detail::resample::usable(sum) || indices.empty())
            return detail::resample::usable(sum);
        const T m = static_cast<T>(indices.size());
        detail::resample::walk(weights, sum, indices, [&](std::size_t i) noexcept(noexcept(g())) { return (static_cast<T>(i) + static_cast<T>(uniform_real(g))) / m; });
        return true;
    }

    /// \brief Multinomial resampling: M independent draws, generated already sorted (normalised partial sums of
    /// exponentials), so no sort is needed.
    /// \ingroup core
    /// \tparam T The weight type.
    /// \tparam G A `full_range_generator`.
    /// \param weights The N weights, not necessarily normalised.
    /// \param indices Receives M indices, non-decreasing.
    /// \param g The generator.
    /// \return Whether the weights allowed a draw.
    template <typename T, full_range_generator G>
    constexpr bool multinomial_resample(std::span<const T> weights, std::span<std::size_t> indices, G& g) noexcept(noexcept(g()))
    {
        const T sum = detail::resample::total(weights);
        if (!detail::resample::usable(sum) || indices.empty())
            return detail::resample::usable(sum);
        // The order statistics of M uniforms are S_k / S_{M+1} for S_k the partial sums of M + 1 exponentials. Draw
        // S_{M+1} first as a whole (a sum of M + 1 exponentials is not cheaper to draw directly), so draw the M + 1
        // exponentials twice from a copy of the generator: once to total them, once to walk.
        auto exp1 = [](G& gg) { return -static_cast<T>(math::log(T(1) - static_cast<T>(uniform_real(gg)))); };
        G copy = g;
        T total(0);
        for (std::size_t i = 0; i <= indices.size(); ++i)
            total += exp1(copy);
        T partial(0);
        detail::resample::walk(weights, sum, indices, [&](std::size_t) {
            partial += exp1(g);
            return partial / total;
        });
        static_cast<void>(exp1(g));  // the (M + 1)-th, so g ends where the copy did
        return true;
    }

    /// \brief Residual resampling: `⌊M wᵢ / Σw⌋` copies of each particle, the remaining draws by `method` on the
    /// residual weights.
    /// \ingroup core
    /// \tparam T The weight type.
    /// \tparam G A `full_range_generator`.
    /// \param weights The N weights, not necessarily normalised.
    /// \param residual Working space for the N residual weights.
    /// \param indices Receives M indices: the deterministic copies in order, then the remainder in order.
    /// \param g The generator.
    /// \param method How the remainder is drawn.
    /// \return Whether the weights allowed a draw (and `residual` was long enough).
    template <typename T, full_range_generator G>
    constexpr bool residual_resample(std::span<const T> weights, std::span<T> residual, std::span<std::size_t> indices, G& g,
                                     residual_method method = residual_method::systematic) noexcept(noexcept(g()))
    {
        const T sum = detail::resample::total(weights);
        if (!detail::resample::usable(sum) || residual.size() < weights.size())
            return false;
        const T m = static_cast<T>(indices.size());
        std::size_t k = 0;
        for (std::size_t j = 0; j < weights.size(); ++j) {
            const T expected = m * (weights[j] / sum);  // the ratio first: no overflow for weights near the largest double
            const auto copies = static_cast<std::size_t>(math::floor(expected));
            for (std::size_t c = 0; c < copies && k < indices.size(); ++c)
                indices[k++] = j;
            residual[j] = expected - static_cast<T>(copies);
        }
        const auto rest = indices.subspan(k);
        if (rest.empty())
            return true;
        const std::span<const T> r(residual.data(), weights.size());
        if (!detail::resample::usable(detail::resample::total(r))) {
            // Rounding left draws but no residual weight: give them to the heaviest particle
            std::size_t best = 0;
            for (std::size_t j = 1; j < weights.size(); ++j)
                if (weights[j] > weights[best])
                    best = j;
            for (auto& x : rest)
                x = best;
        } else if (method == residual_method::systematic)
            systematic_resample(r, rest, g);
        else if (method == residual_method::stratified)
            stratified_resample(r, rest, g);
        else
            multinomial_resample(r, rest, g);
        return true;
    }

} // namespace snicholls::stats

#endif /* snicholls_core_stats_resample_hpp */
