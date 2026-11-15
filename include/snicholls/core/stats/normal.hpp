//
//  normal.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::stats::normal_sampler`: standard normal draws from a full-range generator, with interchangeable methods (the inverse CDF, Box-Muller).
/// \ingroup core
///
/// **Overview.** Gaussian draws for Monte Carlo, random test matrices (`linalg::truncated_svd`'s `gaussian_sketch`)
/// and noise. Different jobs want different methods, so the method is a value the caller chooses, behind one concept:
///
/// - `normal_sampler<S, G>`: `s(g)` returns one standard normal `double` from a `full_range_generator` `g`.
/// - `normal_inverse_cdf`: `math::normal_quantile` of one uniform, `(2i + 1) 2^-53` for 52 random bits `i`: exact, inside
///   `(0, 1)` and symmetric about 1/2, so the draws reach ±8.21 and never infinity. One generator call per draw,
///   stateless, monotone in the uniform (so it maps a quasi-random point to a quasi-random normal), and the same on
///   every platform. The default.
/// - `normal_box_muller`: two uniforms give two independent normals, `sqrt(-2 ln u1) (cos, sin)(2π u2)`; the second
///   is kept for the next call, so a sampler is stateful (copy it per stream).
///
/// A method of the caller's own (a ziggurat, a device's generator, an antithetic or moment-matched pair) is any type
/// that satisfies `normal_sampler`, and drops in wherever one is taken.
///
/// \snippet stats/normal_demo.cpp normal
///
/// **Relationships.** `stats/random.hpp` (the generators, `uniform_real`), `math/constexpr_math.hpp`
/// (`normal_quantile`, `log`, `sin`, `cos`), `linalg/randomized_svd.hpp` (`gaussian_sketch`). Group: \ref core.
///
/// **Implementations.** `normal_inverse_cdf`: one call of the generator and one `normal_quantile` (a rational
/// approximation refined by Halley's step) per draw. `normal_box_muller`: two calls, a logarithm, a square root, a sine
/// and a cosine per pair.
///
/// **Guarantees.** `constexpr`, `noexcept` when the generator is, no allocation; the same seed gives the same draws on
/// every platform for `normal_inverse_cdf` (its arithmetic is core's); `normal_box_muller` uses core's `log`, `sin` and
/// `cos`, which follow `<cmath>` at run time, so its last bits may differ between standard libraries. Neither draw is
/// ever infinite: the uniforms stay inside `(0, 1)`.
///
/// **Errors.** None.
///
/// **Performance.** Each against the same draws written by hand over the same stream, after a bit-for-bit check, a
/// million draws (`benchmarks/stats/normal_benchmark.cpp`; Release, Apple clang 17, Xeon W-3245, 2026-11-16): the
/// inverse CDF 21.5 million draws a second (by hand 21.2), Box-Muller 82.8 million (by hand 81.3). Box-Muller is the
/// faster by four; the inverse CDF is the default for being the same on every platform and monotone.
///
/// **Tests and demos.** `tests/stats/normal_test.cpp`, `demos/stats/normal_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstdint>`, `math/constexpr_math.hpp`, `stats/random.hpp`.
///
/// **Known issues.** Draws are `double`; convert for `float`.
///
/// **Status.** Experimental, new in 0.9.0.

#ifndef snicholls_core_stats_normal_hpp
#define snicholls_core_stats_normal_hpp

#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stats/random.hpp>

#include <concepts>
#include <cstdint>

namespace snicholls::stats {

    namespace detail::normal {
        // A uniform in (0, 1), (2i + 1) 2^-53 for the top 52 bits i: every value is a double, the largest 1 - 2^-53
        // (with 53 bits, (i + 1/2) 2^-53 rounds to 1 at the top and its quantile is infinite)
        template <typename G>
        constexpr double open_uniform(G& g) noexcept(noexcept(g()))
        {
            return static_cast<double>((random_u64(g) >> 12) * 2 + 1) * 0x1.0p-53;
        }
    } // namespace detail::normal

    /// \brief A source of standard normal draws: `s(g)` takes what it needs from the generator `g`.
    /// \ingroup core
    /// \tparam S The sampler.
    /// \tparam G A `full_range_generator`.
    template <typename S, typename G>
    concept normal_sampler = full_range_generator<G> && requires(S& s, G& g) { { s(g) } -> std::convertible_to<double>; };

    /// \brief Standard normal draws by the inverse CDF: `math::normal_quantile` of one uniform in `(0, 1)`.
    /// \ingroup core
    struct normal_inverse_cdf {
        /// \brief One draw.
        /// \param g The generator: one call.
        /// \return A standard normal.
        template <full_range_generator G>
        [[nodiscard]] constexpr double operator()(G& g) const noexcept(noexcept(g()))
        {
            return math::normal_quantile(detail::normal::open_uniform(g));
        }
    };

    /// \brief Standard normal draws by Box-Muller: two uniforms give two normals, the second kept for the next call.
    /// \ingroup core
    struct normal_box_muller {
        /// \brief One draw.
        /// \param g The generator: two calls every other draw.
        /// \return A standard normal.
        template <full_range_generator G>
        [[nodiscard]] constexpr double operator()(G& g) noexcept(noexcept(g()))
        {
            if (has_spare_) {
                has_spare_ = false;
                return spare_;
            }
            const double u1 = detail::normal::open_uniform(g);   // in (0, 1): the logarithm is finite
            const double u2 = static_cast<double>(random_u64(g) >> 11) * 0x1.0p-53;
            const double r = math::sqrt(-2.0 * math::log(u1)), t = 2.0 * math::pi * u2;
            spare_ = r * math::sin(t);
            has_spare_ = true;
            return r * math::cos(t);
        }

    private:
        double spare_ = 0.0;
        bool has_spare_ = false;
    };

    static_assert(normal_sampler<normal_inverse_cdf, wyrand> && normal_sampler<normal_box_muller, wyrand>);

} // namespace snicholls::stats

#endif /* snicholls_core_stats_normal_hpp */
