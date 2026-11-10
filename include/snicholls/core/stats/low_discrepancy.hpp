//
//  low_discrepancy.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Sobol from Joe and Kuo's direction numbers (detail/sobol_directions.hpp, generated), Halton from the
//  primes, both constexpr.
//

/// \file
/// \brief Sobol and Halton sequences: points that fill the unit cube evenly, for faster-converging Monte Carlo.
/// \ingroup core
///
/// **Overview.** Monte Carlo with random points has error proportional to `1 / sqrt(N)`. Points chosen to
/// fill the cube evenly (a low-discrepancy sequence) bring that close to `1 / N` for smooth integrands of
/// modest effective dimension: the same accuracy from far fewer points, which is the speed-up for option
/// pricing, risk integrals and any expectation over a few factors.
///
/// \snippet stats/low_discrepancy_demo.cpp sobol
///
/// - `sobol<D>`: Joe and Kuo's construction (new-joe-kuo-6.21201), up to 1024 dimensions, 32 bits, in
///   Gray-code order; the first 2^m points of each dimension put exactly one point in each interval
///   of width 2^-m. Prefer `N = 2^m` points. Built with a generator, it applies a random digital shift:
///   each shifted copy is still balanced and is uniformly distributed, so a few independent copies give
///   an unbiased estimate with an error bar (randomised quasi-Monte Carlo).
/// - `halton<D>`: radical inverses in the first D primes; simple and fine in a few dimensions,
///   correlated across dimensions above about ten, where Sobol is the better choice.
///
/// Both are `constexpr`: a table of points can be built at compile time.
///
/// **Relationships.** `stats/random.hpp` for the shift's randomness; `stats/bootstrap.hpp`.
/// Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept`, no allocation; a `sobol<D>` holds `2D` 32-bit words and an
/// index. The direction numbers are computed once per `D` at compile time.
///
/// **Errors.** None: `D` above 1024 is a compile error; more than 2^32 Sobol points is a precondition
/// violation (asserted).
///
/// **Performance.** `next()` is one XOR per dimension; not benchmarked separately.
///
/// **Tests and demos.** `tests/stats/low_discrepancy_test.cpp`, `demos/stats/low_discrepancy_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cassert>`, `<cstddef>`, `<cstdint>`, `<utility>`,
/// `detail/sobol_directions.hpp` (generated, 73 KB), `stats/random.hpp`.
///
/// **Known issues.** Unscrambled Sobol starts at the origin; skip it (`skip(1)`) where a zero breaks the
/// integrand (an inverse normal), or use the shifted form. Resampling a dataset with quasi-random
/// indices does little: a bootstrap's dimension is the dataset's size; see `stats/bootstrap.hpp`.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_stats_low_discrepancy_hpp
#define snicholls_core_stats_low_discrepancy_hpp

#include <snicholls/core/detail/sobol_directions.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace snicholls::stats {

    namespace detail {

        // Joe and Kuo's recurrence for dimension D (0-based): v_k = m_k << (32 - k) for k <= s, then
        // v_k = v_{k-s} ^ (v_{k-s} >> s) ^ (a_1 v_{k-1}) ^ ... ^ (a_{s-1} v_{k-s+1}).
        template <std::size_t D>
        consteval std::array<std::uint32_t, 32> sobol_dimension() noexcept
        {
            std::array<std::uint32_t, 32> v{};
            if constexpr (D == 0) {
                for (std::size_t k = 0; k < 32; ++k)
                    v[k] = std::uint32_t{1} << (31 - k);
            } else {
                const std::size_t s = snicholls::detail::sobol_degree[D - 1];
                const std::uint32_t a = snicholls::detail::sobol_coefficients[D - 1];
                const std::size_t offset = snicholls::detail::sobol_m_offset[D - 1];
                for (std::size_t k = 0; k < 32; ++k) {
                    if (k < s) {
                        v[k] = static_cast<std::uint32_t>(snicholls::detail::sobol_m[offset + k]) << (31 - k);
                    } else {
                        std::uint32_t x = v[k - s] ^ (v[k - s] >> s);
                        for (std::size_t j = 1; j < s; ++j)
                            if (((a >> (s - 1 - j)) & 1u) != 0)
                                x ^= v[k - j];
                        v[k] = x;
                    }
                }
            }
            return v;
        }

        // One constant evaluation per dimension, so a thousand dimensions stay inside every compiler's step limit
        template <std::size_t D>
        inline constexpr std::array<std::uint32_t, 32> sobol_dimension_v = sobol_dimension<D>();

        template <std::size_t... Ds>
        consteval std::array<std::array<std::uint32_t, 32>, sizeof...(Ds)> sobol_table(std::index_sequence<Ds...>) noexcept
        {
            return {sobol_dimension_v<Ds>...};
        }

        template <std::size_t Dims>
        inline constexpr auto sobol_v = sobol_table(std::make_index_sequence<Dims>{});

        template <std::size_t N>
        consteval std::array<std::uint32_t, N> first_primes() noexcept
        {
            std::array<std::uint32_t, N> primes{};
            std::size_t found = 0;
            for (std::uint32_t candidate = 2; found < N; ++candidate) {
                bool prime = true;
                for (std::size_t i = 0; i < found && primes[i] * primes[i] <= candidate; ++i)
                    if (candidate % primes[i] == 0) {
                        prime = false;
                        break;
                    }
                if (prime)
                    primes[found++] = candidate;
            }
            return primes;
        }

    } // namespace detail

    /// \brief The radical inverse of `i` in `base`: its digits mirrored about the point, `0.d0 d1 d2...`.
    /// \ingroup core
    /// \param base The base, at least 2.
    /// \param i    The index.
    /// \return A value in `[0, 1)`.
    [[nodiscard]] constexpr double radical_inverse(std::uint64_t base, std::uint64_t i) noexcept
    {
        assert(base >= 2);
        double result = 0;
        double scale = 1.0 / static_cast<double>(base);
        while (i != 0) {
            result += scale * static_cast<double>(i % base);
            i /= base;
            scale /= static_cast<double>(base);
        }
        return result;
    }

    /// \brief The Halton sequence in `Dims` dimensions: dimension `d` is the radical inverse in the d-th prime.
    /// \ingroup core
    /// \tparam Dims The number of dimensions, at least 1.
    template <std::size_t Dims>
    class halton {
        static_assert(Dims >= 1, "snicholls::stats::halton: at least one dimension");

    public:
        /// \brief A sequence starting at point `first`; the point at 0 is the origin, so 1 is a common start.
        /// \param first The index of the first point.
        constexpr explicit halton(std::uint64_t first = 1) noexcept : index_(first) {}

        /// \brief Point `i`, without moving the sequence.
        /// \param i The index.
        /// \return The point.
        [[nodiscard]] static constexpr std::array<double, Dims> point(std::uint64_t i) noexcept
        {
            std::array<double, Dims> p{};
            for (std::size_t d = 0; d < Dims; ++d)
                p[d] = radical_inverse(primes_[d], i);
            return p;
        }

        /// \brief The next point.
        /// \return The point.
        constexpr std::array<double, Dims> next() noexcept { return point(index_++); }

        /// \brief The index of the next point.
        /// \return It.
        [[nodiscard]] constexpr std::uint64_t index() const noexcept { return index_; }

    private:
        static constexpr std::array<std::uint32_t, Dims> primes_ = detail::first_primes<Dims>();
        std::uint64_t index_;
    };

    /// \brief The Sobol sequence in `Dims` dimensions (Joe and Kuo), in Gray-code order, optionally digitally shifted.
    /// \ingroup core
    /// \tparam Dims The number of dimensions, 1 to 1024.
    template <std::size_t Dims>
    class sobol {
        static_assert(Dims >= 1 && Dims <= snicholls::detail::sobol_max_dimensions,
                      "snicholls::stats::sobol: 1 to 1024 dimensions (scripts/generate_sobol_directions.py makes more)");

    public:
        /// \brief The unshifted sequence: its first point is the origin.
        constexpr sobol() noexcept = default;

        /// \brief A randomly shifted copy: every point XORed with one random 32-bit word per dimension.
        /// \tparam G The generator's type.
        /// \param g The generator; independent shifts give independent, equally balanced copies.
        template <full_range_generator G>
        constexpr explicit sobol(G& g) noexcept(noexcept(g()))
        {
            for (std::uint32_t& s : shift_)
                s = static_cast<std::uint32_t>(random_u64(g) >> 32);
        }

        /// \brief The next point.
        /// \return The point, each coordinate in `[0, 1)`.
        constexpr std::array<double, Dims> next() noexcept
        {
            assert(index_ < (std::uint64_t{1} << 32) && "snicholls::stats::sobol: more than 2^32 points");
            std::array<double, Dims> p{};
            for (std::size_t d = 0; d < Dims; ++d)
                p[d] = static_cast<double>(x_[d] ^ shift_[d]) * 0x1.0p-32;
            const auto c = static_cast<std::size_t>(std::countr_zero(~index_));   // the rightmost zero bit
            for (std::size_t d = 0; d < Dims; ++d)
                x_[d] ^= detail::sobol_v<Dims>[d][c];
            ++index_;
            return p;
        }

        /// \brief Point `i` of the unshifted sequence, without moving it: the XOR of the directions of `i`'s Gray code.
        /// \param i The index, below 2^32.
        /// \return The point.
        [[nodiscard]] static constexpr std::array<double, Dims> point(std::uint64_t i) noexcept
        {
            std::array<double, Dims> p{};
            const std::uint64_t gray = i ^ (i >> 1);
            for (std::size_t d = 0; d < Dims; ++d) {
                std::uint32_t x = 0;
                for (std::uint64_t bits = gray; bits != 0; bits &= bits - 1)
                    x ^= detail::sobol_v<Dims>[d][static_cast<std::size_t>(std::countr_zero(bits))];
                p[d] = static_cast<double>(x) * 0x1.0p-32;
            }
            return p;
        }

        /// \brief Jump to point `i`, keeping the shift.
        /// \param i The index, below 2^32.
        constexpr void seek(std::uint64_t i) noexcept
        {
            const std::uint64_t gray = i ^ (i >> 1);
            for (std::size_t d = 0; d < Dims; ++d) {
                std::uint32_t x = 0;
                for (std::uint64_t bits = gray; bits != 0; bits &= bits - 1)
                    x ^= detail::sobol_v<Dims>[d][static_cast<std::size_t>(std::countr_zero(bits))];
                x_[d] = x;
            }
            index_ = i;
        }

        /// \brief Skip `n` points.
        /// \param n How many.
        constexpr void skip(std::uint64_t n) noexcept { seek(index_ + n); }

        /// \brief The index of the next point.
        /// \return It.
        [[nodiscard]] constexpr std::uint64_t index() const noexcept { return index_; }

    private:
        std::array<std::uint32_t, Dims> x_{};
        std::array<std::uint32_t, Dims> shift_{};
        std::uint64_t index_ = 0;
    };

} // namespace snicholls::stats

#endif /* snicholls_core_stats_low_discrepancy_hpp */
