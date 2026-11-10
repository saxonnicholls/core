//
//  random.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `wyrand`, a fast 64-bit generator, and unbiased draws from any full-range generator.
/// \ingroup core
///
/// **Overview.** Resampling spends its time drawing indices. `std::mt19937_64` with
/// `std::uniform_int_distribution` costs several nanoseconds per index, and the distribution's output
/// is not specified by the standard, so the same seed gives different resamples on libstdc++, libc++
/// and MSVC. `wyrand` is the generator of wyhash (Wang Yi): one add and one 64 x 64 -> 128 multiply
/// per 64 bits, passing BigCrush and PractRand. `uniform_index` is Lemire's nearly divisionless
/// unbiased draw; `uniform_index32` draws from 32 bits, so one 64-bit call gives two indices.
///
/// \snippet stats/random_demo.cpp random
///
/// Any `std::uniform_random_bit_generator` of 32 or 64 full bits works (`std::mt19937`,
/// `std::mt19937_64`, `wyrand`); the draws here are specified to the bit, so the same generator state
/// gives the same draws everywhere.
///
/// **Relationships.** `stats/bootstrap.hpp`, `stats/low_discrepancy.hpp`, `game/shapley.hpp`'s
/// samplers; `hash/wyhash.hpp` for the mixer. Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept`, no allocation. `wyrand` is 8 bytes and trivially copyable;
/// `stream(seed, k)` gives independent-looking generators for parallel work. Not cryptographic.
///
/// **Errors.** None: `uniform_index` of 0 is a precondition violation (asserted).
///
/// **Performance.** See `stats/bootstrap.hpp` and `BENCHMARKS.md`.
///
/// **Tests and demos.** `tests/stats/random_test.cpp`, `demos/stats/random_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstdint>`, `<limits>`, `<random>` (for the concept only),
/// `<span>`, `<type_traits>`, `<utility>`, `detail/wide_mul.hpp`, `hash/wyhash.hpp`.
///
/// **Known issues.** wyrand's period is 2^64, ample for resampling and Monte Carlo, not for
/// cryptography.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_stats_random_hpp
#define snicholls_core_stats_random_hpp

#include <snicholls/core/detail/wide_mul.hpp>
#include <snicholls/core/hash/wyhash.hpp>

#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls::stats {

    /// \brief wyhash's generator: 64 random bits per call from one add and one wide multiply.
    /// \ingroup core
    class wyrand {
    public:
        /// \brief The type of a draw.
        using result_type = std::uint64_t;

        /// \brief A generator from a seed; every seed, zero included, is good.
        /// \param seed The seed.
        constexpr explicit wyrand(std::uint64_t seed = 0) noexcept : state_(seed) {}

        /// \brief The `k`th of a family of generators from one seed, for parallel streams.
        /// \param seed The family's seed.
        /// \param k    The stream.
        /// \return A generator whose state is `seed` and `k` mixed by splitmix64.
        [[nodiscard]] static constexpr wyrand stream(std::uint64_t seed, std::uint64_t k) noexcept
        {
            return wyrand(splitmix64(seed ^ splitmix64(k + 0x9e3779b97f4a7c15ull)));
        }

        /// \brief The smallest draw.
        /// \return 0.
        [[nodiscard]] static constexpr result_type min() noexcept { return 0; }

        /// \brief The largest draw.
        /// \return 2^64 - 1.
        [[nodiscard]] static constexpr result_type max() noexcept { return std::numeric_limits<result_type>::max(); }

        /// \brief The next 64 random bits.
        /// \return The draw.
        constexpr result_type operator()() noexcept
        {
            state_ += 0x2d358dccaa6c78a5ull;
            return wymix(state_, state_ ^ 0x8bb84b93962eacc9ull);
        }

        /// \brief Two generators are equal when they will draw the same.
        /// \return Whether the states are equal.
        friend constexpr bool operator==(const wyrand&, const wyrand&) noexcept = default;

    private:
        std::uint64_t state_;
    };

    /// \brief A generator of 32 or 64 full bits: `[0, 2^32)` or `[0, 2^64)`.
    /// \ingroup core
    template <typename G>
    concept full_range_generator =
        std::uniform_random_bit_generator<std::remove_cvref_t<G>> && std::remove_cvref_t<G>::min() == 0 &&
        (std::remove_cvref_t<G>::max() == std::numeric_limits<std::uint64_t>::max() ||
         std::remove_cvref_t<G>::max() == std::numeric_limits<std::uint32_t>::max());

    /// \brief 64 random bits from a 64-bit generator: one call.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \param g The generator.
    /// \return The bits.
    template <full_range_generator G>
        requires(G::max() == std::numeric_limits<std::uint64_t>::max())
    [[nodiscard]] constexpr std::uint64_t random_u64(G& g) noexcept(noexcept(g()))
    {
        return static_cast<std::uint64_t>(g());
    }

    /// \brief 64 random bits from a 32-bit generator: two calls, the first the high half.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \param g The generator.
    /// \return The bits.
    template <full_range_generator G>
        requires(G::max() == std::numeric_limits<std::uint32_t>::max())
    [[nodiscard]] constexpr std::uint64_t random_u64(G& g) noexcept(noexcept(g()))
    {
        const auto high = static_cast<std::uint64_t>(g());
        return (high << 32) | static_cast<std::uint64_t>(g());
    }

    /// \brief An unbiased index in `[0, n)` (Lemire's nearly divisionless method): one draw almost always.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \param g The generator.
    /// \param n The bound, above zero.
    /// \return The index.
    template <full_range_generator G>
    [[nodiscard]] constexpr std::uint64_t uniform_index(G& g, std::uint64_t n) noexcept(noexcept(g()))
    {
        assert(n > 0);
        snicholls::detail::wide_product m = snicholls::detail::umul128(random_u64(g), n);
        if (m.low < n) {
            const std::uint64_t threshold = (std::uint64_t{0} - n) % n;
            while (m.low < threshold)
                m = snicholls::detail::umul128(random_u64(g), n);
        }
        return m.high;
    }

    /// \brief An unbiased index in `[0, n)` from 32 random bits, so that one 64-bit draw makes two.
    ///
    /// `bits` is consumed; when the draw must be rejected (probability below n / 2^32) `refill` is called for
    /// 32 fresh bits.
    /// \ingroup core
    /// \tparam Refill Callable returning 32 fresh random bits.
    /// \param bits   32 random bits.
    /// \param n      The bound, above zero and below 2^32.
    /// \param refill Where fresh bits come from on a rejection.
    /// \return The index.
    template <typename Refill>
    [[nodiscard]] constexpr std::uint32_t uniform_index32(std::uint32_t bits, std::uint32_t n, Refill&& refill) noexcept(noexcept(refill()))
    {
        assert(n > 0);
        std::uint64_t m = static_cast<std::uint64_t>(bits) * n;
        auto low = static_cast<std::uint32_t>(m);
        if (low < n) {
            const std::uint32_t threshold = (std::uint32_t{0} - n) % n;
            while (low < threshold) {
                m = static_cast<std::uint64_t>(static_cast<std::uint32_t>(refill())) * n;
                low = static_cast<std::uint32_t>(m);
            }
        }
        return static_cast<std::uint32_t>(m >> 32);
    }

    /// \brief A uniform double in `[0, 1)` with 53 random bits.
    /// \ingroup core
    /// \tparam G The generator's type.
    /// \param g The generator.
    /// \return The draw.
    template <full_range_generator G>
    [[nodiscard]] constexpr double uniform_real(G& g) noexcept(noexcept(g()))
    {
        return static_cast<double>(random_u64(g) >> 11) * 0x1.0p-53;
    }

    /// \brief Shuffle in place, uniformly (Fisher-Yates with `uniform_index`): the same order everywhere for the same state.
    /// \ingroup core
    /// \tparam T The element type.
    /// \tparam G The generator's type.
    /// \param values The values.
    /// \param g      The generator.
    template <typename T, full_range_generator G>
    constexpr void shuffle(std::span<T> values, G& g) noexcept(noexcept(g()) && std::is_nothrow_swappable_v<T>)
    {
        for (std::size_t i = values.size(); i > 1; --i) {
            using std::swap;
            swap(values[i - 1], values[static_cast<std::size_t>(uniform_index(g, i))]);
        }
    }

} // namespace snicholls::stats

#endif /* snicholls_core_stats_random_hpp */
