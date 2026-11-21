//
//  uniform_int.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `uniform_int<T, D>`: a drop-in `std::uniform_int_distribution` whose draws are named, not left to the
/// standard library: core's own, or libstdc++'s or libc++'s reproduced to the bit on every platform.
/// \ingroup core
///
/// **Overview.** The standard specifies `std::mt19937` to the bit but leaves `std::uniform_int_distribution` to
/// each library, so the same seed picks different numbers under libstdc++ (GCC's, and Clang's on Linux), libc++
/// (Apple's, LLVM's) and MSVC. A program that must reproduce another program's choices - llama.cpp's HellaSwag
/// selection, a published benchmark subset, a simulation built on Linux and checked on a Mac - cannot use the
/// distribution its own library happens to ship. `uniform_int<T, D>` takes the algorithm as a template parameter:
///
/// - `draw::core` (the default): `uniform_index`, Lemire's nearly divisionless draw over 64 bits; the fastest,
///   and core's choice for anything new. It needs a `full_range_generator` (32 or 64 full bits).
/// - `draw::libstdcxx`: libstdc++'s algorithm from GCC 11 on (Lemire's draw on 32- or 64-bit generators, two
///   divisions on others, a composition of draws when the range is wider than the generator), as on a 64-bit
///   target with `__int128`.
/// - `draw::libcxx`: libc++'s algorithm (`__independent_bits_engine`: the fewest bits that cover the range,
///   rejected until they fall inside it).
///
/// The choice is made once, at compile time: no flag, no branch on the hot path. When not to use it: for new
/// work, `draw::core` or `uniform_index` directly; the library algorithms are for agreeing with code you do not own.
///
/// \snippet stats/uniform_int_demo.cpp uniform_int
///
/// **Relationships.** `stats/random.hpp` (`uniform_index`, `wyrand`, `full_range_generator`). Group: \ref core.
///
/// **Guarantees.** `constexpr` (with a `constexpr` generator such as `wyrand`), no allocation, `noexcept` exactly
/// when the generator's call is. The same generator state gives the same draws on every platform and compiler.
/// `uniform_int` is two `T`s, trivially copyable, and stateless between draws: `reset()` does nothing, as in both
/// libraries.
///
/// **Errors.** None. `a <= b` is a precondition (asserted), as for the standard distribution.
///
/// **Performance.** Not benchmarked: the library algorithms are there to agree, not to be fast.
///
/// **Tests and demos.** `tests/stats/uniform_int_test.cpp` checks each algorithm draw for draw against the real
/// `std::uniform_int_distribution` of the library it reproduces, where the test is built with that library (macOS
/// for libc++, Ubuntu's GCC and Clang for libstdc++), and against committed sequences everywhere else.
/// `demos/stats/uniform_int_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<random>` (the
/// generator concept), `<type_traits>`, `stats/random.hpp`.
///
/// **Known issues.** libstdc++ before GCC 11 used two divisions everywhere, and on targets without `__int128`
/// still does for 64-bit generators: `draw::libstdcxx` reproduces GCC 11 and later on 64-bit targets (checked
/// against GCC 13 and 15). MSVC's algorithm is not reproduced. `T` is one of the standard's eight
/// `short`..`unsigned long long`; the standard does not allow `char` or `bool` either.
///
/// **Status.** New in 0.18.0.

#ifndef snicholls_core_stats_uniform_int_hpp
#define snicholls_core_stats_uniform_int_hpp

#include <snicholls/core/stats/random.hpp>

#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace snicholls::stats {

    /// \brief Whose algorithm a `uniform_int` draws with.
    /// \ingroup core
    enum class draw {
        core,      ///< `uniform_index`: Lemire's nearly divisionless draw; needs 32 or 64 full bits.
        libstdcxx, ///< libstdc++'s `std::uniform_int_distribution`, GCC 11 and later, 64-bit targets.
        libcxx,    ///< libc++'s `std::uniform_int_distribution`.
    };

    /// \brief The integer types `std::uniform_int_distribution` accepts: `short` to `unsigned long long`.
    /// \ingroup core
    template <typename T>
    concept standard_int_type = std::same_as<T, short> || std::same_as<T, int> || std::same_as<T, long> ||
                                std::same_as<T, long long> || std::same_as<T, unsigned short> ||
                                std::same_as<T, unsigned int> || std::same_as<T, unsigned long> ||
                                std::same_as<T, unsigned long long>;

} // namespace snicholls::stats

namespace snicholls::detail {

    template <stats::draw D>
    struct uniform_int_algorithm;

    template <>
    struct uniform_int_algorithm<stats::draw::core> {
        template <typename T, stats::full_range_generator G>
        static constexpr T run(G& g, T a, T b) noexcept(noexcept(g()))
        {
            using U = std::make_unsigned_t<T>;
            const auto span = static_cast<std::uint64_t>(static_cast<U>(static_cast<U>(b) - static_cast<U>(a)));
            const std::uint64_t r = span == std::numeric_limits<std::uint64_t>::max() ? stats::random_u64(g)
                                                                                      : stats::uniform_index(g, span + 1);
            return static_cast<T>(static_cast<U>(static_cast<U>(a) + r));
        }
    };

    // libstdc++'s uniform_int_distribution::operator() (bits/uniform_int_dist.h, GCC 11 and later), with
    // __int128 available: the same types, the same branches, the same arithmetic.
    template <>
    struct uniform_int_algorithm<stats::draw::libstdcxx> {
        template <typename W, typename Up, typename G>
        static constexpr Up nearly_divisionless(G& g, Up range) noexcept(noexcept(g()))
        {
            W product = static_cast<W>(static_cast<W>(g()) * static_cast<W>(range));
            auto low = static_cast<Up>(product);
            if (low < range) {
                const auto threshold = static_cast<Up>(static_cast<Up>(0 - range) % range);
                while (low < threshold) {
                    product = static_cast<W>(static_cast<W>(g()) * static_cast<W>(range));
                    low = static_cast<Up>(product);
                }
            }
            return static_cast<Up>(product >> std::numeric_limits<Up>::digits);
        }

        // The 64-bit case multiplies into 128 bits: core's umul128 is __int128's product, high and low.
        template <typename G>
        static constexpr std::uint64_t nearly_divisionless64(G& g, std::uint64_t range) noexcept(noexcept(g()))
        {
            auto product = snicholls::detail::umul128(static_cast<std::uint64_t>(g()), range);
            if (product.low < range) {
                const std::uint64_t threshold = (0 - range) % range;
                while (product.low < threshold)
                    product = snicholls::detail::umul128(static_cast<std::uint64_t>(g()), range);
            }
            return product.high;
        }

        template <typename T, typename G>
        static constexpr T run(G& g, T a, T b) noexcept(noexcept(g()))
        {
            using U = std::common_type_t<typename G::result_type, std::make_unsigned_t<T>>;
            constexpr U gmin = static_cast<U>(G::min());
            constexpr U grange = static_cast<U>(static_cast<U>(G::max()) - gmin);
            const auto range = static_cast<U>(static_cast<U>(b) - static_cast<U>(a));
            U ret{};
            if (grange > range) {
                const auto erange = static_cast<U>(range + 1);
                if constexpr (grange == std::numeric_limits<std::uint64_t>::max())
                    ret = static_cast<U>(nearly_divisionless64(g, static_cast<std::uint64_t>(erange)));
                else if constexpr (grange == std::numeric_limits<std::uint32_t>::max())
                    ret = static_cast<U>(nearly_divisionless<std::uint64_t>(g, static_cast<std::uint32_t>(erange)));
                else {
                    const auto scaling = static_cast<U>(grange / erange);
                    const auto past = static_cast<U>(erange * scaling);
                    do
                        ret = static_cast<U>(static_cast<U>(g()) - gmin);
                    while (ret >= past);
                    ret = static_cast<U>(ret / scaling);
                }
            }
            else if (grange < range) {
                // Every value in [0, range] is (grange + 1) * high + low, one way: draw high, then low. (Only a
                // generator narrower than U gets here; the guard keeps grange + 1 from being 0 in the others.)
                if constexpr (grange < std::numeric_limits<U>::max()) {
                    constexpr auto egrange = static_cast<U>(grange + 1);
                    U tmp{};
                    do {
                        tmp = static_cast<U>(egrange * static_cast<U>(run<T>(g, T{0}, static_cast<T>(range / egrange))));
                        ret = static_cast<U>(tmp + static_cast<U>(static_cast<U>(g()) - gmin));
                    } while (ret > range || ret < tmp);
                }
            }
            else
                ret = static_cast<U>(static_cast<U>(g()) - gmin);
            return static_cast<T>(static_cast<U>(ret + static_cast<U>(a)));
        }
    };

    // libc++'s uniform_int_distribution::operator() and __independent_bits_engine (__random/), member for member.
    template <>
    struct uniform_int_algorithm<stats::draw::libcxx> {
        template <typename E, typename Ut>
        struct bits_engine {
            using engine_type = typename E::result_type;
            using working = std::conditional_t<sizeof(engine_type) <= sizeof(Ut), Ut, engine_type>;
            static constexpr working rp = static_cast<working>(static_cast<working>(E::max() - E::min()) + working{1});
            static constexpr std::size_t wdt = std::numeric_limits<working>::digits;
            static constexpr std::size_t edt = std::numeric_limits<engine_type>::digits;
            // __log2<working, rp>: floor(log2 rp), or every bit of `working` when rp wrapped to 0.
            static constexpr std::size_t m = rp == 0 ? wdt : static_cast<std::size_t>(std::bit_width(rp) - 1);

            E& e;
            std::size_t w, w0 = 0, n = 0, n0 = 0;
            working y0 = 0, y1 = 0;
            engine_type mask0 = 0, mask1 = 0;

            constexpr bits_engine(E& engine, std::size_t bits) noexcept : e(engine), w(bits)
            {
                n = w / m + (w % m != 0);
                w0 = w / n;
                if constexpr (rp == 0)
                    y0 = rp;
                else if (w0 < wdt)
                    y0 = static_cast<working>((rp >> w0) << w0);
                else
                    y0 = 0;
                if (static_cast<working>(rp - y0) > static_cast<working>(y0 / n)) {
                    ++n;
                    w0 = w / n;
                    y0 = w0 < wdt ? static_cast<working>((rp >> w0) << w0) : working{0};
                }
                n0 = n - w % n;
                y1 = w0 < wdt - 1 ? static_cast<working>((rp >> (w0 + 1)) << (w0 + 1)) : working{0};
                mask0 = w0 > 0 ? static_cast<engine_type>(static_cast<engine_type>(~engine_type{0}) >> (edt - w0)) : engine_type{0};
                mask1 = w0 < edt - 1 ? static_cast<engine_type>(static_cast<engine_type>(~engine_type{0}) >> (edt - (w0 + 1)))
                                     : static_cast<engine_type>(~engine_type{0});
            }

            constexpr Ut operator()() noexcept(noexcept(e()))
            {
                if constexpr (rp == 0)
                    return static_cast<Ut>(e() & mask0);
                else {
                    constexpr std::size_t wrt = std::numeric_limits<Ut>::digits;
                    Ut sp = 0;
                    for (std::size_t k = 0; k < n0; ++k) {
                        engine_type u{};
                        do
                            u = static_cast<engine_type>(e() - E::min());
                        while (u >= y0);
                        sp = w0 < wrt ? static_cast<Ut>(sp << w0) : Ut{0};
                        sp = static_cast<Ut>(sp + (u & mask0));
                    }
                    for (std::size_t k = n0; k < n; ++k) {
                        engine_type u{};
                        do
                            u = static_cast<engine_type>(e() - E::min());
                        while (u >= y1);
                        sp = w0 < wrt - 1 ? static_cast<Ut>(sp << (w0 + 1)) : Ut{0};
                        sp = static_cast<Ut>(sp + (u & mask1));
                    }
                    return sp;
                }
            }
        };

        template <typename T, typename G>
        static constexpr T run(G& g, T a, T b) noexcept(noexcept(g()))
        {
            using Ut = std::conditional_t<sizeof(T) <= sizeof(std::uint32_t), std::uint32_t, std::make_unsigned_t<T>>;
            const auto rp = static_cast<Ut>(static_cast<Ut>(static_cast<Ut>(b) - static_cast<Ut>(a)) + Ut{1});
            if (rp == 1)
                return a;
            constexpr std::size_t dt = std::numeric_limits<Ut>::digits;
            if (rp == 0)
                return static_cast<T>(bits_engine<G, Ut>(g, dt)());
            auto w = static_cast<std::size_t>(dt - static_cast<std::size_t>(std::countl_zero(rp)) - 1);
            if ((rp & (std::numeric_limits<Ut>::max() >> (dt - w))) != 0)
                ++w;
            bits_engine<G, Ut> e(g, w);
            Ut u{};
            do
                u = e();
            while (u >= rp);
            return static_cast<T>(static_cast<Ut>(u + static_cast<Ut>(a)));
        }
    };

} // namespace snicholls::detail

namespace snicholls::stats {

    /// \brief A uniform integer in `[a, b]`, drawn by the algorithm `D` names: a drop-in for
    /// `std::uniform_int_distribution<T>` whose output does not depend on the standard library.
    /// \ingroup core
    /// \tparam T The integer type, one of `standard_int_type`.
    /// \tparam D Whose algorithm: `draw::core` (default), `draw::libstdcxx` or `draw::libcxx`.
    template <standard_int_type T = int, draw D = draw::core>
    class uniform_int {
    public:
        /// \brief The type of a draw.
        using result_type = T;

        /// \brief The distribution over `[a, b]`.
        /// \param a The smallest value, by default 0.
        /// \param b The largest value, by default `T`'s largest; `a <= b`.
        constexpr explicit uniform_int(T a = 0, T b = std::numeric_limits<T>::max()) noexcept : a_(a), b_(b) { assert(a <= b); }

        /// \brief The next draw.
        /// \tparam G The generator's type: any `std::uniform_random_bit_generator`, or a `full_range_generator`
        /// for `draw::core`.
        /// \param g The generator.
        /// \return A value in `[a(), b()]`.
        template <std::uniform_random_bit_generator G>
            requires(D != draw::core || full_range_generator<G>)
        constexpr T operator()(G& g) const noexcept(noexcept(g()))
        {
            return snicholls::detail::uniform_int_algorithm<D>::template run<T>(g, a_, b_);
        }

        /// \brief The smallest value.
        /// \return `a`.
        [[nodiscard]] constexpr T a() const noexcept { return a_; }

        /// \brief The largest value.
        /// \return `b`.
        [[nodiscard]] constexpr T b() const noexcept { return b_; }

        /// \brief The smallest value, as the standard's `min()`.
        /// \return `a`.
        [[nodiscard]] constexpr T min() const noexcept { return a_; }

        /// \brief The largest value, as the standard's `max()`.
        /// \return `b`.
        [[nodiscard]] constexpr T max() const noexcept { return b_; }

        /// \brief Nothing: the draws keep no state, in core and in both libraries.
        constexpr void reset() noexcept {}

        /// \brief Two distributions are equal when their bounds are.
        /// \return Whether `a` and `b` are equal.
        friend constexpr bool operator==(const uniform_int&, const uniform_int&) noexcept = default;

    private:
        T a_;
        T b_;
    };

} // namespace snicholls::stats

#endif /* snicholls_core_stats_uniform_int_hpp */
