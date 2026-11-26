//
//  option_pricing.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Vanilla options priced by the compiler: Black-Scholes-Merton and its greeks in closed form, a CRR tree for American exercise.
/// \ingroup core
///
/// **Overview.** A `vanilla_option` is a call or a put on an underlying paying a continuous yield `q` (a dividend
/// yield, a foreign rate, a futures' `r`): spot, strike, years to expiry, rate, yield and volatility, all in one
/// floating type. `black_scholes` prices the European option in closed form; `black_scholes_greeks` gives delta,
/// gamma, vega, vanna, volga, theta and rho as a `finance::greeks` (so a book's greeks add, and `explain` takes
/// them). `binomial<Steps>` prices it on a CRR tree solved by `stochastic::lattice_envelope`, European or
/// American, with delta and gamma off the tree and the early-exercise boundary as a spot price per step. Every
/// function is `constexpr` in the option's floating type through `snicholls::math`, so a price, a greek or an
/// exercise boundary is a `static_assert`.
///
/// \snippet finance/option_pricing_demo.cpp black_scholes
///
/// \snippet finance/option_pricing_demo.cpp binomial
///
/// Use these for the quick, exact-to-the-model figure: a showcase, a test oracle, a risk check. Volatility
/// surfaces, discrete dividends, calibration and exotic payoffs belong in the libraries above core.
///
/// **Relationships.** `stochastic/snell.hpp` (`lattice_envelope`), `finance/attribution.hpp` (`greeks`, `explain`),
/// `math/constexpr_math.hpp` (`normal_cdf`, `normal_pdf`, `exp`, `log`, `sqrt`). Group: \ref core.
///
/// **Guarantees.** `constexpr`, no allocation. Black-Scholes is O(1): two logarithm-and-exponential evaluations and
/// two normal distribution functions. `binomial<Steps>` is O(Steps^2) time in O(Steps) `std::array` state, its
/// loops bounded by `Steps`. The greeks are those of the closed form, in the units of the inputs: vega per unit of
/// volatility, rho per unit of rate, theta per year of calendar time (`dV/dt`, the value lost as the option
/// ages, so `-dV/d(expiry)`).
///
/// **Errors.** None; \pre `spot`, `strike`, `vol` and `expiry` are positive. Outside that the arithmetic returns
/// infinities or NaN rather than throwing.
///
/// **Performance.** `benchmarks/finance/option_pricing_benchmark.cpp`, per option over 4,096 random options
/// against the same arithmetic by hand: `black_scholes` 63 ns (hand 72 ns), `black_scholes_greeks` 84 ns (84 ns),
/// `binomial<64>` American with its boundary 6.1 us (5.3 us) (Intel Xeon W-3245, macOS 26, Apple clang 17, Release, 2026-11-26).
///
/// **Tests and demos.** `tests/finance/option_pricing_test.cpp` (Hull's examples, put-call parity, the greeks against
/// central differences, the tree's convergence to the closed form, American against European),
/// `demos/finance/option_pricing_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<limits>`, `finance/attribution.hpp`,
/// `math/constexpr_math.hpp`, `stochastic/snell.hpp`.
///
/// **Known issues.** A CRR tree's error oscillates with the parity of `Steps` and falls as `1/Steps`; average two
/// adjacent step counts for a smoother figure. With `float`, the closed form is good to about six figures and the
/// greeks' cancellation-prone ones (theta, vanna) to fewer; prefer `double` for risk.
///
/// **Status.** New in 0.27.0, experimental.

#ifndef snicholls_core_finance_option_pricing_hpp
#define snicholls_core_finance_option_pricing_hpp

#include <snicholls/core/finance/attribution.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/stochastic/snell.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace snicholls::finance {

    /// \brief A call or a put.
    /// \ingroup core
    enum class option_kind : std::uint8_t { call, put };

    /// \brief When the holder may exercise: at expiry only, or at any time up to it.
    /// \ingroup core
    enum class exercise_style : std::uint8_t { european, american };

    /// \brief A vanilla option and its market: the inputs of Black-Scholes-Merton.
    /// \ingroup core
    /// \tparam T The floating type everything is computed in.
    template <std::floating_point T = double>
    struct vanilla_option {
        option_kind kind = option_kind::call;   ///< Call or put.
        T spot{};     ///< The underlying's price.
        T strike{};   ///< The strike.
        T expiry{};   ///< Years to expiry.
        T rate{};     ///< The continuously compounded risk-free rate.
        T yield{};    ///< The continuous dividend yield (or foreign rate); zero for a non-dividend stock.
        T vol{};      ///< The volatility, per square root of a year.

        /// \brief `+1` for a call, `-1` for a put: the sign that turns one formula into the other.
        /// \return The sign.
        [[nodiscard]] constexpr T sign() const noexcept { return kind == option_kind::call ? T(1) : T(-1); }

        /// \brief What exercising now pays.
        /// \param s The underlying's price.
        /// \return `max(s - K, 0)` for a call, `max(K - s, 0)` for a put.
        [[nodiscard]] constexpr T intrinsic(T s) const noexcept
        {
            const T x = sign() * (s - strike);
            return x > T(0) ? x : T(0);
        }
    };

    namespace detail {
        // d1, d2 and the two discount factors, shared by the price and every greek
        template <std::floating_point T>
        struct bsm_terms {
            T d1, d2, sd, qdf, rdf;
        };
        template <std::floating_point T>
        constexpr bsm_terms<T> bsm(const vanilla_option<T>& o) noexcept
        {
            const T sd = o.vol * math::sqrt(o.expiry);
            const T d1 = (math::log(o.spot / o.strike) + (o.rate - o.yield + o.vol * o.vol / 2) * o.expiry) / sd;
            return {d1, d1 - sd, sd, math::exp(-o.yield * o.expiry), math::exp(-o.rate * o.expiry)};
        }
    } // namespace detail

    /// \brief The Black-Scholes-Merton price of a European option.
    /// \ingroup core
    /// \tparam T The floating type.
    /// \param o The option; \pre positive spot, strike, volatility and expiry.
    /// \return `w (S e^{-qT} N(w d1) - K e^{-rT} N(w d2))`, `w` +1 for a call and -1 for a put.
    template <std::floating_point T>
    [[nodiscard]] constexpr T black_scholes(const vanilla_option<T>& o) noexcept
    {
        const auto b = detail::bsm(o);
        const T w = o.sign();
        return w * (o.spot * b.qdf * math::normal_cdf(w * b.d1) - o.strike * b.rdf * math::normal_cdf(w * b.d2));
    }

    /// \brief The Black-Scholes-Merton greeks of a European option, in closed form.
    /// \ingroup core
    /// \tparam T The floating type.
    /// \param o The option; \pre positive spot, strike, volatility and expiry.
    /// \return delta, gamma, vega, vanna, volga, theta (per year of calendar time) and rho, per unit of the option.
    template <std::floating_point T>
    [[nodiscard]] constexpr greeks<T> black_scholes_greeks(const vanilla_option<T>& o) noexcept
    {
        const auto b = detail::bsm(o);
        const T w = o.sign(), density = math::normal_pdf(b.d1), nd1 = math::normal_cdf(w * b.d1), nd2 = math::normal_cdf(w * b.d2);
        const T vega = o.spot * b.qdf * density * math::sqrt(o.expiry);
        return {.delta = w * b.qdf * nd1,
                .gamma = b.qdf * density / (o.spot * b.sd),
                .vega = vega,
                .vanna = -b.qdf * density * b.d2 / o.vol,
                .volga = vega * b.d1 * b.d2 / o.vol,
                .theta = -o.spot * b.qdf * density * o.vol / (2 * math::sqrt(o.expiry)) - w * o.rate * o.strike * b.rdf * nd2
                         + w * o.yield * o.spot * b.qdf * nd1,
                .rho = w * o.strike * o.expiry * b.rdf * nd2};
    }

    /// \brief An option priced on a binomial tree: the value, delta and gamma off the tree, and the early-exercise boundary.
    /// \ingroup core
    /// \tparam Steps The tree's steps.
    /// \tparam T     The floating type.
    template <std::size_t Steps, std::floating_point T = double>
    struct binomial_price {
        T value{};   ///< The option's value today.
        T delta{};   ///< `(V_u - V_d) / (S_u - S_d)` at the first step.
        T gamma{};   ///< The change in delta across the second step's nodes, over half their spot range; zero with fewer than two steps.
        /// \brief Per step `t < Steps` (at `t * expiry / Steps` years), the critical spot: the highest at which a put is
        /// exercised, the lowest for a call; zero where no node exercises (always for a European option).
        std::array<T, Steps> boundary{};
    };

    /// \brief An option priced on a Cox-Ross-Rubinstein tree, European or American, solved by `stochastic::lattice_envelope`.
    /// \ingroup core
    /// \tparam Steps The tree's steps; the error falls as `1/Steps`.
    /// \tparam T     The floating type.
    /// \param o     The option; \pre positive spot, strike, volatility and expiry, and `Steps >= 1`.
    /// \param style European or American.
    /// \return The value, the tree's delta and gamma, and the early-exercise boundary.
    template <std::size_t Steps, std::floating_point T>
    [[nodiscard]] constexpr binomial_price<Steps, T> binomial(const vanilla_option<T>& o, exercise_style style = exercise_style::american) noexcept
    {
        static_assert(Steps >= 1, "snicholls::finance::binomial: a tree needs at least one step");
        const T dt = o.expiry / T(Steps), jump = o.vol * math::sqrt(dt);
        const T up = math::exp(jump), down = math::exp(-jump), p = (math::exp((o.rate - o.yield) * dt) - down) / (up - down);
        std::array<T, 2 * Steps + 1> level{};   // level[k] = spot * up^(k - Steps); node j of step t is level Steps + 2j - t
        for (std::size_t k = 0; k <= 2 * Steps; ++k)
            level[k] = o.spot * math::exp(jump * (static_cast<T>(k) - static_cast<T>(Steps)));
        const auto spot_at = [&](std::size_t t, std::size_t j) { return level[Steps + 2 * j - t]; };
        const bool american = style == exercise_style::american;
        const T w = o.sign(), strike = o.strike;   // by value: the payoff is the inner loop
        const auto g = [&level, w, strike, american](std::size_t t, std::size_t j) {
            const T x = w * (level[Steps + 2 * j - t] - strike);
            return american || t == Steps ? (x > T(0) ? x : T(0)) : std::numeric_limits<T>::lowest();
        };
        const auto tree = stochastic::lattice_envelope<Steps>(std::array<T, 2>{1 - p, p}, g, math::exp(-o.rate * dt));

        binomial_price<Steps, T> out{};
        out.value = tree.value;
        out.delta = (tree.head[1][1] - tree.head[1][0]) / (spot_at(1, 1) - spot_at(1, 0));
        if constexpr (Steps >= 2) {
            const T upper = (tree.head[2][2] - tree.head[2][1]) / (spot_at(2, 2) - spot_at(2, 1));
            const T lower = (tree.head[2][1] - tree.head[2][0]) / (spot_at(2, 1) - spot_at(2, 0));
            out.gamma = (upper - lower) / ((spot_at(2, 2) - spot_at(2, 0)) / 2);
        }
        for (std::size_t t = 0; t < Steps; ++t)
            out.boundary[t] = !tree.exercises(t) ? T(0)
                              : o.kind == option_kind::put ? spot_at(t, tree.last_exercise[t])
                                                           : spot_at(t, tree.first_exercise[t]);
        return out;
    }

} // namespace snicholls::finance

#endif /* snicholls_core_finance_option_pricing_hpp */
