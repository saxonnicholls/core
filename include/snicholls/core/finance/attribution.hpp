//
//  attribution.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief P&L explained by greeks and factors, and risk by contributor: Taylor terms, shared fairly by Shapley.
/// \ingroup core
///
/// **Overview.** A position's P&L over a move of its risk factors is, to second order,
/// `sum g_i dx_i + 1/2 sum H_ij dx_i dx_j + carry`. Each term is a greek's P&L: delta, gamma, vega,
/// volga, theta, rho, and the cross terms such as vanna (spot and volatility together). `explain`
/// reports every term, the part the model leaves unexplained against the actual P&L, and the P&L by
/// factor: each factor's own terms plus half of every cross term it is in. That split is the Shapley
/// attribution of the Taylor model, exactly: its pair dividends are the cross terms, and Shapley shares
/// each equally between the pair. So "how much was spot, how much was volatility" has the one fair
/// answer, without the 2^n revaluations of `game::attribute_change`.
///
/// \snippet finance/attribution_demo.cpp greeks
///
/// For one underlying, `greeks` and `market_move` name the terms; for a book of several underlyings with
/// cross-gammas, `taylor_model<K>` is the general form. Greeks add across positions (`+`, scaled by `*`),
/// so a book's explain is its positions' summed. "Long or short volatility" is the sign of vega, and the
/// volatility P&L is `vega + volga + vanna / 2`.
///
/// Risk is explained the same way. `risk_contributions` splits a portfolio's variance into each
/// holding's `w_i (Sigma w)_i`, which is both the Euler allocation and the Shapley attribution of the
/// variance game; the volatility split it reports, the same shares divided by sigma, is Euler's, and is
/// not the Shapley attribution of volatility, which `game::shapley` computes when wanted.
///
/// **Relationships.** `game/shapley.hpp`, `game/harsanyi.hpp`, `game/change.hpp` (exact by
/// revaluation). Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept`, no allocation. `explained` is the sum of the terms; the
/// factors' P&L sums to `explained - carry`.
///
/// **Errors.** None.
///
/// **Performance.** O(K^2); not benchmarked.
///
/// **Tests and demos.** `tests/finance/attribution_test.cpp`, `demos/finance/attribution_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cmath>`, `<concepts>`, `<cstddef>`.
///
/// **Known issues.** The greeks are in the units of the moves: a vega per unit of volatility with the
/// move in units of volatility, a theta per year with the move in years. Third-order terms (speed,
/// ultima, zomma) are not modelled and land in `unexplained`.
///
/// **Status.** New in 0.1.0, experimental.

#ifndef snicholls_core_finance_attribution_hpp
#define snicholls_core_finance_attribution_hpp

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace snicholls::finance {

    /// \brief A second-order model of P&L in `K` factors: gradient, Hessian, and a carry that no factor causes.
    /// \ingroup core
    /// \tparam K The number of factors.
    /// \tparam T The number type.
    template <std::size_t K, std::floating_point T = double>
    struct taylor_model {
        std::array<T, K> gradient{};                ///< First derivatives: delta, vega, ...
        std::array<std::array<T, K>, K> hessian{};  ///< Second derivatives, symmetric: gamma, volga, vanna, cross-gamma, ...
        T carry{};                                  ///< P&L with no factor moving (theta times the time, financing).
    };

    /// \brief A Taylor model's P&L explained term by term and factor by factor.
    /// \ingroup core
    /// \tparam K The number of factors.
    /// \tparam T The number type.
    template <std::size_t K, std::floating_point T = double>
    struct taylor_explain {
        std::array<T, K> first_order{};                 ///< `g_i dx_i`.
        std::array<std::array<T, K>, K> second_order{}; ///< `1/2 H_ij dx_i dx_j`, both halves of a cross term: they sum to `1/2 dx' H dx`.
        T carry{};                                      ///< The model's carry.
        T explained{};                                  ///< Every term.
        T actual{};                                     ///< The P&L observed.
        T unexplained{};                                ///< `actual - explained`.
        std::array<T, K> by_factor{};                   ///< Shapley: a factor's own terms and half of each cross term it is in.
    };

    /// \brief Explain a P&L by a Taylor model.
    /// \ingroup core
    /// \tparam K The number of factors.
    /// \tparam T The number type.
    /// \param model  The model.
    /// \param moves  The factors' moves.
    /// \param actual The P&L observed, by full revaluation or from the books.
    /// \return The explain.
    template <std::size_t K, std::floating_point T>
    [[nodiscard]] constexpr taylor_explain<K, T> explain(const taylor_model<K, T>& model, const std::array<T, K>& moves, T actual) noexcept
    {
        taylor_explain<K, T> out{};
        out.carry = model.carry;
        out.explained = model.carry;
        out.actual = actual;
        for (std::size_t i = 0; i < K; ++i) {
            out.first_order[i] = model.gradient[i] * moves[i];
            out.by_factor[i] = out.first_order[i];
            out.explained += out.first_order[i];
            for (std::size_t j = 0; j < K; ++j) {
                out.second_order[i][j] = T(0.5) * model.hessian[i][j] * moves[i] * moves[j];
                out.by_factor[i] += out.second_order[i][j];
                out.explained += out.second_order[i][j];
            }
        }
        out.unexplained = actual - out.explained;
        return out;
    }

    /// \brief The greeks of a position in one underlying, in the units of the moves.
    /// \ingroup core
    /// \tparam T The number type.
    template <std::floating_point T = double>
    struct greeks {
        T delta{};   ///< dV/dS.
        T gamma{};   ///< d2V/dS2.
        T vega{};    ///< dV/dsigma.
        T vanna{};   ///< d2V/dS dsigma.
        T volga{};   ///< d2V/dsigma2 (vomma).
        T theta{};   ///< dV/dt, per unit of the time move.
        T rho{};     ///< dV/dr.

        /// \brief Whether the position gains when volatility rises.
        /// \return `vega > 0`.
        [[nodiscard]] constexpr bool long_volatility() const noexcept { return vega > T(0); }

        /// \brief A book's greeks: two positions', added.
        /// \param a A position.
        /// \param b Another, in the same underlying.
        /// \return Their sum.
        friend constexpr greeks operator+(const greeks& a, const greeks& b) noexcept
        {
            return {a.delta + b.delta, a.gamma + b.gamma, a.vega + b.vega, a.vanna + b.vanna, a.volga + b.volga, a.theta + b.theta, a.rho + b.rho};
        }

        /// \brief A position's greeks scaled by a quantity.
        /// \param g        The greeks of one unit.
        /// \param quantity The units held; negative when short.
        /// \return The scaled greeks.
        friend constexpr greeks operator*(const greeks& g, T quantity) noexcept
        {
            return {g.delta * quantity, g.gamma * quantity, g.vega * quantity, g.vanna * quantity, g.volga * quantity, g.theta * quantity,
                    g.rho * quantity};
        }
    };

    /// \brief How the market moved: the four factors of one underlying.
    /// \ingroup core
    /// \tparam T The number type.
    template <std::floating_point T = double>
    struct market_move {
        T spot{};   ///< The change in the underlying's price.
        T vol{};    ///< The change in implied volatility.
        T time{};   ///< The time elapsed, in theta's units.
        T rate{};   ///< The change in the rate.
    };

    /// \brief A P&L explained by the greeks, and by the factors that moved.
    /// \ingroup core
    /// \tparam T The number type.
    template <std::floating_point T = double>
    struct greeks_explain {
        T delta{};        ///< `delta dS`.
        T gamma{};        ///< `1/2 gamma dS^2`.
        T vega{};         ///< `vega dsigma`.
        T vanna{};        ///< `vanna dS dsigma`, the whole cross term.
        T volga{};        ///< `1/2 volga dsigma^2`.
        T theta{};        ///< `theta dt`.
        T rho{};          ///< `rho dr`.
        T explained{};    ///< Every term.
        T actual{};       ///< The P&L observed.
        T unexplained{};  ///< `actual - explained`: third-order and model error.
        T spot{};         ///< By factor (Shapley): `delta + gamma + vanna / 2`.
        T volatility{};   ///< By factor (Shapley): `vega + volga + vanna / 2`; positive when the volatility view paid.
        T time{};         ///< By factor: `theta`.
        T rates{};        ///< By factor: `rho`.
    };

    /// \brief The Taylor model of one underlying's greeks, factors in the order spot, volatility, time, rate.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param g The greeks.
    /// \return The model.
    template <std::floating_point T>
    [[nodiscard]] constexpr taylor_model<4, T> as_taylor_model(const greeks<T>& g) noexcept
    {
        taylor_model<4, T> model{};
        model.gradient = {g.delta, g.vega, g.theta, g.rho};
        model.hessian[0][0] = g.gamma;
        model.hessian[1][1] = g.volga;
        model.hessian[0][1] = g.vanna;
        model.hessian[1][0] = g.vanna;
        return model;
    }

    /// \brief Explain a P&L by the greeks: every term, the residual, and the P&L by factor.
    /// \ingroup core
    /// \tparam T The number type.
    /// \param g      The position's (or book's) greeks.
    /// \param move   How the market moved.
    /// \param actual The P&L observed.
    /// \return The explain.
    template <std::floating_point T>
    [[nodiscard]] constexpr greeks_explain<T> explain(const greeks<T>& g, const market_move<T>& move, T actual) noexcept
    {
        const taylor_explain<4, T> terms = explain(as_taylor_model(g), std::array<T, 4>{move.spot, move.vol, move.time, move.rate}, actual);
        greeks_explain<T> out{};
        out.delta = terms.first_order[0];
        out.vega = terms.first_order[1];
        out.theta = terms.first_order[2];
        out.rho = terms.first_order[3];
        out.gamma = terms.second_order[0][0];
        out.volga = terms.second_order[1][1];
        out.vanna = terms.second_order[0][1] + terms.second_order[1][0];
        out.explained = terms.explained;
        out.actual = actual;
        out.unexplained = terms.unexplained;
        out.spot = terms.by_factor[0];
        out.volatility = terms.by_factor[1];
        out.time = terms.by_factor[2];
        out.rates = terms.by_factor[3];
        return out;
    }

    /// \brief A portfolio's variance and volatility, and each holding's contribution to them.
    /// \ingroup core
    /// \tparam K The number of holdings.
    /// \tparam T The number type.
    template <std::size_t K, std::floating_point T = double>
    struct risk_contribution {
        T variance{};                       ///< `w' Sigma w`.
        T volatility{};                     ///< Its square root.
        std::array<T, K> to_variance{};     ///< `w_i (Sigma w)_i`: Euler and Shapley at once; they sum to the variance.
        std::array<T, K> to_volatility{};   ///< `to_variance / volatility`: Euler; they sum to the volatility.
    };

    /// \brief Each holding's contribution to a portfolio's variance and volatility.
    /// \ingroup core
    /// \tparam K The number of holdings.
    /// \tparam T The number type.
    /// \param covariance The holdings' covariance matrix, symmetric.
    /// \param weights    The holdings: weights or exposures, in the covariance's units.
    /// \return The contributions; `to_volatility` is zero when the volatility is.
    template <std::size_t K, std::floating_point T>
    [[nodiscard]] constexpr risk_contribution<K, T> risk_contributions(const std::array<std::array<T, K>, K>& covariance,
                                                                       const std::array<T, K>& weights) noexcept
    {
        risk_contribution<K, T> out{};
        for (std::size_t i = 0; i < K; ++i) {
            T marginal{};
            for (std::size_t j = 0; j < K; ++j)
                marginal += covariance[i][j] * weights[j];
            out.to_variance[i] = weights[i] * marginal;
            out.variance += out.to_variance[i];
        }
        out.volatility = out.variance > T(0) ? std::sqrt(out.variance) : T(0);
        for (std::size_t i = 0; i < K; ++i)
            out.to_volatility[i] = out.volatility > T(0) ? out.to_variance[i] / out.volatility : T(0);
        return out;
    }

} // namespace snicholls::finance

#endif /* snicholls_core_finance_attribution_hpp */
