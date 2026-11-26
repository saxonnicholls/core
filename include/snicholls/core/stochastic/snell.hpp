//
//  snell.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The Snell envelope: the value of stopping optimally on a finite Markov chain, its stopping region, and its Doob-Meyer compensator.
/// \ingroup core
///
/// **Overview.** Holding a right to stop (exercise an option, close a trade) and collect `g(t, s)` in state
/// `s` at time `t`, the best you can do is the Snell envelope: the smallest supermartingale above the
/// payoff. On a finite chain it is backward induction, `U_T = g(T, .)` and
/// `U_t = max(g(t, .), discount * E[U_{t+1} | .])`; stop the first time `U_t = g(t, s_t)`.
///
/// \snippet stochastic/snell_demo.cpp snell
///
/// The envelope's Doob-Meyer decomposition is `U = M - A` with `A` increasing; its increment
/// `U_t - discount * E[U_{t+1}]` is zero exactly where continuing is optimal, and positive where waiting
/// loses value. `compensator_increment` is that increment. `stops` is "the payoff is at least what continuing
/// is worth", so ties stop and `stopping_time` is the smallest optimal stopping time; the first time the
/// compensator rises is the largest.
///
/// Three forms: `snell_envelope<States, Horizon>` takes a `std::array` transition matrix and is `constexpr`
/// throughout, so a stopping table for a small chain is computed by the compiler; the `markov_chain` form
/// takes any size at run time; and `lattice_envelope<Steps>` solves a recombining lattice. On a binomial
/// (or trinomial, or any `Branches`-way) lattice, step `t` has `(Branches - 1) t + 1` nodes and node `j` moves to
/// `j, j + 1, ..., j + Branches - 1` with the same probabilities everywhere: a fixed stencil instead of a
/// `States x States` matrix. It keeps one row of `std::array` storage, O(Steps) state and O(Steps^2) time, and
/// returns the value, the first two steps' values (for delta and gamma off the tree) and the early-exercise
/// boundary: at each step, the lowest and highest node where stopping is strictly better than continuing.
///
/// \snippet stochastic/snell_demo.cpp lattice
///
/// **Relationships.** `stochastic/martingale.hpp`; `math/constexpr_math.hpp` for payoffs at compile time. Continuous
/// models (LSM, HJB solvers) belong in the consuming library. Group: \ref core.
///
/// **Guarantees.** O(Horizon x States^2) arithmetic; the fixed form allocates nothing. `value(t, s)` is at
/// least `g(t, s)` and is a supermartingale along the chain. `lattice_envelope` is O(Steps^2 x Branches)
/// arithmetic in O(Steps) state, `constexpr`, allocation-free and loop-bounded by its template arguments
/// (fit for a GPU kernel or FPGA synthesis); its result is trivially copyable. On the same lattice written as a
/// matrix it agrees with the dense form exactly: the same products, summed in the same order.
///
/// **Errors.** None.
///
/// **Performance.** `benchmarks/stochastic/snell_benchmark.cpp`, a 64-step binomial American put with its
/// boundary: `lattice_envelope` 3.7 us, the same loop by hand 3.5 us, the dense `snell_envelope` over the lattice as a
/// 65 x 65 matrix 205 us (Intel Xeon W-3245, macOS 26, Apple clang 17, Release, 2026-11-26). The dense form is not benchmarked otherwise.
///
/// **Tests and demos.** `tests/stochastic/snell_test.cpp` (Hull's American put; the lattice form against the dense
/// form, binomial and trinomial), `demos/stochastic/snell_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<limits>`, `<span>`, `<type_traits>`, `<vector>`,
/// `stochastic/martingale.hpp`.
///
/// **Known issues.** Discrete time and finite states: discretising a diffusion is the caller's. The lattice
/// form keeps no table of every node, so it has no `stops(t, j)`: its boundary reports the strict stopping
/// region's extremes per step, which describe the region exactly when it is an interval at one end of the
/// step (a monotone payoff such as a put's or a call's). Where the full table is wanted, use the dense form.
///
/// **Status.** New in 0.5.0, experimental.

#ifndef snicholls_core_stochastic_snell_hpp
#define snicholls_core_stochastic_snell_hpp

#include <snicholls/core/stochastic/martingale.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace snicholls::stochastic {

    /// \brief The Snell envelope of a fixed-size chain: the value, and what continuing is worth, at every time and state.
    /// \ingroup core
    /// \tparam States  The number of states.
    /// \tparam Horizon The last time.
    template <std::size_t States, std::size_t Horizon>
    struct fixed_snell {
        std::array<std::array<double, States>, Horizon + 1> value{};          ///< `U_t(s)`.
        std::array<std::array<double, States>, Horizon + 1> continuation{};   ///< `discount * E[U_{t+1} | s]`; minus infinity at the horizon.
        std::array<std::array<double, States>, Horizon + 1> payoff{};         ///< `g(t, s)`.

        /// \brief Whether stopping is optimal at time `t` in state `s`: the payoff is at least what continuing is worth.
        /// \param t The time.
        /// \param s The state.
        /// \return Whether it is; ties stop, so `stopping_time` is the smallest optimal stopping time.
        [[nodiscard]] constexpr bool stops(std::size_t t, std::size_t s) const noexcept { return payoff[t][s] >= continuation[t][s]; }

        /// \brief The envelope's Doob-Meyer compensator increment at `(t, s)`: zero where continuing is optimal.
        /// \param t The time, below the horizon.
        /// \param s The state.
        /// \return `U_t(s) - discount * E[U_{t+1} | s]`, at least zero; positive exactly where stopping is strictly better.
        [[nodiscard]] constexpr double compensator_increment(std::size_t t, std::size_t s) const noexcept { return value[t][s] - continuation[t][s]; }

        /// \brief The optimal stopping time along a path of states: the first `t` where stopping is optimal.
        /// \param path The states from time 0; at least `Horizon + 1` of them, or the time the path ends.
        /// \return The stopping time.
        [[nodiscard]] constexpr std::size_t stopping_time(std::span<const std::size_t> path) const noexcept
        {
            for (std::size_t t = 0; t < path.size() && t <= Horizon; ++t)
                if (stops(t, path[t]))
                    return t;
            return path.size() - 1 < Horizon ? path.size() - 1 : Horizon;
        }
    };

    /// \brief The Snell envelope of a chain given as a `std::array` matrix, at compile time if called in a constant expression.
    /// \ingroup core
    /// \tparam States  The number of states.
    /// \tparam Horizon The last time.
    /// \tparam Payoff  Callable as `g(t, s)` returning the payoff.
    /// \param p        `p[i][j]`, the probability of moving from `i` to `j`.
    /// \param g        The payoff.
    /// \param discount The discount factor per step.
    /// \return The envelope.
    template <std::size_t States, std::size_t Horizon, typename Payoff>
    [[nodiscard]] constexpr fixed_snell<States, Horizon> snell_envelope(const std::array<std::array<double, States>, States>& p, Payoff&& g,
                                                                      double discount = 1.0)
    {
        fixed_snell<States, Horizon> out{};
        for (std::size_t s = 0; s < States; ++s) {
            out.payoff[Horizon][s] = std::invoke(g, Horizon, s);
            out.value[Horizon][s] = out.payoff[Horizon][s];
            out.continuation[Horizon][s] = -std::numeric_limits<double>::infinity();
        }
        for (std::size_t t = Horizon; t-- > 0;)
            for (std::size_t s = 0; s < States; ++s) {
                double expected = 0;
                for (std::size_t j = 0; j < States; ++j)
                    expected += p[s][j] * out.value[t + 1][j];
                out.continuation[t][s] = discount * expected;
                out.payoff[t][s] = std::invoke(g, t, s);
                out.value[t][s] = out.payoff[t][s] > out.continuation[t][s] ? out.payoff[t][s] : out.continuation[t][s];
            }
        return out;
    }

    /// \brief The Snell envelope of a chain of any size, at run time.
    /// \ingroup core
    struct snell {
        std::size_t states = 0;              ///< The number of states.
        std::size_t horizon = 0;             ///< The last time.
        std::vector<double> value;           ///< `U_t(s)` at `t * states + s`.
        std::vector<double> continuation;    ///< `discount * E[U_{t+1} | s]`, minus infinity at the horizon.
        std::vector<double> payoff;          ///< `g(t, s)`.

        /// \brief The value at `(t, s)`.
        /// \param t The time.
        /// \param s The state.
        /// \return `U_t(s)`.
        [[nodiscard]] double at(std::size_t t, std::size_t s) const noexcept { return value[t * states + s]; }

        /// \brief Whether stopping is optimal at `(t, s)`.
        /// \param t The time.
        /// \param s The state.
        /// \return Whether it is.
        [[nodiscard]] bool stops(std::size_t t, std::size_t s) const noexcept
        {
            return payoff[t * states + s] >= continuation[t * states + s];
        }

        /// \brief The optimal stopping time along a path.
        /// \param path The states from time 0.
        /// \return The stopping time.
        [[nodiscard]] std::size_t stopping_time(std::span<const std::size_t> path) const noexcept
        {
            for (std::size_t t = 0; t < path.size() && t <= horizon; ++t)
                if (stops(t, path[t]))
                    return t;
            return path.size() - 1 < horizon ? path.size() - 1 : horizon;
        }
    };

    /// \brief The Snell envelope of a `markov_chain` over `horizon` steps.
    /// \ingroup core
    /// \tparam Payoff Callable as `g(t, s)`.
    /// \param chain    The chain.
    /// \param horizon  The last time.
    /// \param g        The payoff.
    /// \param discount The discount factor per step.
    /// \return The envelope.
    /// \throws std::bad_alloc; whatever the payoff throws.
    template <typename Payoff>
    [[nodiscard]] snell snell_envelope(const markov_chain& chain, std::size_t horizon, Payoff&& g, double discount = 1.0)
    {
        const std::size_t n = chain.size();
        snell out;
        out.states = n;
        out.horizon = horizon;
        out.value.assign((horizon + 1) * n, 0.0);
        out.continuation.assign((horizon + 1) * n, -std::numeric_limits<double>::infinity());
        out.payoff.assign((horizon + 1) * n, 0.0);
        for (std::size_t s = 0; s < n; ++s) {
            out.payoff[horizon * n + s] = std::invoke(g, horizon, s);
            out.value[horizon * n + s] = out.payoff[horizon * n + s];
        }
        for (std::size_t t = horizon; t-- > 0;) {
            const std::span<const double> next(out.value.data() + (t + 1) * n, n);
            for (std::size_t s = 0; s < n; ++s) {
                const double c = discount * chain.expect(s, next);
                const double stop = std::invoke(g, t, s);
                out.continuation[t * n + s] = c;
                out.payoff[t * n + s] = stop;
                out.value[t * n + s] = stop > c ? stop : c;
            }
        }
        return out;
    }

    /// \brief The Snell envelope of a recombining lattice: the value, the first steps' values and the early-exercise boundary.
    /// \ingroup core
    /// \tparam Steps    The number of steps; the horizon.
    /// \tparam Branches The stencil's width: 2 for a binomial lattice, 3 for a trinomial one.
    /// \tparam T        The number type.
    template <std::size_t Steps, std::size_t Branches, typename T = double>
    struct lattice_snell {
        static_assert(Branches >= 2, "snicholls::stochastic::lattice_snell: a lattice branches at least two ways");
        static constexpr std::size_t no_node = static_cast<std::size_t>(-1);   ///< No node stops at this step.
        static constexpr std::size_t head_steps = Steps < 2 ? Steps + 1 : 3;   ///< Steps kept in `head`: 0, 1, 2.

        T value{};   ///< `U_0`, at the root.
        /// \brief `U_t(j)` for the first steps: `head[t][j]`, `j <= (Branches - 1) t`; delta and gamma off the tree.
        std::array<std::array<T, 2 * (Branches - 1) + 1>, head_steps> head{};
        /// \brief Per step below the horizon, the lowest node where stopping strictly beats continuing, or `no_node`.
        std::array<std::size_t, Steps> first_exercise{};
        std::array<std::size_t, Steps> last_exercise{};    ///< The highest such node, or `no_node`.

        /// \brief Whether stopping strictly beats continuing anywhere at step `t`.
        /// \param t The step, below the horizon.
        /// \return Whether it does.
        [[nodiscard]] constexpr bool exercises(std::size_t t) const noexcept { return first_exercise[t] != no_node; }
    };

    /// \brief The Snell envelope of a recombining lattice, in O(Steps) state, at compile time if called in a constant expression.
    /// \ingroup core
    /// \tparam Steps    The number of steps; the horizon.
    /// \tparam T        The number type.
    /// \tparam Branches The stencil's width.
    /// \tparam Payoff   Callable as `g(t, j)` for node `j <= (Branches - 1) t` of step `t`, returning the payoff.
    /// \param stencil  `stencil[k]`, the probability of moving from node `j` to node `j + k` of the next step.
    /// \param g        The payoff; minus infinity (or `lowest()`) before the horizon makes a European claim.
    /// \param discount The discount factor per step.
    /// \return The envelope's value, first steps and boundary.
    template <std::size_t Steps, typename T, std::size_t Branches, typename Payoff>
    [[nodiscard]] constexpr lattice_snell<Steps, Branches, T> lattice_envelope(const std::array<T, Branches>& stencil, Payoff&& g,
                                                                               std::type_identity_t<T> discount = T(1))
    {
        constexpr std::size_t width = Branches - 1;
        lattice_snell<Steps, Branches, T> out{};
        std::array<T, width * Steps + 1> row{};
        for (std::size_t j = 0; j <= width * Steps; ++j)
            row[j] = static_cast<T>(std::invoke(g, Steps, j));
        for (std::size_t t = Steps + 1; t-- > 0;) {
            if (t < Steps) {
                const auto node = [&](std::size_t j) {   // in place: node j reads j .. j + width, none yet overwritten but j
                    T expected = 0;
                    for (std::size_t k = 0; k < Branches; ++k)
                        expected += stencil[k] * row[j + k];
                    const T continuation = discount * expected, stop = static_cast<T>(std::invoke(g, t, j));
                    row[j] = stop > continuation ? stop : continuation;
                    return stop > continuation;   // stopping strictly beats continuing
                };
                std::size_t j = 0, first = out.no_node, last = out.no_node;
                for (; j <= width * t && first == out.no_node; ++j)   // up to the first exercise, then only the last is tracked
                    first = last = node(j) ? j : out.no_node;
                for (; j <= width * t; ++j)
                    if (node(j))
                        last = j;
                out.first_exercise[t] = first;
                out.last_exercise[t] = last;
            }
            if (t < out.head_steps)
                for (std::size_t j = 0; j <= width * t; ++j)
                    out.head[t][j] = row[j];
        }
        out.value = row[0];
        return out;
    }

} // namespace snicholls::stochastic

#endif /* snicholls_core_stochastic_snell_hpp */
