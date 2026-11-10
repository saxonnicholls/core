//
//  shapley.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Shapley attribution of a game, exactly or by sampling, and the attribution methods it is compared with.
/// \ingroup core
///
/// **Overview.** An investment process has `n` features (signals, constraints, risk factors), and a
/// performance measure `v(S)` for every set `S` of them switched on, from a backtest, a pricer or a model:
/// a game. Attribution splits `v(all) - v(none)` among the features. The Shapley attribution is the only
/// split with full attribution, the correct baseline, fairness and monotonicity (Young 1985); it is the
/// average, over all `n!` orders of switching features on, of each feature's lift when it is switched on.
///
/// \snippet game/shapley_demo.cpp shapley
///
/// | method                     | function                         | evaluations   | full | fair |
/// | -------------------------- | -------------------------------- | ------------- | ---- | ---- |
/// | Shapley, exact             | `shapley`                        | 2^n           | yes  | yes  |
/// | Shapley, by definition     | `shapley_by_permutations`        | 2^n, n! paths | yes  | yes  |
/// | permuted sequential        | `permuted_sequential`, `sequential` | n + 1      | yes  | no   |
/// | one at a time (Brinson)    | `one_at_a_time`                  | n + 2         | no   | yes  |
/// | leave one out              | `leave_one_out`                  | n + 2         | no   | yes  |
/// | Banzhaf                    | `banzhaf`                        | 2^n           | no   | yes  |
/// | Shapley, sampled orders    | `shapley_sampled`                | chosen        | yes  | in expectation |
/// | Shapley, sampled lifts     | `shapley_sampled_lifts`          | chosen        | if scaled | in expectation |
///
/// `shapley_interactions` gives the Shapley interaction index of every pair: which features act together.
/// For a game that is cheap to tabulate, `harsanyi.hpp` gives the same values from the dividends; for
/// features in groups, `owen.hpp`.
///
/// **Relationships.** `game/coalition.hpp` for games; `game/harsanyi.hpp`, `game/owen.hpp`,
/// `game/properties.hpp`, `game/tabular.hpp` (models over a dataset), `game/change.hpp` (explaining a
/// change of inputs). Group: \ref core.
///
/// **Guarantees.** The exact methods are `constexpr`, evaluate each coalition once, and take
/// O(n 2^n) arithmetic after that. A value type need only be a `game_value`, so several metrics can be
/// attributed together. The sampling methods take a floating-point value, report a standard error per
/// feature, and draw from the caller's generator: the same generator state gives the same attribution
/// on every platform, which `std::shuffle` does not promise.
///
/// **Errors.** None of their own: limits on `n` are preconditions (`static_assert` where `n` is a
/// template argument, `assert` otherwise). Whatever the game throws propagates.
///
/// **Performance.** `benchmarks/game/shapley_benchmark.cpp`, exact Shapley of a 16-player game
/// from its table against formula (3) of the paper written out, after a fairness check. Release, Apple
/// clang 17, Xeon W-3245, 2026-11-07: 481 us through `shapley` against
/// 1,405 us for the formula written out, agreeing to 1e-12 of the total. A game is a table lookup here;
/// with a real backtest the 2^n evaluations dominate, and `memoized` or sampling is the lever.
///
/// **Tests and demos.** `tests/game/shapley_test.cpp`, with the paper's tables 1, 3 and 4 and its
/// quadratic example; `demos/game/shapley_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cassert>`, `<cmath>`, `<concepts>`, `<cstddef>`, `<cstdint>`,
/// `<limits>`, `<ostream>`, `<random>` (for the generator concept only), `<span>`, `<vector>`,
/// `meta/concepts.hpp`, `stats/random.hpp`, `game/coalition.hpp`, `combinatorics/permutations.hpp`.
///
/// **Known issues.** The paper writes its formulas (3) and (4) with "- b" at the end and gives `2P1`
/// as the attribution of `x'Px`; neither agrees with its own tables or with full attribution, and this
/// header follows the tables: no `- b`, and `P1`. The tests pin both.
///
/// **Status.** New in 0.1.0, experimental.

#ifndef snicholls_core_game_shapley_hpp
#define snicholls_core_game_shapley_hpp

#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/game/coalition.hpp>
#include <snicholls/core/combinatorics/permutations.hpp>
#include <snicholls/core/stats/random.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <random>
#include <span>
#include <vector>

namespace snicholls::game {

    /// \brief An attribution: the baseline `v(none)`, the amount for each player, and the total `v(all)`.
    /// \ingroup core
    /// \tparam T      The value type.
    /// \tparam Values The players' amounts: `std::array<T, N>` or `std::vector<T>`.
    template <typename T, typename Values>
    struct attribution {
        // Copy-initialised, not braced: GCC 13 crashes (verify_ctor_sanity) constant-evaluating a braced
        // member of a user's value type with default member initialisers.
        T baseline = T();         ///< The value with no player: b.
        T total = T();            ///< The value with every player: what is attributed.
        Values values = Values(); ///< The amount attributed to each player: a.

        /// \brief The baseline plus every player's amount.
        /// \return b + sum(a).
        [[nodiscard]] constexpr T attributed() const
        {
            T sum = baseline;
            for (const T& a : values)
                sum = sum + a;
            return sum;
        }

        /// \brief What is left over: zero for a method with full attribution.
        /// \return total - b - sum(a).
        [[nodiscard]] constexpr T unattributed() const { return total - attributed(); }

        /// \brief The `snicholls::printable` member: `baseline b, [a0, a1, ...], total t`.
        /// \param os The stream.
        void print(std::ostream& os) const
            requires streamable<T>
        {
            os << "baseline " << baseline << ", [";
            const char* separator = "";
            for (const T& a : values) {
                os << separator << a;
                separator = ", ";
            }
            os << "], total " << total;
        }
    };

    /// \brief An attribution to `N` players, all on the stack.
    /// \ingroup core
    template <typename T, std::size_t N>
    using fixed_attribution = attribution<T, std::array<T, N>>;

    /// \brief An attribution to a number of players known at run time.
    /// \ingroup core
    template <typename T>
    using dynamic_attribution = attribution<T, std::vector<T>>;

    /// \brief A sampled attribution, with the standard error of each player's amount and the cost.
    /// \ingroup core
    /// \tparam T      The value type, floating point.
    /// \tparam Values The players' amounts.
    template <std::floating_point T, typename Values>
    struct sampled_attribution : attribution<T, Values> {
        Values standard_errors = Values();   ///< One per player; NaN with fewer than two samples.
        std::size_t samples = 0;      ///< Samples averaged: orders (antithetic pairs count once), or lifts per player.
        std::size_t evaluations = 0;  ///< Calls made to the game, repeats included.
    };

    /// \brief How `shapley_sampled` samples.
    /// \ingroup core
    struct permutation_sampling {
        std::size_t permutations = 256;   ///< Orders drawn; with `antithetic`, each also walked in reverse.
        bool antithetic = true;           ///< Walk each order both ways and average: lower variance, as SHAP does.
    };

    /// \brief How `shapley_sampled_lifts` samples.
    /// \ingroup core
    struct lift_sampling {
        std::size_t samples_per_player = 256;   ///< Lifts drawn for each player.
        bool scale_to_total = true;             ///< Rescale so that the attribution is full, as the paper suggests.
    };

    namespace detail {

        template <typename T>
        constexpr game_scalar_t<T> shapley_weight(std::size_t n, std::size_t size) noexcept
        {
            using scalar = game_scalar_t<T>;
            return scalar(1) / (static_cast<scalar>(n) * static_cast<scalar>(binomial(n - 1, size)));
        }

        // The table of a game: the table itself for a table_game, otherwise every value, once.
        template <game V, typename F>
        constexpr void with_table(std::size_t n, const V& v, F&& f)
        {
            const std::vector<game_value_t<V>> table = tabulate(n, v);
            f(std::span<const game_value_t<V>>(table));
        }
        template <typename T, typename F>
        constexpr void with_table(std::size_t n, const table_game<T>& v, F&& f)
        {
            assert(v.players() == n && "snicholls::game: the table is not for this number of players");
            static_cast<void>(n);
            f(v.table());
        }

        // Formula (3): each edge S -> S + i of the hypercube, weighted |S|! (n - |S| - 1)! / n!.
        template <typename T>
        constexpr void shapley_from_table(std::size_t n, std::span<const T> table, std::span<T> out)
        {
            std::array<game_scalar_t<T>, 64> weight{};
            for (std::size_t k = 0; k < n; ++k)
                weight[k] = shapley_weight<T>(n, k);
            const coalition all = grand_coalition(n);
            for (coalition s = 0; s < all; ++s) {
                const auto w = weight[coalition_size(s)];
                const T& here = table[s];
                for (coalition missing = all & ~s; missing != 0; missing &= missing - 1) {
                    const auto i = static_cast<std::size_t>(std::countr_zero(missing));
                    out[i] = out[i] + (table[s | singleton(i)] - here) * w;
                }
            }
        }

        template <typename T>
        constexpr void banzhaf_from_table(std::size_t n, std::span<const T> table, std::span<T> out)
        {
            const game_scalar_t<T> w = n == 0 ? game_scalar_t<T>(1) : game_scalar_t<T>(1) / static_cast<game_scalar_t<T>>(table_size(n - 1));
            const coalition all = grand_coalition(n);
            for (coalition s = 0; s < all; ++s)
                for (coalition missing = all & ~s; missing != 0; missing &= missing - 1) {
                    const auto i = static_cast<std::size_t>(std::countr_zero(missing));
                    out[i] = out[i] + (table[s | singleton(i)] - table[s]) * w;
                }
        }

        template <game V, typename Values>
        constexpr void sequential_walk(const V& v, std::span<const std::size_t> order, attribution<game_value_t<V>, Values>& out)
        {
            coalition seen = 0;
            for (const std::size_t i : order) {
                assert(i < order.size() && !contains(seen, i) && "snicholls::game::permuted_sequential: the order is not a permutation");
                seen = with(seen, i);
            }
            out.baseline = std::invoke(v, coalition{0});
            game_value_t<V> previous = out.baseline;
            coalition s = 0;
            for (const std::size_t i : order) {
                s = with(s, i);
                game_value_t<V> current = std::invoke(v, s);
                out.values[i] = current - previous;
                previous = std::move(current);
            }
            out.total = previous;
        }

        template <game V, typename Values>
        constexpr void one_at_a_time_into(std::size_t n, const V& v, attribution<game_value_t<V>, Values>& out)
        {
            out.baseline = std::invoke(v, coalition{0});
            for (std::size_t i = 0; i < n; ++i)
                out.values[i] = std::invoke(v, singleton(i)) - out.baseline;
            out.total = std::invoke(v, grand_coalition(n));
        }

        template <game V, typename Values>
        constexpr void leave_one_out_into(std::size_t n, const V& v, attribution<game_value_t<V>, Values>& out)
        {
            out.baseline = std::invoke(v, coalition{0});
            out.total = std::invoke(v, grand_coalition(n));
            for (std::size_t i = 0; i < n; ++i)
                out.values[i] = out.total - std::invoke(v, without(grand_coalition(n), i));
        }

        // Welford's running mean and variance, one per player.
        template <std::floating_point T>
        struct moments {
            std::vector<T> mean, m2;
            std::size_t count = 0;

            explicit moments(std::size_t n) : mean(n), m2(n) {}

            void add(std::span<const T> sample)
            {
                ++count;
                for (std::size_t i = 0; i < mean.size(); ++i) {
                    const T delta = sample[i] - mean[i];
                    mean[i] += delta / static_cast<T>(count);
                    m2[i] += delta * (sample[i] - mean[i]);
                }
            }

            T standard_error(std::size_t i) const noexcept
            {
                if (count < 2)
                    return std::numeric_limits<T>::quiet_NaN();
                return std::sqrt(m2[i] / static_cast<T>(count - 1) / static_cast<T>(count));
            }
        };

        template <game V, stats::full_range_generator G, typename Values>
        void sample_orders(std::size_t n, const V& v, G& g, permutation_sampling options, sampled_attribution<game_value_t<V>, Values>& out)
        {
            using T = game_value_t<V>;
            out.baseline = std::invoke(v, coalition{0});
            out.total = std::invoke(v, grand_coalition(n));
            out.evaluations = 2;
            moments<T> stats(n);
            std::vector<std::size_t> order(n);
            for (std::size_t i = 0; i < n; ++i)
                order[i] = i;
            std::vector<T> forward(n), sample(n);
            const auto walk = [&](auto first, auto last, std::vector<T>& into) {
                T previous = out.baseline;
                coalition s = 0;
                for (auto it = first; it != last; ++it) {
                    s = with(s, *it);
                    const bool complete = s == grand_coalition(n);
                    T current = complete ? out.total : std::invoke(v, s);
                    out.evaluations += complete ? 0 : 1;
                    into[*it] = current - previous;
                    previous = current;
                }
            };
            for (std::size_t p = 0; p < options.permutations && n > 0; ++p) {
                stats::shuffle(std::span<std::size_t>(order), g);
                walk(order.begin(), order.end(), forward);
                if (options.antithetic) {
                    walk(order.rbegin(), order.rend(), sample);
                    for (std::size_t i = 0; i < n; ++i)
                        sample[i] = (sample[i] + forward[i]) / T(2);
                } else {
                    sample = forward;
                }
                stats.add(std::span<const T>(sample));
            }
            for (std::size_t i = 0; i < n; ++i) {
                out.values[i] = stats.mean[i];
                out.standard_errors[i] = stats.standard_error(i);
            }
            out.samples = stats.count;
        }

        template <game V, stats::full_range_generator G, typename Values>
        void sample_lifts(std::size_t n, const V& v, G& g, lift_sampling options, sampled_attribution<game_value_t<V>, Values>& out)
        {
            using T = game_value_t<V>;
            out.baseline = std::invoke(v, coalition{0});
            out.total = std::invoke(v, grand_coalition(n));
            out.evaluations = 2;
            std::vector<std::size_t> others;
            T sum = T(0);
            for (std::size_t i = 0; i < n; ++i) {
                others.clear();
                for (std::size_t j = 0; j < n; ++j)
                    if (j != i)
                        others.push_back(j);
                T mean = T(0), m2 = T(0);
                std::size_t count = 0;
                for (std::size_t k = 0; k < options.samples_per_player; ++k) {
                    // |S| is uniform on 0 .. n-1 under the Shapley weights, then S uniform of that size
                    const auto size = static_cast<std::size_t>(stats::uniform_index(g, n));
                    coalition s = 0;
                    for (std::size_t j = 0; j < size; ++j) {
                        const auto pick = j + static_cast<std::size_t>(stats::uniform_index(g, others.size() - j));
                        std::swap(others[j], others[pick]);
                        s = with(s, others[j]);
                    }
                    const T lift = std::invoke(v, with(s, i)) - std::invoke(v, s);
                    out.evaluations += 2;
                    ++count;
                    const T delta = lift - mean;
                    mean += delta / static_cast<T>(count);
                    m2 += delta * (lift - mean);
                }
                out.values[i] = mean;
                out.standard_errors[i] = count < 2 ? std::numeric_limits<T>::quiet_NaN()
                                                   : std::sqrt(m2 / static_cast<T>(count - 1) / static_cast<T>(count));
                sum += mean;
            }
            out.samples = options.samples_per_player;
            const T target = out.total - out.baseline;
            if (options.scale_to_total && sum != T(0)) {
                const T factor = target / sum;
                for (std::size_t i = 0; i < n; ++i) {
                    out.values[i] *= factor;
                    out.standard_errors[i] *= std::abs(factor);
                }
            }
        }

    } // namespace detail

    // ---- exact ---------------------------------------------------------------------------------------------------

    /// \brief The exact Shapley attribution of a game of `N` players: 2^N evaluations, at compile time if the game is `constexpr`.
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v The game; a `table_game` is used without evaluating anything.
    /// \return The attribution, full: `unattributed()` is zero up to rounding.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> shapley(const V& v)
    {
        static_assert(N <= max_table_players, "snicholls::game::shapley: too many players for an exact attribution; sample");
        using T = game_value_t<V>;
        fixed_attribution<T, N> out{};
        detail::with_table(N, v, [&out](std::span<const T> table) {
            out.baseline = table.front();
            out.total = table.back();
            detail::shapley_from_table<T>(N, table, std::span<T>(out.values));
        });
        return out;
    }

    /// \brief The exact Shapley attribution of a game of `n` players known at run time.
    /// \ingroup core
    /// \tparam V The game's type.
    /// \param n The number of players, at most `max_table_players`.
    /// \param v The game.
    /// \return The attribution.
    /// \throws std::bad_alloc; whatever the game throws.
    template <game V>
    [[nodiscard]] constexpr dynamic_attribution<game_value_t<V>> shapley(std::size_t n, const V& v)
    {
        using T = game_value_t<V>;
        dynamic_attribution<T> out{};
        out.values.resize(n);
        detail::with_table(n, v, [&](std::span<const T> table) {
            out.baseline = table.front();
            out.total = table.back();
            detail::shapley_from_table<T>(n, table, std::span<T>(out.values));
        });
        return out;
    }

    /// \brief The Shapley attribution by its definition: the average of the permuted sequential attributions over all `N!` orders.
    ///
    /// Slower than `shapley` by a factor of about `N! / 2^N`, and equal to it; for teaching, and to check
    /// that the two agree. The orders come from `snicholls::for_each_permutation`.
    /// \ingroup core
    /// \tparam N The number of players, at most 10.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return The attribution.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> shapley_by_permutations(const V& v)
    {
        static_assert(N <= 10, "snicholls::game::shapley_by_permutations: more than 10! orders; use shapley");
        using T = game_value_t<V>;
        fixed_attribution<T, N> out{};
        detail::with_table(N, v, [&out](std::span<const T> table) {
            out.baseline = table.front();
            out.total = table.back();
            for_each_permutation<N>([&](const std::array<std::size_t, N>& order) {
                coalition s = 0;
                for (const std::size_t i : order) {
                    const coalition next = with(s, i);
                    out.values[i] = out.values[i] + (table[next] - table[s]);
                    s = next;
                }
            });
            const auto scale = game_scalar_t<T>(1) / static_cast<game_scalar_t<T>>(permutation_count_v<N>);
            for (T& a : out.values)
                a = a * scale;
        });
        return out;
    }

    /// \brief The Banzhaf value: each player's average lift over all coalitions, unweighted. Fair, not full.
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return The attribution; `unattributed()` is generally not zero.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> banzhaf(const V& v)
    {
        static_assert(N <= max_table_players, "snicholls::game::banzhaf: too many players");
        using T = game_value_t<V>;
        fixed_attribution<T, N> out{};
        detail::with_table(N, v, [&out](std::span<const T> table) {
            out.baseline = table.front();
            out.total = table.back();
            detail::banzhaf_from_table<T>(N, table, std::span<T>(out.values));
        });
        return out;
    }

    /// \brief The Shapley interaction index of every pair: `[i][j]` is how much `i` and `j` act together.
    ///
    /// Positive: complements, worth more together than apart; negative: substitutes. Zero for every
    /// pair exactly when the game is additive. The diagonal is zero.
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return A symmetric matrix.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr std::array<std::array<game_value_t<V>, N>, N> shapley_interactions(const V& v)
    {
        static_assert(N <= max_table_players, "snicholls::game::shapley_interactions: too many players");
        using T = game_value_t<V>;
        std::array<std::array<T, N>, N> out{};
        // With fewer than two players no coalition misses two, so the loop body never runs and nothing is divided.
        detail::with_table(N, v, [&out](std::span<const T> table) {
            using scalar = game_scalar_t<T>;
            const coalition all = grand_coalition(N);
            for (coalition s = 0; s <= all; ++s) {
                const coalition missing = all & ~s;
                if (coalition_size(missing) < 2)
                    continue;
                const scalar w = scalar(1) / (static_cast<scalar>(N - 1) * static_cast<scalar>(binomial(N - 2, coalition_size(s))));
                for (coalition a = missing; a != 0; a &= a - 1) {
                    const auto i = static_cast<std::size_t>(std::countr_zero(a));
                    for (coalition b = a & (a - 1); b != 0; b &= b - 1) {
                        const auto j = static_cast<std::size_t>(std::countr_zero(b));
                        const T delta = table[s | singleton(i) | singleton(j)] - table[s | singleton(i)] - table[s | singleton(j)] + table[s];
                        out[i][j] = out[i][j] + delta * w;
                        out[j][i] = out[i][j];
                    }
                }
            }
        });
        return out;
    }

    // ---- the comparison methods (n + 1 or n + 2 evaluations) --------------------------------------------------------

    /// \brief Permuted sequential attribution: switch the players on in `order`; each is credited its lift. Full, not fair.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v     The game, evaluated N + 1 times.
    /// \param order A permutation of 0 .. N-1.
    /// \return The attribution.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> permuted_sequential(const V& v, const std::array<std::size_t, N>& order)
    {
        fixed_attribution<game_value_t<V>, N> out{};
        detail::sequential_walk(v, std::span<const std::size_t>(order), out);
        return out;
    }

    /// \brief Permuted sequential attribution for a number of players known at run time.
    /// \ingroup core
    /// \tparam V The game's type.
    /// \param v     The game, evaluated `order.size() + 1` times.
    /// \param order A permutation of 0 .. n-1.
    /// \return The attribution.
    /// \throws std::bad_alloc; whatever the game throws.
    template <game V>
    [[nodiscard]] constexpr dynamic_attribution<game_value_t<V>> permuted_sequential(const V& v, std::span<const std::size_t> order)
    {
        dynamic_attribution<game_value_t<V>> out{};
        out.values.resize(order.size());
        detail::sequential_walk(v, order, out);
        return out;
    }

    /// \brief Sequential (off-the-top) attribution: players switched on in index order, 0 first.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return The attribution.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> sequential(const V& v)
    {
        std::array<std::size_t, N> order{};
        for (std::size_t i = 0; i < N; ++i)
            order[i] = i;
        return permuted_sequential<N>(v, order);
    }

    /// \brief One-at-a-time attribution: each player's lift from the baseline alone. Brinson-Hood-Beebower; not full.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return The attribution.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> one_at_a_time(const V& v)
    {
        fixed_attribution<game_value_t<V>, N> out{};
        detail::one_at_a_time_into(N, v, out);
        return out;
    }

    /// \brief Leave-one-out attribution: each player's lift when added last. Not full.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return The attribution.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> leave_one_out(const V& v)
    {
        fixed_attribution<game_value_t<V>, N> out{};
        detail::leave_one_out_into(N, v, out);
        return out;
    }

    // ---- sampled ---------------------------------------------------------------------------------------------------

    /// \brief The Shapley attribution estimated from random orders (Castro et al.; SHAP's permutation explainer).
    ///
    /// Each order is a permuted sequential attribution, so every estimate is full. Its cost is
    /// `n - 1` evaluations per walk, plus two; wrap an expensive game in `memoized` to reuse repeats.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type, floating-point valued.
    /// \tparam G A generator of 32 or 64 full bits (`std::mt19937`, `std::mt19937_64`, ...).
    /// \param v       The game.
    /// \param g       The generator; the same state gives the same result on every platform.
    /// \param options How many orders, and whether antithetic.
    /// \return The estimate, its standard errors and its cost.
    /// \throws std::bad_alloc; whatever the game throws.
    template <std::size_t N, game V, stats::full_range_generator G>
        requires std::floating_point<game_value_t<V>>
    [[nodiscard]] sampled_attribution<game_value_t<V>, std::array<game_value_t<V>, N>> shapley_sampled(const V& v, G& g,
                                                                                                        permutation_sampling options = {})
    {
        sampled_attribution<game_value_t<V>, std::array<game_value_t<V>, N>> out{};
        detail::sample_orders(N, v, g, options, out);
        return out;
    }

    /// \brief The Shapley attribution estimated from random orders, for a number of players known at run time.
    /// \ingroup core
    /// \tparam V The game's type, floating-point valued.
    /// \tparam G A generator of 32 or 64 full bits.
    /// \param n       The number of players, below 64.
    /// \param v       The game.
    /// \param g       The generator.
    /// \param options How many orders, and whether antithetic.
    /// \return The estimate.
    /// \throws std::bad_alloc; whatever the game throws.
    template <game V, stats::full_range_generator G>
        requires std::floating_point<game_value_t<V>>
    [[nodiscard]] sampled_attribution<game_value_t<V>, std::vector<game_value_t<V>>> shapley_sampled(std::size_t n, const V& v, G& g,
                                                                                                      permutation_sampling options = {})
    {
        sampled_attribution<game_value_t<V>, std::vector<game_value_t<V>>> out{};
        out.values.resize(n);
        out.standard_errors.resize(n);
        detail::sample_orders(n, v, g, options, out);
        return out;
    }

    /// \brief The Shapley attribution estimated from random lifts, the importance sampling of Moehle, Boyd and Ang.
    ///
    /// For each player, coalitions are drawn with the Shapley weights (the size uniformly, then the
    /// members) and the player's lift averaged. More accurate than sampling orders for the same number of
    /// evaluations in the paper's tests; full only when scaled.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type, floating-point valued.
    /// \tparam G A generator of 32 or 64 full bits.
    /// \param v       The game.
    /// \param g       The generator.
    /// \param options How many lifts per player, and whether to scale to full attribution.
    /// \return The estimate.
    /// \throws std::bad_alloc; whatever the game throws.
    template <std::size_t N, game V, stats::full_range_generator G>
        requires std::floating_point<game_value_t<V>>
    [[nodiscard]] sampled_attribution<game_value_t<V>, std::array<game_value_t<V>, N>> shapley_sampled_lifts(const V& v, G& g,
                                                                                                              lift_sampling options = {})
    {
        sampled_attribution<game_value_t<V>, std::array<game_value_t<V>, N>> out{};
        detail::sample_lifts(N, v, g, options, out);
        return out;
    }

    /// \brief The Shapley attribution estimated from random lifts, for a number of players known at run time.
    /// \ingroup core
    /// \tparam V The game's type, floating-point valued.
    /// \tparam G A generator of 32 or 64 full bits.
    /// \param n       The number of players, below 64.
    /// \param v       The game.
    /// \param g       The generator.
    /// \param options How many lifts per player, and whether to scale.
    /// \return The estimate.
    /// \throws std::bad_alloc; whatever the game throws.
    template <game V, stats::full_range_generator G>
        requires std::floating_point<game_value_t<V>>
    [[nodiscard]] sampled_attribution<game_value_t<V>, std::vector<game_value_t<V>>> shapley_sampled_lifts(std::size_t n, const V& v, G& g,
                                                                                                            lift_sampling options = {})
    {
        sampled_attribution<game_value_t<V>, std::vector<game_value_t<V>>> out{};
        out.values.resize(n);
        out.standard_errors.resize(n);
        detail::sample_lifts(n, v, g, options, out);
        return out;
    }

} // namespace snicholls::game

#endif /* snicholls_core_game_shapley_hpp */
