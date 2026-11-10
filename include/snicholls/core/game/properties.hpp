//
//  properties.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Properties of a game: supermodular (convex), submodular, monotone, superadditive, additive, and the core.
/// \ingroup core
///
/// **Overview.** Whether an attribution is stable depends on the game's shape.
///
/// - **Supermodular**, or convex: `v(S + T) + v(S & T) >= v(S) + v(T)`; each player's lift grows as
///   others join, so features are complements. A long-gamma option portfolio's payoff in its factor moves is one. For a
///   supermodular game the Shapley value lies in the core (Shapley 1971): no group of features is
///   credited less than it earns on its own, so the attribution is one nobody can argue down.
/// - **Submodular**: the reverse; features are substitutes, as risk factors in a diversified book are.
/// - **Monotone**: adding a feature never lowers the value.
/// - **Superadditive**: two disjoint groups together earn at least what they earn apart.
/// - **Additive**: no interaction at all; every attribution method agrees, and Shapley is the lift.
///
/// Each check returns the first counterexample, so a failing property says where it fails.
///
/// \snippet game/properties_demo.cpp properties
///
/// **Relationships.** `game/shapley.hpp`, `game/harsanyi.hpp` (additive means no dividend of two or
/// more players). Group: \ref core.
///
/// **Guarantees.** `constexpr`; each tabulates the game once. Supermodularity, submodularity and
/// additivity are checked on the n(n-1)/2 2^(n-2) second differences, which suffices; superadditivity
/// on all 3^n pairs of disjoint coalitions.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked: O(n^2 2^n), and O(3^n) for superadditivity.
///
/// **Tests and demos.** `tests/game/properties_test.cpp`, `demos/game/properties_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<optional>`, `<span>`, `<vector>`, `game/coalition.hpp`,
/// `game/shapley.hpp`.
///
/// **Known issues.** The tolerance is absolute; scale it to the game.
///
/// **Status.** New in 0.5.0, experimental.

#ifndef snicholls_core_game_properties_hpp
#define snicholls_core_game_properties_hpp

#include <snicholls/core/game/coalition.hpp>
#include <snicholls/core/game/shapley.hpp>

#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace snicholls::game {

    /// \brief A game value that can be compared, for the checks.
    /// \ingroup core
    template <typename T>
    concept ordered_game_value = game_value<T> && std::totally_ordered<T>;

    /// \brief Where a second difference has the wrong sign: `v(S+i+j) - v(S+i) - v(S+j) + v(S)`.
    /// \ingroup core
    /// \tparam T The value type.
    template <typename T>
    struct interaction_violation {
        coalition base = 0;         ///< S.
        std::size_t first = 0;      ///< i.
        std::size_t second = 0;     ///< j.
        T difference = T();          ///< The second difference.
    };

    /// \brief Where adding a player lowers the value.
    /// \ingroup core
    /// \tparam T The value type.
    template <typename T>
    struct monotonicity_violation {
        coalition base = 0;      ///< S.
        std::size_t player = 0;  ///< i.
        T lift = T();             ///< v(S+i) - v(S), negative.
    };

    /// \brief Where two disjoint coalitions earn less together than apart.
    /// \ingroup core
    /// \tparam T The value type.
    template <typename T>
    struct superadditivity_violation {
        coalition first = 0;   ///< S.
        coalition second = 0;  ///< T, disjoint from S.
        T shortfall = T();      ///< v(S + T) - v(S) - v(T), negative.
    };

    /// \brief Where an allocation fails to be in the core.
    /// \ingroup core
    /// \tparam T The value type.
    template <typename T>
    struct core_violation {
        coalition blocking = 0;   ///< A coalition given less than it earns alone; the grand coalition if the total is wrong.
        T shortfall = T();         ///< Its allocation minus its value.
    };

    namespace detail {

        template <typename T, typename Keep>
        constexpr std::optional<interaction_violation<T>> find_second_difference(std::size_t n, std::span<const T> table, Keep keep)
        {
            const coalition all = grand_coalition(n);
            for (coalition s = 0; s <= all; ++s) {
                const coalition missing = all & ~s;
                for (coalition a = missing; a != 0; a &= a - 1) {
                    const auto i = static_cast<std::size_t>(std::countr_zero(a));
                    for (coalition b = a & (a - 1); b != 0; b &= b - 1) {
                        const auto j = static_cast<std::size_t>(std::countr_zero(b));
                        const T d = table[s | singleton(i) | singleton(j)] - table[s | singleton(i)] - table[s | singleton(j)] + table[s];
                        if (!keep(d))
                            return interaction_violation<T>{s, i, j, d};
                    }
                }
            }
            return std::nullopt;
        }

    } // namespace detail

    /// \brief The first second difference below `-tolerance`, or none if the game is supermodular (convex).
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far below zero still counts as zero.
    /// \return The counterexample, if any.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr std::optional<interaction_violation<game_value_t<V>>> find_supermodularity_violation(const V& v,
                                                                                                             game_value_t<V> tolerance = {})
    {
        using T = game_value_t<V>;
        std::optional<interaction_violation<T>> found;
        detail::with_table(N, v, [&](std::span<const T> table) {
            found = detail::find_second_difference<T>(N, table, [&](const T& d) { return !(d < T{} - tolerance); });
        });
        return found;
    }

    /// \brief The first second difference above `tolerance`, or none if the game is submodular.
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far above zero still counts as zero.
    /// \return The counterexample, if any.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr std::optional<interaction_violation<game_value_t<V>>> find_submodularity_violation(const V& v,
                                                                                                           game_value_t<V> tolerance = {})
    {
        using T = game_value_t<V>;
        std::optional<interaction_violation<T>> found;
        detail::with_table(N, v, [&](std::span<const T> table) {
            found = detail::find_second_difference<T>(N, table, [&](const T& d) { return !(T{} + tolerance < d); });
        });
        return found;
    }

    /// \brief The first second difference further than `tolerance` from zero, or none if the game is additive.
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far from zero still counts as zero.
    /// \return The counterexample, if any.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr std::optional<interaction_violation<game_value_t<V>>> find_interaction(const V& v, game_value_t<V> tolerance = {})
    {
        using T = game_value_t<V>;
        std::optional<interaction_violation<T>> found;
        detail::with_table(N, v, [&](std::span<const T> table) {
            found = detail::find_second_difference<T>(N, table, [&](const T& d) { return !(d < T{} - tolerance) && !(T{} + tolerance < d); });
        });
        return found;
    }

    /// \brief The first lift below `-tolerance`, or none if the game is monotone.
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far below zero still counts as zero.
    /// \return The counterexample, if any.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr std::optional<monotonicity_violation<game_value_t<V>>> find_monotonicity_violation(const V& v,
                                                                                                           game_value_t<V> tolerance = {})
    {
        using T = game_value_t<V>;
        std::optional<monotonicity_violation<T>> found;
        detail::with_table(N, v, [&](std::span<const T> table) {
            const coalition all = grand_coalition(N);
            for (coalition s = 0; s < all && !found; ++s)
                for (coalition missing = all & ~s; missing != 0 && !found; missing &= missing - 1) {
                    const auto i = static_cast<std::size_t>(std::countr_zero(missing));
                    const T lift = table[with(s, i)] - table[s];
                    if (lift < T{} - tolerance)
                        found = monotonicity_violation<T>{s, i, lift};
                }
        });
        return found;
    }

    /// \brief The first pair of disjoint coalitions that earn less together, or none if the game is superadditive.
    /// \ingroup core
    /// \tparam N The number of players, at most 20 (3^N pairs).
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far below zero still counts as zero.
    /// \return The counterexample, if any.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr std::optional<superadditivity_violation<game_value_t<V>>> find_superadditivity_violation(const V& v,
                                                                                                                 game_value_t<V> tolerance = {})
    {
        static_assert(N <= 20, "snicholls::game::find_superadditivity_violation: 3^N pairs; at most 20 players");
        using T = game_value_t<V>;
        std::optional<superadditivity_violation<T>> found;
        detail::with_table(N, v, [&](std::span<const T> table) {
            const coalition all = grand_coalition(N);
            for (coalition s = 1; s <= all && !found; ++s)
                for_each_subset_of(all & ~s, [&](coalition t) {
                    if (found || t == 0 || t < s)
                        return;
                    const T shortfall = table[s | t] - table[s] - table[t];
                    if (shortfall < T{} - tolerance)
                        found = superadditivity_violation<T>{s, t, shortfall};
                });
        });
        return found;
    }

    /// \brief Whether the game is supermodular (convex).
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far below zero still counts as zero.
    /// \return Whether it is.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr bool is_supermodular(const V& v, game_value_t<V> tolerance = {})
    {
        return !find_supermodularity_violation<N>(v, tolerance).has_value();
    }

    /// \brief Whether the game is submodular.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far above zero still counts as zero.
    /// \return Whether it is.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr bool is_submodular(const V& v, game_value_t<V> tolerance = {})
    {
        return !find_submodularity_violation<N>(v, tolerance).has_value();
    }

    /// \brief Whether the game is additive: no interaction between any players.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far from zero still counts as zero.
    /// \return Whether it is.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr bool is_additive(const V& v, game_value_t<V> tolerance = {})
    {
        return !find_interaction<N>(v, tolerance).has_value();
    }

    /// \brief Whether the game is monotone.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far below zero still counts as zero.
    /// \return Whether it is.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr bool is_monotone(const V& v, game_value_t<V> tolerance = {})
    {
        return !find_monotonicity_violation<N>(v, tolerance).has_value();
    }

    /// \brief Whether the game is superadditive.
    /// \ingroup core
    /// \tparam N The number of players, at most 20.
    /// \tparam V The game's type.
    /// \param v         The game.
    /// \param tolerance How far below zero still counts as zero.
    /// \return Whether it is.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr bool is_superadditive(const V& v, game_value_t<V> tolerance = {})
    {
        return !find_superadditivity_violation<N>(v, tolerance).has_value();
    }

    /// \brief Where an allocation fails to be in the core, or none if it is: it hands out exactly `v(all)` and every
    /// coalition gets at least what it earns alone.
    /// \ingroup core
    /// \tparam N The number of players, at most `max_table_players`.
    /// \tparam V The game's type.
    /// \param v          The game.
    /// \param allocation What each player gets; the baseline `v(none)` is subtracted from the game first.
    /// \param tolerance  How far short still counts as enough.
    /// \return The counterexample, if any.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr std::optional<core_violation<game_value_t<V>>> find_core_violation(const V& v,
                                                                                           const std::array<game_value_t<V>, N>& allocation,
                                                                                           game_value_t<V> tolerance = {})
    {
        using T = game_value_t<V>;
        std::optional<core_violation<T>> found;
        detail::with_table(N, v, [&](std::span<const T> table) {
            const coalition all = grand_coalition(N);
            const T base = table[0];
            for (coalition s = 1; s <= all && !found; ++s) {
                T given{};
                for (coalition m = s; m != 0; m &= m - 1)
                    given = given + allocation[static_cast<std::size_t>(std::countr_zero(m))];
                const T shortfall = given - (table[s] - base);
                const bool wrong_total = s == all && (shortfall < T{} - tolerance || T{} + tolerance < shortfall);
                if (wrong_total || shortfall < T{} - tolerance)
                    found = core_violation<T>{s, shortfall};
            }
        });
        return found;
    }

    /// \brief Whether an allocation is in the core.
    /// \ingroup core
    /// \tparam N The number of players.
    /// \tparam V The game's type.
    /// \param v          The game.
    /// \param allocation What each player gets, above the baseline.
    /// \param tolerance  How far short still counts as enough.
    /// \return Whether it is.
    template <std::size_t N, game V>
        requires ordered_game_value<game_value_t<V>>
    [[nodiscard]] constexpr bool in_core(const V& v, const std::array<game_value_t<V>, N>& allocation, game_value_t<V> tolerance = {})
    {
        return !find_core_violation<N>(v, allocation, tolerance).has_value();
    }

} // namespace snicholls::game

#endif /* snicholls_core_game_properties_hpp */
