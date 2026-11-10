//
//  harsanyi.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Harsanyi dividends: the part of a game's value created by each coalition and by no smaller one.
/// \ingroup core
///
/// **Overview.** Every game is a sum of "unanimity" pieces: `v(S) = sum over T within S of d(T)`. The
/// dividend `d(T)` is what the players of `T` create together beyond everything their smaller coalitions
/// already create: `d({i})` is player `i` alone, `d({i, j})` the pure interaction of the pair, and so on.
/// The Shapley value shares each dividend equally among its coalition, `phi_i = sum over T containing i
/// of d(T) / |T|`, which is the cleanest way to see why it is fair. A game is additive exactly when every
/// dividend of two or more players is zero, and the dividends of a Taylor expansion are its terms: the
/// pair dividend of spot and volatility is the vanna term.
///
/// \snippet game/harsanyi_demo.cpp dividends
///
/// `to_dividends` is the Mobius transform and `from_dividends` the zeta transform, each in place in
/// O(n 2^n) additions, each the other's inverse.
///
/// **Relationships.** `game/shapley.hpp` for the same values from lifts; `game/coalition.hpp` for
/// tables. Group: \ref core.
///
/// **Guarantees.** `constexpr`; the transforms allocate nothing and are exact for integer values.
///
/// **Errors.** None; a table's size must be a power of two (asserted).
///
/// **Performance.** Not benchmarked: n 2^(n-1) subtractions.
///
/// **Tests and demos.** `tests/game/harsanyi_test.cpp`, `demos/game/harsanyi_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<cassert>`, `<cstddef>`, `<span>`, `<vector>`, `game/coalition.hpp`,
/// `game/shapley.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** New in 0.5.0, experimental.

#ifndef snicholls_core_game_harsanyi_hpp
#define snicholls_core_game_harsanyi_hpp

#include <snicholls/core/game/coalition.hpp>
#include <snicholls/core/game/shapley.hpp>

#include <bit>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace snicholls::game {

    /// \brief Turn a table of values into its dividends, in place (the Mobius transform).
    /// \ingroup core
    /// \tparam T The value type.
    /// \param table 2^n values, indexed by coalition; on return, the dividends.
    template <game_value T>
    constexpr void to_dividends(std::span<T> table)
    {
        assert(std::has_single_bit(table.size()));
        for (std::size_t bit = 1; bit < table.size(); bit <<= 1)
            for (std::size_t s = 0; s < table.size(); ++s)
                if ((s & bit) != 0)
                    table[s] = table[s] - table[s ^ bit];
    }

    /// \brief Turn dividends back into the game's values, in place (the zeta transform).
    /// \ingroup core
    /// \tparam T The value type.
    /// \param table 2^n dividends, indexed by coalition; on return, the values.
    template <game_value T>
    constexpr void from_dividends(std::span<T> table)
    {
        assert(std::has_single_bit(table.size()));
        for (std::size_t bit = 1; bit < table.size(); bit <<= 1)
            for (std::size_t s = 0; s < table.size(); ++s)
                if ((s & bit) != 0)
                    table[s] = table[s] + table[s ^ bit];
    }

    /// \brief The Harsanyi dividend of every coalition of a game of `N` players, at compile time if the game is `constexpr`.
    /// \ingroup core
    /// \tparam N The number of players, at most 20.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return The dividends, indexed by coalition.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr std::array<game_value_t<V>, (std::size_t{1} << N)> harsanyi_dividends(const V& v)
    {
        auto table = make_table<N>(v);
        to_dividends(std::span<game_value_t<V>>(table));
        return table;
    }

    /// \brief The Harsanyi dividend of every coalition of a game of `n` players known at run time.
    /// \ingroup core
    /// \tparam V The game's type.
    /// \param n The number of players, at most `max_table_players`.
    /// \param v The game.
    /// \return The dividends, indexed by coalition.
    /// \throws std::bad_alloc; whatever the game throws.
    template <game V>
    [[nodiscard]] constexpr std::vector<game_value_t<V>> harsanyi_dividends(std::size_t n, const V& v)
    {
        auto table = tabulate(n, v);
        to_dividends(std::span<game_value_t<V>>(table));
        return table;
    }

    /// \brief The Shapley attribution from dividends: each dividend shared equally among its coalition.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param dividends 2^n dividends, indexed by coalition.
    /// \return The attribution; equal to `shapley` of the game they came from.
    /// \throws std::bad_alloc.
    template <game_value T>
    [[nodiscard]] constexpr dynamic_attribution<T> shapley_from_dividends(std::span<const T> dividends)
    {
        assert(std::has_single_bit(dividends.size()));
        const auto n = static_cast<std::size_t>(std::countr_zero(dividends.size()));
        dynamic_attribution<T> out{};
        out.values.resize(n);
        out.baseline = dividends[0];
        out.total = dividends[0];
        for (coalition s = 1; s < dividends.size(); ++s) {
            out.total = out.total + dividends[s];
            const T share = dividends[s] * (game_scalar_t<T>(1) / static_cast<game_scalar_t<T>>(coalition_size(s)));
            for (coalition members = s; members != 0; members &= members - 1) {
                const auto i = static_cast<std::size_t>(std::countr_zero(members));
                out.values[i] = out.values[i] + share;
            }
        }
        return out;
    }

    /// \brief The Banzhaf value from dividends: each dividend weighted 1 / 2^(|T| - 1) for each member.
    /// \ingroup core
    /// \tparam T The value type.
    /// \param dividends 2^n dividends, indexed by coalition.
    /// \return The attribution; equal to `banzhaf` of the game they came from.
    /// \throws std::bad_alloc.
    template <game_value T>
    [[nodiscard]] constexpr dynamic_attribution<T> banzhaf_from_dividends(std::span<const T> dividends)
    {
        assert(std::has_single_bit(dividends.size()));
        const auto n = static_cast<std::size_t>(std::countr_zero(dividends.size()));
        dynamic_attribution<T> out{};
        out.values.resize(n);
        out.baseline = dividends[0];
        out.total = dividends[0];
        for (coalition s = 1; s < dividends.size(); ++s) {
            out.total = out.total + dividends[s];
            const T share = dividends[s] * (game_scalar_t<T>(1) / static_cast<game_scalar_t<T>>(table_size(coalition_size(s) - 1)));
            for (coalition members = s; members != 0; members &= members - 1) {
                const auto i = static_cast<std::size_t>(std::countr_zero(members));
                out.values[i] = out.values[i] + share;
            }
        }
        return out;
    }

} // namespace snicholls::game

#endif /* snicholls_core_game_harsanyi_hpp */
