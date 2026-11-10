//
//  owen.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The Owen value: Shapley attribution when the players come in groups (a coalition structure).
/// \ingroup core
///
/// **Overview.** Factors often come in groups: rates, credit, equity, volatility. The Owen value shares
/// the total first among the groups, as the Shapley value of the game between groups, and then within
/// each group, as a Shapley value of its members given every combination of the other groups. Players
/// in one group are switched on together in every order the groups are taken in, so a group of
/// substitutes is not penalised for being split. With every player in its own group, or all in one, it
/// is the Shapley value.
///
/// \snippet game/owen_demo.cpp owen
///
/// It does not tabulate the game. For `m` groups of sizes `b_k` it evaluates
/// `sum over k of 2^(m-1) 2^(b_k)` coalitions, so 50 factors in 5 groups of 10 take about 80,000
/// evaluations where Shapley would take 2^50: groups are how attribution scales.
///
/// **Relationships.** `game/shapley.hpp`; `combinatorics/partition.hpp` for enumerating groupings
/// (`for_each_set_partition`). Group: \ref core.
///
/// **Guarantees.** `constexpr`; full attribution; the sum over a group is that group's Shapley value in
/// the game between groups (the quotient game). Up to 64 players.
///
/// **Errors.** None: `groups` must partition the players (`is_partition_of`), which is asserted.
///
/// **Performance.** Not benchmarked; the count of evaluations above.
///
/// **Tests and demos.** `tests/game/owen_test.cpp`, `demos/game/owen_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cassert>`, `<cstddef>`, `<span>`, `<vector>`,
/// `game/coalition.hpp`, `game/shapley.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** New in 0.1.0, experimental.

#ifndef snicholls_core_game_owen_hpp
#define snicholls_core_game_owen_hpp

#include <snicholls/core/game/coalition.hpp>
#include <snicholls/core/game/shapley.hpp>

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace snicholls::game {

    /// \brief Whether `groups` partition `n` players: none empty, none overlapping, every player in one.
    /// \ingroup core
    /// \param n      The number of players, at most 64.
    /// \param groups The groups, as coalitions.
    /// \return Whether they are a partition.
    [[nodiscard]] constexpr bool is_partition_of(std::size_t n, std::span<const coalition> groups) noexcept
    {
        coalition seen = 0;
        for (const coalition g : groups) {
            if (g == 0 || (g & seen) != 0)
                return false;
            seen |= g;
        }
        return seen == grand_coalition(n);
    }

    namespace detail {

        template <game V, typename Values>
        constexpr void owen_into(std::size_t n, const V& v, std::span<const coalition> groups, attribution<game_value_t<V>, Values>& out)
        {
            using T = game_value_t<V>;
            using scalar = game_scalar_t<T>;
            assert(is_partition_of(n, groups) && "snicholls::game::owen: the groups do not partition the players");
            out.baseline = std::invoke(v, coalition{0});
            out.total = std::invoke(v, grand_coalition(n));
            const std::size_t m = groups.size();
            std::vector<T> inner;
            std::vector<std::size_t> members;
            for (std::size_t k = 0; k < m; ++k) {
                members.clear();
                for (coalition g = groups[k]; g != 0; g &= g - 1)
                    members.push_back(static_cast<std::size_t>(std::countr_zero(g)));
                const std::size_t b = members.size();
                inner.assign(table_size(b), T{});
                const coalition other_groups = grand_coalition(m) & ~singleton(k);
                for_each_subset_of(other_groups, [&](coalition r) {
                    coalition union_of_r = 0;
                    for (coalition rs = r; rs != 0; rs &= rs - 1)
                        union_of_r |= groups[static_cast<std::size_t>(std::countr_zero(rs))];
                    // Every subset of the group, on top of the other groups in r, once
                    for (coalition t = 0; t < table_size(b); ++t) {
                        coalition s = union_of_r;
                        for (coalition ts = t; ts != 0; ts &= ts - 1)
                            s = with(s, members[static_cast<std::size_t>(std::countr_zero(ts))]);
                        inner[t] = std::invoke(v, s);
                    }
                    const scalar outer_weight = shapley_weight<T>(m, coalition_size(r));
                    for (std::size_t j = 0; j < b; ++j)
                        for (coalition t = 0; t < table_size(b); ++t) {
                            if (contains(t, j))
                                continue;
                            const scalar w = outer_weight * shapley_weight<T>(b, coalition_size(t));
                            out.values[members[j]] = out.values[members[j]] + (inner[with(t, j)] - inner[t]) * w;
                        }
                });
            }
        }

    } // namespace detail

    /// \brief The Owen value of a game of `N` players in `groups`.
    /// \ingroup core
    /// \tparam N The number of players, at most 64.
    /// \tparam V The game's type.
    /// \param v      The game.
    /// \param groups A partition of the players, as coalitions.
    /// \return The attribution, full.
    /// \throws std::bad_alloc; whatever the game throws.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr fixed_attribution<game_value_t<V>, N> owen(const V& v, std::span<const coalition> groups)
    {
        static_assert(N <= 64, "snicholls::game::owen: at most 64 players");
        fixed_attribution<game_value_t<V>, N> out{};
        detail::owen_into(N, v, groups, out);
        return out;
    }

    /// \brief The Owen value of a game of `n` players known at run time.
    /// \ingroup core
    /// \tparam V The game's type.
    /// \param n      The number of players, at most 64.
    /// \param v      The game.
    /// \param groups A partition of the players, as coalitions.
    /// \return The attribution, full.
    /// \throws std::bad_alloc; whatever the game throws.
    template <game V>
    [[nodiscard]] constexpr dynamic_attribution<game_value_t<V>> owen(std::size_t n, const V& v, std::span<const coalition> groups)
    {
        dynamic_attribution<game_value_t<V>> out{};
        out.values.resize(n);
        detail::owen_into(n, v, groups, out);
        return out;
    }

    /// \brief The game between groups (the quotient game): group `k` plays as one, `v` of the union of the groups in `r`.
    /// \ingroup core
    /// \tparam V The game's type.
    template <game V>
    class quotient_game {
    public:
        /// \brief The game between these groups.
        /// \param v      The game; it must outlive this one.
        /// \param groups The groups; they must outlive this game.
        constexpr quotient_game(const V& v, std::span<const coalition> groups) noexcept : game_(&v), groups_(groups) {}

        /// \brief The value of a coalition of groups.
        /// \param r Bit `k` set for group `k`.
        /// \return `v` of the union of those groups.
        constexpr game_value_t<V> operator()(coalition r) const
        {
            coalition s = 0;
            for (; r != 0; r &= r - 1)
                s |= groups_[static_cast<std::size_t>(std::countr_zero(r))];
            return std::invoke(*game_, s);
        }

    private:
        const V* game_;
        std::span<const coalition> groups_;
    };

} // namespace snicholls::game

#endif /* snicholls_core_game_owen_hpp */
