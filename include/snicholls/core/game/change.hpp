//
//  change.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Explain `f(to) - f(from)` by the inputs that moved: a game whose players are the moves.
/// \ingroup core
///
/// **Overview.** A book is worth `f(market)`. Overnight the market moved from `from` to `to`, each risk
/// factor a little, and the book's value changed. `change_game` is the game in which a coalition of
/// factors has moved and the rest have not; its Shapley attribution splits the change exactly, with no
/// residual and no Taylor approximation, and shares every cross effect (spot and volatility together)
/// fairly between the factors that cause it. `attribute_change` is that in one call.
///
/// \snippet game/change_demo.cpp change
///
/// The cost is 2^n revaluations of `f`. For n up to about 12 factors and a fast pricer that is real
/// time; above it, group the factors (`owen`) or sample (`shapley_sampled`). `finance/attribution.hpp`
/// gives the closed form for a second-order Taylor model, which costs O(n^2).
///
/// **Relationships.** `game/shapley.hpp`, `game/owen.hpp`; `game/tabular.hpp` is the same with many
/// background points; `finance/attribution.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`. A point is any copyable random-access container of inputs (`std::array`,
/// `std::vector`, a struct with `operator[]`); the game copies `from` for each evaluation and refers to
/// `f`, `from` and `to`, which must outlive it.
///
/// **Errors.** None of its own; `from` and `to` must be the same size (asserted).
///
/// **Performance.** The revaluations; nothing else to speak of.
///
/// **Tests and demos.** `tests/game/change_test.cpp`, `demos/game/change_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<cassert>`, `<cstddef>`, `<functional>`, `<ranges>`, `<type_traits>`, `<utility>`,
/// `game/coalition.hpp`, `game/shapley.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** New in 0.5.0, experimental.

#ifndef snicholls_core_game_change_hpp
#define snicholls_core_game_change_hpp

#include <snicholls/core/game/coalition.hpp>
#include <snicholls/core/game/shapley.hpp>

#include <bit>
#include <cassert>
#include <cstddef>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>

namespace snicholls::game {

    /// \brief The game of the inputs that moved: `v(S)` is `f` with the inputs in `S` at `to` and the rest at `from`.
    /// \ingroup core
    /// \tparam F     Callable with a `Point`, returning a `game_value`.
    /// \tparam Point A copyable random-access range of inputs.
    template <typename F, std::ranges::random_access_range Point>
        requires std::copyable<Point> && game_value<std::remove_cvref_t<std::invoke_result_t<const F&, const Point&>>>
    class change_game {
    public:
        /// \brief The game of a move from `from` to `to`.
        /// \param f    The function; it must outlive the game.
        /// \param from The inputs before; it must outlive the game.
        /// \param to   The inputs after, the same size; it must outlive the game.
        constexpr change_game(const F& f, const Point& from, const Point& to) : f_(&f), from_(&from), to_(&to)
        {
            assert(std::ranges::size(from) == std::ranges::size(to));
        }

        /// \brief The number of players: the inputs.
        /// \return The count.
        [[nodiscard]] constexpr std::size_t players() const noexcept { return static_cast<std::size_t>(std::ranges::size(*from_)); }

        /// \brief `f` with the inputs in `s` moved.
        /// \param s The coalition of inputs that have moved.
        /// \return The value.
        /// \throws Whatever copying a point or `f` throws.
        constexpr auto operator()(coalition s) const
        {
            Point x = *from_;
            for (coalition m = s; m != 0; m &= m - 1) {
                const auto i = static_cast<std::size_t>(std::countr_zero(m));
                x[i] = (*to_)[i];
            }
            return std::invoke(*f_, std::as_const(x));
        }

    private:
        const F* f_;
        const Point* from_;
        const Point* to_;
    };

    /// \brief The exact Shapley attribution of `f(to) - f(from)` to the `N` inputs that moved.
    /// \ingroup core
    /// \tparam N     The number of inputs, at most `max_table_players`.
    /// \tparam F     The function's type.
    /// \tparam Point The inputs' type.
    /// \param f    The function, evaluated 2^N times.
    /// \param from The inputs before.
    /// \param to   The inputs after.
    /// \return The attribution: `baseline` is `f(from)`, `total` is `f(to)`, and nothing is unattributed.
    template <std::size_t N, typename F, std::ranges::random_access_range Point>
    [[nodiscard]] constexpr auto attribute_change(const F& f, const Point& from, const Point& to)
    {
        return shapley<N>(change_game<F, Point>(f, from, to));
    }

    /// \brief The exact Shapley attribution of `f(to) - f(from)`, for a number of inputs known at run time.
    /// \ingroup core
    /// \tparam F     The function's type.
    /// \tparam Point The inputs' type.
    /// \param f    The function, evaluated 2^n times.
    /// \param from The inputs before.
    /// \param to   The inputs after.
    /// \return The attribution.
    /// \throws std::bad_alloc; whatever `f` throws.
    template <typename F, std::ranges::random_access_range Point>
    [[nodiscard]] constexpr auto attribute_change(const F& f, const Point& from, const Point& to)
    {
        const change_game<F, Point> moved(f, from, to);
        return shapley(moved.players(), moved);
    }

} // namespace snicholls::game

#endif /* snicholls_core_game_change_hpp */
