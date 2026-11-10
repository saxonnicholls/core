//
//  coalition.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Coalitions as bit strings, their enumeration, and games: functions from 2^N to a value.
/// \ingroup core
///
/// **Overview.** A coalition of players `0 .. n-1` is a `std::uint64_t` with bit `i` set when player `i`
/// is in it, so the `2^n` coalitions are the integers `0 .. 2^n - 1` and a game, a value for every
/// coalition, is a function or a table of `2^n` entries. This header enumerates coalitions in the orders
/// attribution needs, all `constexpr`:
///
/// | order                          | function                         | use                                     |
/// | ------------------------------ | -------------------------------- | --------------------------------------- |
/// | every coalition, ascending     | `for_each_coalition`             | filling a table                         |
/// | one player changes per step    | `for_each_coalition_gray`        | revaluing incrementally                 |
/// | every coalition of size k      | `for_each_coalition_of_size`     | one layer of the hypercube              |
/// | every subset / superset of S   | `for_each_subset_of` / `..._superset_of` | Mobius transforms, local checks |
///
/// \snippet game/coalition_demo.cpp coalitions
///
/// A game is anything callable with a `coalition` that returns a `game_value`: a lambda, a backtest
/// wrapper, `table_game` over a precomputed table, `memoized` over an expensive function. `make_table`
/// builds the table at compile time when the game is `constexpr`.
///
/// **Relationships.** `game/shapley.hpp`, `game/harsanyi.hpp`, `game/owen.hpp` and `game/properties.hpp`
/// take these games; `combinatorics/permutations.hpp` for orderings of players. Group: \ref core.
///
/// **Guarantees.** Everything but `memoized` is `constexpr` and allocation-free except where it returns a
/// `std::vector`. Enumerations visit each coalition exactly once. `n` is at most 63 for enumeration and
/// at most `max_table_players` for a table.
///
/// **Errors.** None: the limits are preconditions, checked by `assert` and, where `n` is a template
/// argument, by `static_assert`.
///
/// **Performance.** Each step is a few integer instructions; the cost is the game's.
///
/// **Tests and demos.** `tests/game/coalition_test.cpp`, `demos/game/coalition_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<bit>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`,
/// `<functional>`, `<span>`, `<type_traits>`, `<unordered_map>`, `<utility>`, `<vector>`.
///
/// **Known issues.** `memoized` caches in a hash map and is not thread-safe.
///
/// **Status.** New in 0.5.0, experimental.

#ifndef snicholls_core_game_coalition_hpp
#define snicholls_core_game_coalition_hpp

#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace snicholls::game {

    /// \brief A coalition: bit `i` set when player `i` is in it.
    /// \ingroup core
    using coalition = std::uint64_t;

    /// \brief The most players a table of every coalition may have: 2^30 entries.
    /// \ingroup core
    inline constexpr std::size_t max_table_players = 30;

    /// \brief "No player": what `for_each_coalition_gray` reports for the first, empty coalition.
    /// \ingroup core
    inline constexpr std::size_t no_player = static_cast<std::size_t>(-1);

    /// \brief The coalition of player `i` alone.
    /// \ingroup core
    /// \param i The player, below 64.
    /// \return The coalition.
    [[nodiscard]] constexpr coalition singleton(std::size_t i) noexcept { return coalition{1} << i; }

    /// \brief The coalition of all `n` players.
    /// \ingroup core
    /// \param n The number of players, at most 64.
    /// \return The coalition.
    [[nodiscard]] constexpr coalition grand_coalition(std::size_t n) noexcept { return n >= 64 ? ~coalition{0} : (coalition{1} << n) - 1; }

    /// \brief Whether player `i` is in `s`.
    /// \ingroup core
    /// \param s The coalition.
    /// \param i The player.
    /// \return Whether the bit is set.
    [[nodiscard]] constexpr bool contains(coalition s, std::size_t i) noexcept { return ((s >> i) & 1u) != 0; }

    /// \brief `s` with player `i` added.
    /// \ingroup core
    /// \param s The coalition.
    /// \param i The player.
    /// \return The coalition.
    [[nodiscard]] constexpr coalition with(coalition s, std::size_t i) noexcept { return s | singleton(i); }

    /// \brief `s` with player `i` removed.
    /// \ingroup core
    /// \param s The coalition.
    /// \param i The player.
    /// \return The coalition.
    [[nodiscard]] constexpr coalition without(coalition s, std::size_t i) noexcept { return s & ~singleton(i); }

    /// \brief The number of players in `s`.
    /// \ingroup core
    /// \param s The coalition.
    /// \return Its size.
    [[nodiscard]] constexpr std::size_t coalition_size(coalition s) noexcept { return static_cast<std::size_t>(std::popcount(s)); }

    /// \brief The number of coalitions of `n` players, 2^n.
    /// \ingroup core
    /// \param n The number of players, below 64.
    /// \return The count.
    [[nodiscard]] constexpr std::size_t table_size(std::size_t n) noexcept { return std::size_t{1} << n; }

    /// \brief The binomial coefficient C(n, k), exactly, from Pascal's triangle; 0 when k > n.
    /// \ingroup core
    /// \param n At most 64, where every coefficient fits 64 bits.
    /// \param k Any.
    /// \return C(n, k).
    [[nodiscard]] constexpr std::uint64_t binomial(std::size_t n, std::size_t k) noexcept
    {
        if (k > n)
            return 0;
        std::array<std::uint64_t, 65> row{};
        row[0] = 1;
        for (std::size_t r = 1; r <= n; ++r)
            for (std::size_t c = r; c > 0; --c)
                row[c] += row[c - 1];
        return row[k];
    }

    /// \brief Call `f(s)` for every coalition of `n` players, in ascending order of `s`.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \param n The number of players, below 64.
    /// \param f Called with each coalition.
    template <typename F>
    constexpr void for_each_coalition(std::size_t n, F&& f) noexcept(noexcept(f(coalition{})))
    {
        assert(n < 64);
        const coalition last = grand_coalition(n);
        for (coalition s = 0;; ++s) {
            f(s);
            if (s == last)
                break;
        }
    }

    /// \brief Call `f(sub)` for every subset of `s`, from `s` itself down to the empty coalition.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \param s The coalition.
    /// \param f Called with each subset.
    template <typename F>
    constexpr void for_each_subset_of(coalition s, F&& f) noexcept(noexcept(f(coalition{})))
    {
        for (coalition sub = s;; sub = (sub - 1) & s) {
            f(sub);
            if (sub == 0)
                break;
        }
    }

    /// \brief Call `f(sup)` for every coalition of `n` players containing `s`, from `s` up to all of them.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \param s The coalition, within the `n` players.
    /// \param n The number of players, below 64.
    /// \param f Called with each superset.
    template <typename F>
    constexpr void for_each_superset_of(coalition s, std::size_t n, F&& f) noexcept(noexcept(f(coalition{})))
    {
        const coalition all = grand_coalition(n);
        for (coalition sup = s;; sup = (sup + 1) | s) {
            f(sup);
            if (sup == all)
                break;
        }
    }

    /// \brief Call `f(s)` for every coalition of exactly `k` of `n` players, in ascending order (Gosper's hack).
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \param n The number of players, below 64.
    /// \param k The size; nothing is called when `k > n`.
    /// \param f Called with each coalition.
    template <typename F>
    constexpr void for_each_coalition_of_size(std::size_t n, std::size_t k, F&& f) noexcept(noexcept(f(coalition{})))
    {
        assert(n < 64);
        if (k > n)
            return;
        if (k == 0) {
            f(coalition{0});
            return;
        }
        const coalition limit = coalition{1} << n;
        for (coalition s = (coalition{1} << k) - 1; s < limit;) {
            f(s);
            const coalition lowest = s & (~s + 1);
            const coalition ripple = s + lowest;
            s = (((ripple ^ s) >> 2) / lowest) | ripple;
        }
    }

    /// \brief Call `f(s, player)` for every coalition in Gray-code order: each step adds or removes one player.
    ///
    /// The first call is the empty coalition with `no_player`; after it, `player` is the one that changed,
    /// added if `contains(s, player)`. A game that is cheaper to update than to evaluate afresh (moving one
    /// risk factor at a time) visits the whole hypercube in 2^n - 1 single moves.
    /// \ingroup core
    /// \tparam F The callable's type.
    /// \param n The number of players, below 64.
    /// \param f Called as `f(coalition, std::size_t)`.
    template <typename F>
    constexpr void for_each_coalition_gray(std::size_t n, F&& f) noexcept(noexcept(f(coalition{}, std::size_t{})))
    {
        assert(n < 64);
        coalition s = 0;
        f(s, no_player);
        const coalition count = grand_coalition(n);
        for (coalition step = 1; step <= count && step != 0; ++step) {
            const auto player = static_cast<std::size_t>(std::countr_zero(step));
            s ^= singleton(player);
            f(s, player);
        }
    }

    /// \brief A value a game may take: copyable, zero by default, added, subtracted and scaled.
    ///
    /// `double` is one; so is a struct of several metrics (return, risk, turnover) with those operators,
    /// which attributes them all in one pass.
    /// \ingroup core
    template <typename T>
    concept game_value = std::copyable<T> && std::default_initializable<T> && requires(const T a, const T b, double w) {
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * w } -> std::convertible_to<T>;
    };

    /// \brief A game: callable with a `coalition`, returning a `game_value`.
    /// \ingroup core
    template <typename V>
    concept game = std::invocable<const V&, coalition> && game_value<std::remove_cvref_t<std::invoke_result_t<const V&, coalition>>>;

    /// \brief The value type of a game.
    /// \ingroup core
    template <game V>
    using game_value_t = std::remove_cvref_t<std::invoke_result_t<const V&, coalition>>;

    /// \brief The scalar weights are computed in: `T` itself for a floating-point `T`, otherwise `double`.
    /// \ingroup core
    template <typename T>
    struct game_scalar {
        /// \brief The scalar.
        using type = double;
    };
    /// \brief The scalar for a floating-point value: itself.
    /// \ingroup core
    template <std::floating_point T>
    struct game_scalar<T> {
        /// \brief The scalar.
        using type = T;
    };
    /// \brief `game_scalar<T>::type`.
    /// \ingroup core
    template <typename T>
    using game_scalar_t = typename game_scalar<T>::type;

    /// \brief A game over a table of 2^n values, indexed by coalition: no copy, no evaluation.
    /// \ingroup core
    /// \tparam T The value type.
    template <game_value T>
    class table_game {
    public:
        /// \brief A game over this table.
        /// \pre `table.size()` is a power of two, 2^n with n at most `max_table_players`.
        /// \param table The values; it must outlive the game.
        constexpr explicit table_game(std::span<const T> table) noexcept : table_(table)
        {
            assert(std::has_single_bit(table.size()) && "snicholls::game::table_game: the table size is not a power of two");
        }

        /// \brief The value of `s`.
        /// \param s A coalition of the table's players.
        /// \return The value.
        [[nodiscard]] constexpr const T& operator()(coalition s) const noexcept { return table_[s]; }

        /// \brief The number of players, n.
        /// \return n.
        [[nodiscard]] constexpr std::size_t players() const noexcept { return static_cast<std::size_t>(std::countr_zero(table_.size())); }

        /// \brief The table.
        /// \return The view.
        [[nodiscard]] constexpr std::span<const T> table() const noexcept { return table_; }

    private:
        std::span<const T> table_;
    };

    /// \brief A game over a contiguous table: `shapley<3>(from_table(values))`.
    /// \ingroup core
    /// \tparam R The table's type: contiguous, of a `game_value`.
    /// \param table The table, 2^n entries; it must outlive the game.
    /// \return The game.
    template <std::ranges::contiguous_range R>
        requires game_value<std::ranges::range_value_t<R>>
    [[nodiscard]] constexpr table_game<std::ranges::range_value_t<R>> from_table(const R& table) noexcept
    {
        return table_game<std::ranges::range_value_t<R>>(std::span<const std::ranges::range_value_t<R>>(table));
    }

    /// \brief Every value of a game of `N` players, as a `std::array`; at compile time when the game is `constexpr`.
    /// \ingroup core
    /// \tparam N The number of players, at most 20 (a table on the stack); use `tabulate` above that.
    /// \tparam V The game's type.
    /// \param v The game.
    /// \return The table, indexed by coalition.
    template <std::size_t N, game V>
    [[nodiscard]] constexpr std::array<game_value_t<V>, (std::size_t{1} << N)> make_table(const V& v)
    {
        static_assert(N <= 20, "snicholls::game::make_table: more than 20 players; use tabulate, which allocates");
        std::array<game_value_t<V>, (std::size_t{1} << N)> table{};
        for_each_coalition(N, [&](coalition s) { table[s] = std::invoke(v, s); });
        return table;
    }

    /// \brief Every value of a game of `n` players, as a `std::vector`.
    /// \ingroup core
    /// \tparam V The game's type.
    /// \param n The number of players, at most `max_table_players`.
    /// \param v The game.
    /// \return The table, indexed by coalition.
    /// \throws std::bad_alloc; whatever the game throws.
    template <game V>
    [[nodiscard]] constexpr std::vector<game_value_t<V>> tabulate(std::size_t n, const V& v)
    {
        assert(n <= max_table_players);
        std::vector<game_value_t<V>> table(table_size(n));
        for_each_coalition(n, [&](coalition s) { table[s] = std::invoke(v, s); });
        return table;
    }

    /// \brief A game that evaluates each coalition at most once and remembers it: for a backtest, or a pricer.
    ///
    /// Use it with the sampling estimators and `owen`, which visit only some coalitions, some of them
    /// repeatedly. The exact methods tabulate anyway and gain nothing from it. Not thread-safe.
    /// \ingroup core
    /// \tparam V The game it wraps.
    template <game V>
    class memoized {
    public:
        /// \brief Wrap a game.
        /// \param v The game, copied or moved in.
        constexpr explicit memoized(V v) : game_(std::move(v)) {}

        /// \brief The value of `s`, evaluated on the first request only.
        /// \param s The coalition.
        /// \return The value.
        /// \throws std::bad_alloc; whatever the game throws, in which case nothing is cached.
        const game_value_t<V>& operator()(coalition s) const
        {
            const auto found = cache_.find(s);
            if (found != cache_.end())
                return found->second;
            return cache_.emplace(s, std::invoke(game_, s)).first->second;
        }

        /// \brief How many distinct coalitions have been evaluated.
        /// \return The count.
        [[nodiscard]] std::size_t evaluations() const noexcept { return cache_.size(); }

    private:
        V game_;
        mutable std::unordered_map<coalition, game_value_t<V>> cache_;
    };

} // namespace snicholls::game

#endif /* snicholls_core_game_coalition_hpp */
