//
//  permutations.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  That version built each row as a base class of one type, so compile time grew with N! template
//  instantiations: Clang stopped at 5 elements (a fold expression over N! arguments) and GCC took over four
//  minutes at 7. Here the rows are computed by a constexpr function, so only the table is instantiated.
//

/// \file
/// \brief Compile-time permutations of positions and values: `index_permutations`, `permutations`, `for_each_permutation`.
/// \ingroup core
///
/// **Overview.** Every ordering of `N` positions, or of `R` positions chosen from `N`, as a table
/// computed at compile time, in lexicographic order of position. For permutations of types, use
/// `snicholls::meta::permutations_t`.
///
/// \snippet combinatorics/permutations_demo.cpp permutations
///
/// Use `for_each_permutation` where a table would be too large: it visits the rows one at a
/// time and stores none. Use `std::next_permutation` for a run-time sequence of unknown length.
///
/// **Relationships.** `meta/type_list.hpp` builds `meta::permutations_t` on the same walk;
/// `constexpr_for.hpp` for other value loops. Group: \ref core.
///
/// **Guarantees.** Everything is `constexpr` and allocation-free. Rows are in lexicographic order
/// of position, so the first is the identity. Asking for more positions than exist (`R > N`)
/// gives no rows; `R == 0` gives one empty row. These functions permute positions: equal values
/// in `permutations` still give separate rows, so its row count is always
/// `permutation_count_v<N, R>`. Complexity is proportional to the rows visited.
///
/// **Errors.** A count too large for `std::size_t` (above 20! on a 64-bit platform) is a compile
/// error with a message, not a wrapped number.
///
/// **Performance.** Not benchmarked: the tables are built at compile time. `for_each_permutation`
/// at run time costs a recursive call per position per row.
///
/// **Tests and demos.** `tests/combinatorics/permutations_test.cpp`, `demos/combinatorics/permutations_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<limits>`, `<optional>`, `<type_traits>`, `<utility>`.
///
/// **Known issues.** A compiler's constant-evaluation budget limits a table's size: see the
/// figures in `tests/combinatorics/permutations_test.cpp`. Clang's default allows `index_permutations<8>` and MSVC's
/// budget is smaller and is not measured here; raise it with `-fconstexpr-steps` or
/// `/constexpr:steps`, or use `for_each_permutation` at run time, which has no such limit.
///
/// **Status.** New in 0.1.0.

#ifndef snicholls_core_combinatorics_permutations_hpp
#define snicholls_core_combinatorics_permutations_hpp

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {

        consteval std::optional<std::size_t> checked_permutation_count(std::size_t n, std::size_t r) noexcept
        {
            if (r > n)
                return std::size_t{0};
            std::size_t count = 1;
            for (std::size_t i = 0; i < r; ++i) {
                const std::size_t factor = n - i;
                if (count > std::numeric_limits<std::size_t>::max() / factor)
                    return std::nullopt;
                count *= factor;
            }
            return count;
        }

        // The one walk behind every permutation in core: each position takes each value with some
        // left, in order, and recurses. remaining[v] is how many of value v are still unused, so
        // repeated values give each distinct arrangement once, and all ones gives plain permutations.
        template <std::size_t K, std::size_t R, typename F>
        constexpr void for_each_arrangement(std::array<std::size_t, K>& remaining, std::array<std::size_t, R>& row, std::size_t depth,
                                            F& f) noexcept(noexcept(f(std::as_const(row))))
        {
            if (depth == row.size()) {
                f(std::as_const(row));
                return;
            }
            for (std::size_t value = 0; value < remaining.size(); ++value) {
                if (remaining[value] == 0)
                    continue;
                --remaining[value];
                row[depth] = value;
                for_each_arrangement(remaining, row, depth + 1, f);
                ++remaining[value];
            }
        }

        template <std::size_t N>
        consteval std::array<std::size_t, N> all_ones() noexcept
        {
            std::array<std::size_t, N> ones{};
            for (std::size_t i = 0; i < ones.size(); ++i)
                ones[i] = 1;
            return ones;
        }

    } // namespace detail

    /// \brief The number of orderings of `R` positions chosen from `N`: N! / (N - R)!, and 0 when `R > N`.
    /// \ingroup core
    /// \tparam N The number of positions.
    /// \tparam R The length of each ordering.
    template <std::size_t N, std::size_t R = N>
    struct permutation_count : std::integral_constant<std::size_t, detail::checked_permutation_count(N, R).value_or(0)> {
        static_assert(detail::checked_permutation_count(N, R).has_value(),
                      "snicholls::permutation_count: the count does not fit in std::size_t");
    };

    /// \brief `permutation_count<N, R>::value`.
    /// \ingroup core
    /// \tparam N The number of positions.
    /// \tparam R The length of each ordering.
    template <std::size_t N, std::size_t R = N>
    inline constexpr std::size_t permutation_count_v = permutation_count<N, R>::value;

    /// \brief Call `f` with every ordering of `R` positions chosen from `N`, storing no table.
    /// \ingroup core
    /// \tparam N The number of positions.
    /// \tparam R The length of each ordering.
    /// \tparam F The callable's type.
    /// \param f Called as `f(const std::array<std::size_t, R>&)`, in lexicographic order; the array is reused between calls.
    template <std::size_t N, std::size_t R = N, typename F>
    constexpr void for_each_permutation(F&& f) noexcept(noexcept(f(std::declval<const std::array<std::size_t, R>&>())))
    {
        std::array<std::size_t, N> remaining = detail::all_ones<N>();
        std::array<std::size_t, R> row{};
        detail::for_each_arrangement(remaining, row, 0, f);
    }

    /// \brief Every ordering of `R` positions chosen from `N`, as a table.
    /// \ingroup core
    /// \tparam N The number of positions.
    /// \tparam R The length of each ordering.
    /// \return `permutation_count_v<N, R>` rows, in lexicographic order.
    template <std::size_t N, std::size_t R = N>
    [[nodiscard]] constexpr std::array<std::array<std::size_t, R>, permutation_count_v<N, R>> index_permutations() noexcept
    {
        std::array<std::array<std::size_t, R>, permutation_count_v<N, R>> rows{};
        std::size_t next = 0;
        for_each_permutation<N, R>([&rows, &next](const std::array<std::size_t, R>& row) noexcept { rows[next++] = row; });
        return rows;
    }

    namespace detail {

        template <std::size_t N, std::size_t R>
        inline constexpr auto permutation_table = index_permutations<N, R>();

        template <std::size_t R, std::size_t I, typename T, std::size_t N, std::size_t... Js>
        constexpr std::array<T, R> permuted_row(const std::array<T, N>& values, std::index_sequence<Js...>)
        {
            return std::array<T, R>{values[permutation_table<N, R>[I][Js]]...};
        }

        template <std::size_t R, typename T, std::size_t N, std::size_t... Is>
        constexpr std::array<std::array<T, R>, sizeof...(Is)> permuted_rows(const std::array<T, N>& values, std::index_sequence<Is...>)
        {
            return std::array<std::array<T, R>, sizeof...(Is)>{permuted_row<R, Is>(values, std::make_index_sequence<R>{})...};
        }

    } // namespace detail

    /// \brief Every ordering of `R` of the values, by position; `T` need only be copyable, not default-constructible.
    /// \ingroup core
    /// \tparam R The length of each ordering.
    /// \tparam T The value type.
    /// \tparam N The number of values.
    /// \param values The values.
    /// \return `permutation_count_v<N, R>` rows, in lexicographic order of position. Equal values still give separate rows.
    /// \throws Whatever copying a `T` throws.
    template <std::size_t R, typename T, std::size_t N>
    [[nodiscard]] constexpr std::array<std::array<T, R>, permutation_count_v<N, R>>
    permutations(const std::array<T, N>& values) noexcept(std::is_nothrow_copy_constructible_v<T>)
    {
        return detail::permuted_rows<R>(values, std::make_index_sequence<permutation_count_v<N, R>>{});
    }

    /// \brief Every ordering of all the values, by position.
    /// \ingroup core
    /// \tparam T The value type.
    /// \tparam N The number of values.
    /// \param values The values.
    /// \return N! rows, in lexicographic order of position.
    /// \throws Whatever copying a `T` throws.
    template <typename T, std::size_t N>
    [[nodiscard]] constexpr std::array<std::array<T, N>, permutation_count_v<N>>
    permutations(const std::array<T, N>& values) noexcept(std::is_nothrow_copy_constructible_v<T>)
    {
        return permutations<N>(values);
    }

} // namespace snicholls

#endif /* snicholls_core_combinatorics_permutations_hpp */
