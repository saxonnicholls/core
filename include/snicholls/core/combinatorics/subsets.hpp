//
//  subsets.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Wolfram's `Subsets` and `Tuples` at compile time: combinations, power sets and every tuple, as tables or one at a time.
/// \ingroup core
///
/// **Overview.**
///
/// | Wolfram                | here                                      | count        |
/// | ---------------------- | ----------------------------------------- | ------------ |
/// | `Subsets[list, {k}]`   | `subsets<k>(list)`, `for_each_combination<n, k>` | C(n, k) |
/// | `Subsets[list]`        | `for_each_subset<n>` (by size, then in order) | 2^n      |
/// | `Tuples[list, k]`      | `tuples<k>(list)`, `for_each_tuple<n, k>` | n^k          |
///
/// Rows come in Wolfram's order: combinations lexicographic by position, subsets by size and then
/// lexicographically, tuples lexicographically with the last position varying fastest. The `for_each`
/// forms visit index rows and store nothing; the table forms build every row at compile time. For types,
/// `meta::subsets_t` and `meta::cartesian_power_t`.
///
/// \snippet combinatorics/subsets_demo.cpp subsets
///
/// **Relationships.** `combinatorics/permutations.hpp`, `combinatorics/partition.hpp`; `game/coalition.hpp`
/// enumerates subsets as bit masks; `meta/type_list.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, allocation-free; an element type need only be copyable. Visiting is O(1)
/// amortised per row.
///
/// **Errors.** A count too large for `std::size_t` is a compile error with a message.
///
/// **Performance.** Not benchmarked: compile time, or a few increments per row.
///
/// **Tests and demos.** `tests/combinatorics/subsets_test.cpp`, `demos/combinatorics/subsets_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<limits>`, `<optional>`, `<utility>`.
///
/// **Known issues.** A compiler's constant-evaluation budget limits the size of a table; use the `for_each`
/// forms at run time above it.
///
/// **Status.** New in 0.3.0.

#ifndef snicholls_core_combinatorics_subsets_hpp
#define snicholls_core_combinatorics_subsets_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace snicholls {

    /// \brief The binomial coefficient C(n, k), exactly, or none if it does not fit 64 bits; 0 when k > n.
    /// \ingroup core
    /// \param n The set's size.
    /// \param k The subsets' size.
    /// \return C(n, k).
    [[nodiscard]] constexpr std::optional<std::uint64_t> checked_binomial(std::uint64_t n, std::uint64_t k) noexcept
    {
        if (k > n)
            return 0;
        k = k < n - k ? k : n - k;
        std::uint64_t result = 1;
        for (std::uint64_t i = 1; i <= k; ++i) {
            // result * (n - k + i) / i, exactly: result is C(n - k + i - 1, i - 1), divisible after the multiply
            // result * (n - k + i) is divisible by i; with g = gcd(result, i), i / g divides the factor,
            // so (result / g) * (factor / (i / g)) is exact and overflows only if C(n, k) itself does
            const std::uint64_t factor = n - k + i;
            std::uint64_t g = result, h = i;
            while (h != 0) {
                const std::uint64_t r = g % h;
                g = h;
                h = r;
            }
            const std::uint64_t a = result / g, b = factor / (i / g);
            if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b)
                return std::nullopt;
            result = a * b;
        }
        return result;
    }

    /// \brief C(N, K) as a constant; a compile error if it does not fit.
    /// \ingroup core
    template <std::size_t N, std::size_t K>
    inline constexpr std::size_t combination_count_v = [] {
        constexpr auto c = checked_binomial(N, K);
        static_assert(c.has_value(), "snicholls::combination_count_v: C(N, K) does not fit 64 bits");
        return static_cast<std::size_t>(*c);
    }();

    /// \brief Call `f(row)` for every combination of `K` of the positions `0 .. N-1`, in lexicographic order.
    /// \ingroup core
    /// \tparam N The number of positions.
    /// \tparam K The size of each combination.
    /// \tparam F The callable's type.
    /// \param f Called as `f(const std::array<std::size_t, K>&)`, positions increasing; the array is reused.
    template <std::size_t N, std::size_t K, typename F>
    constexpr void for_each_combination(F&& f) noexcept(noexcept(f(std::declval<const std::array<std::size_t, K>&>())))
    {
        if constexpr (K > N) {
            return;
        } else {
            std::array<std::size_t, K> row{};
            for (std::size_t i = 0; i < K; ++i)
                row[i] = i;
            while (true) {
                f(std::as_const(row));
                std::size_t i = K;
                while (i > 0 && row[i - 1] == N - K + i - 1)
                    --i;
                if (i == 0)
                    return;
                ++row[i - 1];
                for (std::size_t j = i; j < K; ++j)
                    row[j] = row[j - 1] + 1;
            }
        }
    }

    /// \brief Every combination of `K` of the positions `0 .. N-1`, as a table.
    /// \ingroup core
    /// \tparam N The number of positions.
    /// \tparam K The size of each combination.
    /// \return `combination_count_v<N, K>` rows, lexicographic.
    template <std::size_t N, std::size_t K>
    [[nodiscard]] constexpr std::array<std::array<std::size_t, K>, combination_count_v<N, K>> index_combinations() noexcept
    {
        std::array<std::array<std::size_t, K>, combination_count_v<N, K>> rows{};
        std::size_t next = 0;
        for_each_combination<N, K>([&](const std::array<std::size_t, K>& row) noexcept { rows[next++] = row; });
        return rows;
    }

    namespace detail {
        template <std::size_t N, std::size_t K>
        inline constexpr auto combination_table = index_combinations<N, K>();

        template <std::size_t K, std::size_t I, typename T, std::size_t N, std::size_t... Js>
        constexpr std::array<T, K> chosen(const std::array<T, N>& values, std::index_sequence<Js...>)
        {
            return {values[combination_table<N, K>[I][Js]]...};
        }
    } // namespace detail

    /// \brief `Subsets[list, {k}]`: every `K`-element selection of the values, in order of position.
    /// \ingroup core
    /// \tparam K The size of each subset.
    /// \tparam T The value type.
    /// \tparam N The number of values.
    /// \param values The values.
    /// \return `combination_count_v<N, K>` rows.
    /// \throws Whatever copying a `T` throws.
    template <std::size_t K, typename T, std::size_t N>
    [[nodiscard]] constexpr std::array<std::array<T, K>, combination_count_v<N, K>> subsets(const std::array<T, N>& values)
    {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::array<std::array<T, K>, sizeof...(Is)>{detail::chosen<K, Is>(values, std::make_index_sequence<K>{})...};
        }(std::make_index_sequence<combination_count_v<N, K>>{});
    }

    /// \brief `Subsets[list]`: call `f(mask)` for every subset of `N` positions, by size and then lexicographically; bit `i` is position `i`.
    /// \ingroup core
    /// \tparam N The number of positions, below 64.
    /// \tparam F The callable's type.
    /// \param f Called as `f(std::uint64_t)`.
    template <std::size_t N, typename F>
    constexpr void for_each_subset(F&& f) noexcept(noexcept(f(std::uint64_t{})))
    {
        static_assert(N < 64, "snicholls::for_each_subset: at most 63 positions");
        [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
            (for_each_combination<N, Ks>([&](const std::array<std::size_t, Ks>& row) {
                 std::uint64_t mask = 0;
                 for (const std::size_t i : row)
                     mask |= std::uint64_t{1} << i;
                 f(mask);
             }),
             ...);
        }(std::make_index_sequence<N + 1>{});
    }

    namespace detail {
        consteval std::optional<std::size_t> checked_power(std::size_t n, std::size_t k) noexcept
        {
            std::size_t c = 1;
            for (std::size_t i = 0; i < k; ++i) {
                if (n != 0 && c > std::numeric_limits<std::size_t>::max() / n)
                    return std::nullopt;
                c *= n;
            }
            return c;
        }
    } // namespace detail

    /// \brief `N^K` as a constant; a compile error if it does not fit.
    /// \ingroup core
    template <std::size_t N, std::size_t K>
    inline constexpr std::size_t tuple_count_v = [] {
        constexpr auto c = detail::checked_power(N, K);
        static_assert(c.has_value(), "snicholls::tuple_count_v: N^K does not fit std::size_t");
        return *c;
    }();

    /// \brief `Tuples[range, K]`: call `f(row)` for every `K`-tuple of positions `0 .. N-1`, the last varying fastest.
    /// \ingroup core
    /// \tparam N The number of positions.
    /// \tparam K The length of each tuple.
    /// \tparam F The callable's type.
    /// \param f Called as `f(const std::array<std::size_t, K>&)`; the array is reused.
    template <std::size_t N, std::size_t K, typename F>
    constexpr void for_each_tuple(F&& f) noexcept(noexcept(f(std::declval<const std::array<std::size_t, K>&>())))
    {
        if constexpr (N == 0 && K > 0) {
            return;
        } else {
            std::array<std::size_t, K> row{};
            while (true) {
                f(std::as_const(row));
                std::size_t i = K;
                while (i > 0 && row[i - 1] == N - 1) {
                    row[i - 1] = 0;
                    --i;
                }
                if (i == 0)
                    return;
                ++row[i - 1];
            }
        }
    }

    /// \brief `Tuples[list, K]`: every `K`-tuple of the values, the last position varying fastest.
    /// \ingroup core
    /// \tparam K The length of each tuple.
    /// \tparam T The value type, default-constructible.
    /// \tparam N The number of values.
    /// \param values The values.
    /// \return `tuple_count_v<N, K>` rows.
    template <std::size_t K, typename T, std::size_t N>
    [[nodiscard]] constexpr std::array<std::array<T, K>, tuple_count_v<N, K>> tuples(const std::array<T, N>& values)
    {
        std::array<std::array<T, K>, tuple_count_v<N, K>> rows{};
        std::size_t next = 0;
        for_each_tuple<N, K>([&](const std::array<std::size_t, K>& row) {
            for (std::size_t i = 0; i < K; ++i)
                rows[next][i] = values[row[i]];
            ++next;
        });
        return rows;
    }

} // namespace snicholls

#endif /* snicholls_core_combinatorics_subsets_hpp */
