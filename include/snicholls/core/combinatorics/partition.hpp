//
//  partition.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Wolfram's `Partition` of a list into sublists, at compile time, and every partition of a set.
/// \ingroup core
///
/// **Overview.** `Partition[list, n, d]` cuts a list into sublists of length `n` with offset `d`:
/// `d == n` gives consecutive chunks, `d == 1` every window. The array is known at compile time, so the
/// result is an array of arrays, and it is `constexpr`.
///
/// | Wolfram                               | here                              |
/// | ------------------------------------- | --------------------------------- |
/// | `Partition[list, n]`                  | `partition<n>(list)`              |
/// | `Partition[list, n, d]`               | `partition<n, d>(list)`           |
/// | `Partition[list, n, d, {1, 1}, x]`    | `partition_padded<n, d>(list, x)` |
/// | `Partition[list, n, d, {1, 1}]`       | `partition_cyclic<n, d>(list)`    |
///
/// For type lists, `meta::partition_n_t<L, n, d>`.
///
/// A partition of a *set* is something else: a grouping of its elements into blocks.
/// `for_each_set_partition<N>` visits all `bell_v<N>` of them as restricted growth strings, which is
/// how to search the groupings of factors for `game::owen`.
///
/// \snippet combinatorics/partition_demo.cpp partition
///
/// **Relationships.** `game/owen.hpp`; `meta/type_list.hpp` for `partition_n_t`;
/// `combinatorics/permutations.hpp`. Group: \ref core.
///
/// **Guarantees.** `constexpr`, allocation-free. An element type need only be copyable. A sublist that
/// would run past the end is dropped by `partition`, padded by `partition_padded`, wrapped by
/// `partition_cyclic`. Set partitions come in lexicographic order of their restricted growth strings.
///
/// **Errors.** Compile errors with a message for `n == 0`, `d == 0`, and a Bell number that does not fit
/// 64 bits (above 25 elements).
///
/// **Performance.** Copies, nothing else.
///
/// **Tests and demos.** `tests/combinatorics/partition_test.cpp`, `demos/combinatorics/partition_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<type_traits>`, `<utility>`.
///
/// **Known issues.** Wolfram's other overhang specifications (`{kL, kR}` other than `{1, 1}`) are not
/// provided.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_combinatorics_partition_hpp
#define snicholls_core_combinatorics_partition_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief How many sublists `Partition[list, n, d]` makes of `m` elements: full ones only.
    /// \ingroup core
    /// \param m The list's length.
    /// \param n The sublists' length, above zero.
    /// \param d The offset, above zero.
    /// \return The count.
    [[nodiscard]] constexpr std::size_t partition_count(std::size_t m, std::size_t n, std::size_t d) noexcept
    {
        return m < n ? 0 : (m - n) / d + 1;
    }

    /// \brief How many sublists the padded and cyclic forms make: one starting at every offset within the list.
    /// \ingroup core
    /// \param m The list's length.
    /// \param d The offset, above zero.
    /// \return The count.
    [[nodiscard]] constexpr std::size_t partition_overhang_count(std::size_t m, std::size_t d) noexcept
    {
        return m == 0 ? 0 : (m - 1) / d + 1;
    }

    namespace detail {

        template <std::size_t N, std::size_t D, std::size_t Count, typename T, std::size_t M, typename Element>
        constexpr auto sublists(const std::array<T, M>& list, Element element)
        {
            return [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
                [[maybe_unused]] const auto one = [&]<std::size_t K, std::size_t... Js>(std::integral_constant<std::size_t, K>, std::index_sequence<Js...>) {
                    return std::array<T, N>{element(list, K * D + Js)...};
                };
                return std::array<std::array<T, N>, Count>{one(std::integral_constant<std::size_t, Ks>{}, std::make_index_sequence<N>{})...};
            }(std::make_index_sequence<Count>{});
        }

    } // namespace detail

    /// \brief `Partition[list, n, d]`: the sublists of length `N` starting every `D` elements, full ones only.
    /// \ingroup core
    /// \tparam N The sublists' length, above zero.
    /// \tparam D The offset, above zero; `N` by default, for consecutive chunks.
    /// \tparam T The element type.
    /// \tparam M The list's length.
    /// \param list The list.
    /// \return The sublists.
    /// \throws Whatever copying a `T` throws.
    template <std::size_t N, std::size_t D = N, typename T, std::size_t M>
    [[nodiscard]] constexpr std::array<std::array<T, N>, partition_count(M, N, D)> partition(const std::array<T, M>& list)
    {
        static_assert(N > 0 && D > 0, "snicholls::partition: the length and the offset must be above zero");
        return detail::sublists<N, D, partition_count(M, N, D)>(list, [](const std::array<T, M>& l, std::size_t i) { return l[i]; });
    }

    /// \brief `Partition[list, n, d, {1, 1}, pad]`: a sublist at every offset within the list, padded past its end.
    /// \ingroup core
    /// \tparam N The sublists' length, above zero.
    /// \tparam D The offset, above zero; `N` by default.
    /// \tparam T The element type.
    /// \tparam M The list's length.
    /// \param list The list.
    /// \param pad  What fills a sublist past the end.
    /// \return The sublists.
    /// \throws Whatever copying a `T` throws.
    template <std::size_t N, std::size_t D = N, typename T, std::size_t M>
    [[nodiscard]] constexpr std::array<std::array<T, N>, partition_overhang_count(M, D)> partition_padded(const std::array<T, M>& list,
                                                                                                        const T& pad)
    {
        static_assert(N > 0 && D > 0, "snicholls::partition_padded: the length and the offset must be above zero");
        return detail::sublists<N, D, partition_overhang_count(M, D)>(list, [&pad](const std::array<T, M>& l, std::size_t i) {
            return i < M ? l[i < M ? i : 0] : pad;
        });
    }

    /// \brief `Partition[list, n, d, {1, 1}]`: a sublist at every offset within the list, wrapping round to the start.
    /// \ingroup core
    /// \tparam N The sublists' length, above zero.
    /// \tparam D The offset, above zero; `N` by default.
    /// \tparam T The element type.
    /// \tparam M The list's length, above zero.
    /// \param list The list.
    /// \return The sublists.
    /// \throws Whatever copying a `T` throws.
    template <std::size_t N, std::size_t D = N, typename T, std::size_t M>
    [[nodiscard]] constexpr std::array<std::array<T, N>, partition_overhang_count(M, D)> partition_cyclic(const std::array<T, M>& list)
    {
        static_assert(N > 0 && D > 0 && M > 0, "snicholls::partition_cyclic: the length, the offset and the list must be above zero");
        return detail::sublists<N, D, partition_overhang_count(M, D)>(list, [](const std::array<T, M>& l, std::size_t i) { return l[i % M]; });
    }

    // ---- set partitions ----------------------------------------------------------------------------------------------

    namespace detail {

        consteval std::uint64_t bell(std::size_t n) noexcept
        {
            // The Bell triangle: each row starts with the previous row's last entry.
            std::array<std::uint64_t, 27> row{};
            std::array<std::uint64_t, 27> next{};
            row[0] = 1;
            for (std::size_t r = 1; r <= n; ++r) {
                next[0] = row[r - 1];
                for (std::size_t c = 1; c <= r; ++c)
                    next[c] = next[c - 1] + row[c - 1];
                row = next;
            }
            return row[0];
        }

    } // namespace detail

    /// \brief The Bell number: how many ways `N` elements can be grouped into blocks.
    /// \ingroup core
    /// \tparam N The number of elements, at most 25.
    template <std::size_t N>
    inline constexpr std::uint64_t bell_v = [] {
        static_assert(N <= 25, "snicholls::bell_v: the Bell number does not fit 64 bits above 25");
        return detail::bell(N);
    }();

    /// \brief Call `f(block_of, blocks)` for every partition of `N` elements into blocks.
    ///
    /// `block_of[i]` is the block of element `i`, numbered in order of first appearance (a restricted growth
    /// string: `block_of[0]` is 0 and each entry is at most one more than the largest before it), and
    /// `blocks` is how many there are. The first partition is one block, the last is every element alone.
    /// \ingroup core
    /// \tparam N The number of elements.
    /// \tparam F The callable's type.
    /// \param f Called as `f(const std::array<std::size_t, N>&, std::size_t)`.
    template <std::size_t N, typename F>
    constexpr void for_each_set_partition(F&& f) noexcept(noexcept(f(std::declval<const std::array<std::size_t, N>&>(), std::size_t{})))
    {
        std::array<std::size_t, N> block{};
        if constexpr (N == 0) {
            f(std::as_const(block), std::size_t{0});
        } else {
            std::array<std::size_t, N> largest{};   // largest[i]: the largest block among 0 .. i
            while (true) {
                f(std::as_const(block), largest[N - 1] + 1);
                std::size_t i = N - 1;
                while (i > 0 && block[i] > largest[i - 1])
                    --i;
                if (i == 0)
                    return;
                ++block[i];
                largest[i] = block[i] > largest[i - 1] ? block[i] : largest[i - 1];
                for (std::size_t j = i + 1; j < N; ++j) {
                    block[j] = 0;
                    largest[j] = largest[i];
                }
            }
        }
    }

    /// \brief The blocks of a set partition as bit masks, bit `i` for element `i`: the groups `game::owen` takes.
    /// \ingroup core
    /// \tparam N The number of elements, at most 64.
    /// \param block_of The block of each element, as `for_each_set_partition` gives it.
    /// \param blocks   How many blocks there are.
    /// \return One mask per block in the first `blocks` entries; zero after.
    template <std::size_t N>
    [[nodiscard]] constexpr std::array<std::uint64_t, N> blocks_as_masks(const std::array<std::size_t, N>& block_of, std::size_t blocks) noexcept
    {
        static_assert(N <= 64, "snicholls::blocks_as_masks: at most 64 elements");
        std::array<std::uint64_t, N> masks{};
        for (std::size_t i = 0; i < N; ++i)
            if (block_of[i] < blocks)
                masks[block_of[i]] |= std::uint64_t{1} << i;
        return masks;
    }

} // namespace snicholls

#endif /* snicholls_core_combinatorics_partition_hpp */
