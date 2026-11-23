//
//  binary.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::metric::hamming`, `jaccard` and `mismatches`: distances between bit vectors packed in unsigned words, and between sequences by the coordinates that differ.
/// \ingroup core
///
/// **Overview.** Binary codes (locality-sensitive hashes, binarised embeddings, Bloom-style signatures, feature
/// sets) are compared bit by bit. Here a bit vector is any `point` whose coordinates are unsigned integers (an
/// `std::array<std::uint64_t, 4>` is 256 bits); the words are compared with `std::popcount` of their XOR, AND and OR.
///
/// | Metric | `m(a, b)` | `is_metric` |
/// |---|---|---|
/// | `hamming` | the number of differing bits, `Σ popcount(aᵢ ⊕ bᵢ)` | yes |
/// | `jaccard` | `1 - |a ∧ b| / |a ∨ b|` over the set bits (Tanimoto distance); 0 for two empty sets | yes |
/// | `mismatches` | the number of coordinates that differ, for points of any coordinate type | yes |
///
/// `mismatches` is the Hamming distance between sequences (strings, categorical vectors); `hamming` is the Hamming
/// distance between their bits. Integer coordinates compare by value across signedness (`std::cmp_equal`), so -1
/// and `0xFFFFFFFFu` differ.
///
/// \snippet metric/binary_demo.cpp binary
///
/// **Relationships.** `metric/concepts.hpp`, `containers/ann_forest.hpp` (its bit-sampling splits),
/// `containers/bloom_filter.hpp` (sets as bits). Group: \ref core.
///
/// **Guarantees.** `constexpr`, `noexcept` when the coordinate reads are, no allocation. `hamming` and `mismatches`
/// return a count (`std::size_t`), `jaccard` a `double`.
///
/// **Errors.** None. Points of different lengths are a precondition violation.
///
/// **Performance.** Not benchmarked: one `popcount` per word.
///
/// **Tests and demos.** `tests/metric/binary_test.cpp`, `demos/metric/binary_demo.cpp`.
///
/// **Dependencies.** `<bit>`, `<concepts>`, `<cstddef>`, `<type_traits>`, `<utility>`, `metric/concepts.hpp`.
///
/// **Known issues.** Signed words are rejected (their bits are an implementation's business): use unsigned ones.
///
/// **Status.** Experimental, new in 0.20.0.

#ifndef snicholls_core_metric_binary_hpp
#define snicholls_core_metric_binary_hpp

#include <snicholls/core/metric/concepts.hpp>

#include <bit>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace snicholls::metric {

    namespace detail {
        // The integers std::cmp_equal takes: not bool, not the character types
        template <typename X>
        concept comparable_integer = std::integral<X> && !std::same_as<X, bool> && !std::same_as<X, char> && !std::same_as<X, wchar_t> &&
                                     !std::same_as<X, char8_t> && !std::same_as<X, char16_t> && !std::same_as<X, char32_t>;

        // Equality of two coordinates: std::cmp_equal for two integers (exact across signedness, no warning), == otherwise
        template <comparable_integer X, comparable_integer Y>
        constexpr bool same(X x, Y y) noexcept
        {
            return std::cmp_equal(x, y);
        }
        template <typename X, typename Y>
            requires(!(comparable_integer<X> && comparable_integer<Y>))
        constexpr bool same(const X& x, const Y& y) noexcept(noexcept(x == y))
        {
            return x == y;
        }
    } // namespace detail

    /// \brief A point whose coordinates are unsigned integer words: a packed bit vector.
    /// \ingroup core
    /// \tparam P The type.
    template <typename P>
    concept bit_vector = point<P> && std::unsigned_integral<value_t<P>> && !std::same_as<value_t<P>, bool>;

    /// \brief The Hamming distance between bit vectors: the number of differing bits.
    /// \ingroup core
    struct hamming {
        static constexpr bool is_metric = true;  ///< It obeys the triangle inequality.
        /// \brief The count. \tparam A A `bit_vector`. \tparam B A compatible `bit_vector`. \param a One vector.
        /// \param b The other, of as many words. \return `Σ popcount(aᵢ ⊕ bᵢ)`.
        template <bit_vector A, bit_vector B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr std::size_t operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            using W = std::common_type_t<value_t<A>, value_t<B>>;
            std::size_t s = 0;
            const std::size_t n = dimension(a);
            for (std::size_t i = 0; i < n; ++i)
                s += static_cast<std::size_t>(std::popcount(static_cast<W>(static_cast<W>(coordinate(a, i)) ^ static_cast<W>(coordinate(b, i)))));
            return s;
        }
    };

    /// \brief The Jaccard (Tanimoto) distance between bit sets: `1 - |a ∧ b| / |a ∨ b|`.
    /// \ingroup core
    struct jaccard {
        static constexpr bool is_metric = true;  ///< It obeys the triangle inequality.
        /// \brief The distance. \tparam A A `bit_vector`. \tparam B A compatible `bit_vector`. \param a One set.
        /// \param b The other, of as many words. \return In `[0, 1]`; 0 when both sets are empty.
        template <bit_vector A, bit_vector B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr double operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            using W = std::common_type_t<value_t<A>, value_t<B>>;
            std::size_t both = 0, either = 0;
            const std::size_t n = dimension(a);
            for (std::size_t i = 0; i < n; ++i) {
                const W x = static_cast<W>(coordinate(a, i)), y = static_cast<W>(coordinate(b, i));
                both += static_cast<std::size_t>(std::popcount(static_cast<W>(x & y)));
                either += static_cast<std::size_t>(std::popcount(static_cast<W>(x | y)));
            }
            return either == 0 ? 0.0 : 1.0 - static_cast<double>(both) / static_cast<double>(either);
        }
    };

    /// \brief The number of coordinates that differ: the Hamming distance between sequences of any coordinate type.
    /// \ingroup core
    struct mismatches {
        static constexpr bool is_metric = true;  ///< It obeys the triangle inequality.
        /// \brief The count. \tparam A A `point`. \tparam B A compatible `point`. \param a One sequence.
        /// \param b The other, of the same length. \return The positions where they differ.
        template <point A, point B>
            requires compatible_points<A, B>
        [[nodiscard]] constexpr std::size_t operator()(const A& a, const B& b) const noexcept(noexcept(coordinate(a, 0)) && noexcept(coordinate(b, 0)))
        {
            std::size_t s = 0;
            const std::size_t n = dimension(a);
            for (std::size_t i = 0; i < n; ++i)
                s += detail::same(coordinate(a, i), coordinate(b, i)) ? 0u : 1u;
            return s;
        }
    };

} // namespace snicholls::metric

#endif /* snicholls_core_metric_binary_hpp */
