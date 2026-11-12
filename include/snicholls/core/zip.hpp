//
//  zip.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  This is the same idea on C++20, on every compiler core supports. The type-level counterpart is meta::zip_t
//  in meta/type_list.hpp.
//

/// \file
/// \brief `snicholls::zip(a, b, ...)`: walk several ranges together, stopping at the shortest.
/// \ingroup core
///
/// **Overview.** `zip` is `std::views::zip` for C++20: each element is a `std::tuple` of references,
/// one per range, so a structured binding reads and writes the originals.
///
/// \snippet zip_demo.cpp zip
///
/// Use it for ranges whose length is known at run time. To walk tuples, whose element types differ,
/// use `thread_apply`. To zip types, use `meta::zip_t`.
///
/// **Relationships.** `thread_apply.hpp` for tuples, `meta/type_list.hpp` for types, and
/// `acout.hpp` prints a `zip` as it prints any range. Group: \ref core.
///
/// **Guarantees.** `constexpr` and allocation-free. A range passed as an lvalue is referred to and must
/// outlive the view; one passed as an rvalue is moved into it (`std::views::all`). The view is a `forward_range`
/// when every range is and an `input_range` otherwise, except as noted under known issues. It has `size()` when
/// every range has, the smallest of them. Iterators are invalidated as the underlying ranges' are.
/// Zipping views costs their sizes, nothing more: see the `static_assert`s in `tests/zip_test.cpp`.
///
/// **Errors.** None of its own; whatever the ranges' iterators do.
///
/// **Performance.** `benchmarks/zip_benchmark.cpp`, a dot product of two `std::vector<double>` of
/// 4,096 elements, against the index loop, after a fairness check. Release, Apple clang 17,
/// Xeon W-3245, 2026-11-07: 4.11 us through `zip` against
/// 4.04 us for the index loop, identical sums.
///
/// **Tests and demos.** `tests/zip_test.cpp`, `demos/zip_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cstddef>`, `<iterator>`, `<ranges>`, `<tuple>`,
/// `<type_traits>`, `<utility>`. It specialises `std::tuple_size`, `std::tuple_element` and
/// `std::basic_common_reference` for `zip_value`, its own type, as the standard permits. `<ranges>` is heavier than it looks.
///
/// **Known issues.** It has no reverse iteration, random access or pipe form (`a | zip(b)`), which
/// `std::views::zip` has no pipe form for either. Its end is a sentinel, so it works with
/// range-for and `std::ranges` algorithms, not classic iterator-pair algorithms. Its value type is
/// `snicholls::zip_value`, a `std::tuple` underneath, which declares the common reference C++20's
/// `std::tuple` lacks; compare a value with a tuple through `std::get`, not `==`.
///
/// **Status.** New in 0.5.0.

#ifndef snicholls_core_zip_hpp
#define snicholls_core_zip_hpp

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The value type of a `zip_view`: a `std::tuple` of the elements' values.
    ///
    /// Its own type, not `std::tuple`, only so that it can declare its common reference with the view's
    /// tuple of references. C++20's `std::tuple` has none, so without this a zip over `const` ranges or
    /// move-only elements would not be a `std::ranges::input_range`; C++23 fixes `std::tuple` itself.
    /// \ingroup core
    /// \tparam Ts The elements' value types.
    template <typename... Ts>
    struct zip_value : std::tuple<Ts...> {
        using std::tuple<Ts...>::tuple;
    };

    /// \brief The view `snicholls::zip` returns: several views walked together, ending with the shortest.
    /// \ingroup core
    /// \tparam Views The views; at least one, each an input range.
    template <std::ranges::view... Views>
        requires(sizeof...(Views) > 0 && (std::ranges::input_range<Views> && ...))
    class zip_view : public std::ranges::view_interface<zip_view<Views...>> {
        template <bool Const, typename View>
        using maybe_const = std::conditional_t<Const, const View, View>;

        template <bool Const>
        class sentinel;

        template <bool Const>
        class iterator {
            static constexpr bool all_forward = (std::ranges::forward_range<maybe_const<Const, Views>> && ...);
            friend class zip_view;
            template <bool>
            friend class sentinel;

            constexpr explicit iterator(std::tuple<std::ranges::iterator_t<maybe_const<Const, Views>>...> current)
                : current_(std::move(current))
            {
            }

        public:
            /// \brief Forward if every range is, otherwise input.
            using iterator_concept = std::conditional_t<all_forward, std::forward_iterator_tag, std::input_iterator_tag>;
            /// \brief Input: a dereference yields a tuple by value, not a reference.
            using iterator_category = std::input_iterator_tag;
            /// \brief A tuple of the elements' values.
            using value_type = zip_value<std::ranges::range_value_t<maybe_const<Const, Views>>...>;
            /// \brief The common difference type.
            using difference_type = std::common_type_t<std::ranges::range_difference_t<maybe_const<Const, Views>>...>;

            /// \brief A singular iterator.
            iterator() = default;

            /// \brief The current elements.
            /// \return A tuple of references, one per range.
            constexpr auto operator*() const
            {
                return std::apply([](const auto&... it) { return std::tuple<decltype(*it)...>(*it...); }, current_);
            }

            /// \brief Advance every range.
            /// \return `*this`.
            constexpr iterator& operator++()
            {
                std::apply([](auto&... it) { (static_cast<void>(++it), ...); }, current_);
                return *this;
            }

            /// \brief Advance every range, for input ranges.
            constexpr void operator++(int)
                requires(!all_forward)
            {
                ++*this;
            }

            /// \brief Advance every range, for forward ranges.
            /// \return The iterator as it was.
            constexpr iterator operator++(int)
                requires all_forward
            {
                iterator before = *this;
                ++*this;
                return before;
            }

            /// \brief Whether two iterators of one view are at the same place; they advance together, so one range decides.
            /// \param a An iterator.
            /// \param b Another.
            /// \return Whether they are equal.
            friend constexpr bool operator==(const iterator& a, const iterator& b)
                requires all_forward
            {
                return std::get<0>(a.current_) == std::get<0>(b.current_);
            }

        private:
            std::tuple<std::ranges::iterator_t<maybe_const<Const, Views>>...> current_{};
        };

        template <bool Const>
        class sentinel {
            friend class zip_view;

            constexpr explicit sentinel(std::tuple<std::ranges::sentinel_t<maybe_const<Const, Views>>...> end) : end_(std::move(end)) {}

            template <std::size_t... Is>
            constexpr bool reached(const iterator<Const>& it, std::index_sequence<Is...>) const
            {
                return (... || (std::get<Is>(it.current_) == std::get<Is>(end_)));
            }

        public:
            /// \brief A singular sentinel.
            sentinel() = default;

            /// \brief Whether any range has ended: the walk stops at the shortest.
            /// \param it The iterator.
            /// \param end The sentinel.
            /// \return Whether the iterator is at the end.
            friend constexpr bool operator==(const iterator<Const>& it, const sentinel& end)
            {
                return end.reached(it, std::index_sequence_for<Views...>{});
            }

        private:
            std::tuple<std::ranges::sentinel_t<maybe_const<Const, Views>>...> end_{};
        };

    public:
        /// \brief An empty view, for the views that can be default-constructed.
        zip_view() = default;

        /// \brief Zip these views.
        /// \param views The views.
        constexpr explicit zip_view(Views... views) : views_(std::move(views)...) {}

        /// \brief The first elements.
        /// \return An iterator.
        constexpr auto begin()
        {
            return iterator<false>(std::apply([](auto&... v) { return std::tuple(std::ranges::begin(v)...); }, views_));
        }

        /// \brief The first elements, of a view whose ranges can all be read when const.
        /// \return An iterator.
        constexpr auto begin() const
            requires(std::ranges::input_range<const Views> && ...)
        {
            return iterator<true>(std::apply([](const auto&... v) { return std::tuple(std::ranges::begin(v)...); }, views_));
        }

        /// \brief The end of the shortest range.
        /// \return A sentinel.
        constexpr auto end()
        {
            return sentinel<false>(std::apply([](auto&... v) { return std::tuple(std::ranges::end(v)...); }, views_));
        }

        /// \brief The end of the shortest range, of a view whose ranges can all be read when const.
        /// \return A sentinel.
        constexpr auto end() const
            requires(std::ranges::input_range<const Views> && ...)
        {
            return sentinel<true>(std::apply([](const auto&... v) { return std::tuple(std::ranges::end(v)...); }, views_));
        }

        /// \brief The length of the shortest range.
        /// \return The size.
        constexpr auto size()
            requires(std::ranges::sized_range<Views> && ...)
        {
            return std::apply(
                [](auto&... v) {
                    using common = std::common_type_t<std::make_unsigned_t<decltype(std::ranges::distance(v))>...>;
                    return std::min({static_cast<common>(std::ranges::size(v))...});
                },
                views_);
        }

        /// \brief The length of the shortest range, of a view whose ranges are all sized when const.
        /// \return The size.
        constexpr auto size() const
            requires(std::ranges::sized_range<const Views> && ...)
        {
            return std::apply(
                [](const auto&... v) {
                    using common = std::common_type_t<std::make_unsigned_t<decltype(std::ranges::distance(v))>...>;
                    return std::min({static_cast<common>(std::ranges::size(v))...});
                },
                views_);
        }

    private:
        std::tuple<Views...> views_;
    };

    /// \brief Walk these ranges together: `for (auto [price, size] : zip(prices, sizes))`.
    /// \ingroup core
    /// \tparam Ranges The ranges.
    /// \param ranges The ranges; an lvalue is referred to and must outlive the view, an rvalue is moved in.
    /// \return A `zip_view`, as long as the shortest range.
    template <std::ranges::viewable_range... Ranges>
        requires(sizeof...(Ranges) > 0)
    [[nodiscard]] constexpr auto zip(Ranges&&... ranges)
    {
        return zip_view<std::views::all_t<Ranges>...>(std::views::all(std::forward<Ranges>(ranges))...);
    }

} // namespace snicholls

// A zip_value is a tuple: structured bindings and std::get work on it as on its base.
/// \cond SNICHOLLS_INTERNAL
template <typename... Ts>
struct std::tuple_size<snicholls::zip_value<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {};

template <std::size_t I, typename... Ts>
struct std::tuple_element<I, snicholls::zip_value<Ts...>> : std::tuple_element<I, std::tuple<Ts...>> {};

// The common reference of a zip's reference and value: each element read through a const reference
// (int& with int gives const int&), which both convert to. Both orders, as the standard asks.
template <typename... Rs, typename... Ts, template <typename> class RQual, template <typename> class TQual>
    requires(sizeof...(Rs) == sizeof...(Ts))
struct std::basic_common_reference<std::tuple<Rs...>, snicholls::zip_value<Ts...>, RQual, TQual> {
    using type = std::tuple<std::common_reference_t<Rs, const Ts&>...>;
};

template <typename... Ts, typename... Rs, template <typename> class TQual, template <typename> class RQual>
    requires(sizeof...(Rs) == sizeof...(Ts))
struct std::basic_common_reference<snicholls::zip_value<Ts...>, std::tuple<Rs...>, TQual, RQual> {
    using type = std::tuple<std::common_reference_t<Rs, const Ts&>...>;
};
/// \endcond

#endif /* snicholls_core_zip_hpp */
