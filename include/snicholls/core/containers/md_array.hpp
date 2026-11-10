//
//  md_array.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `containers::md_array<T, E...>` and `md_span`: multi-dimensional storage and views, row-major, zero-overhead indexing.
/// \ingroup core
///
/// **Overview.** A 3-, 4- or 5-dimensional table (scenario x tenor x strike, currency x book x day) is one
/// contiguous block indexed by a row-major offset. With the extents in the type, `md_array<T, E...>` holds the
/// block in place and its strides are constants, so `a(i, j, k)` compiles to the arithmetic a hand-written
/// `a[(i * J + j) * K + k]` does; with the extents known only at run time, `md_span<T, Rank>` is a view over
/// any contiguous storage. `for_each_index` visits every index in row-major order, unrolled at compile time
/// for an `md_array` through `constexpr_nest`.
///
/// \snippet containers/md_array_demo.cpp md_array
///
/// **Relationships.** `constexpr_for.hpp`'s `constexpr_nest`; C++23's `std::mdspan`, which this is a small
/// part of. Group: \ref core.
///
/// **Guarantees.** `constexpr`; `md_array` is `sizeof(T) * E0 * E1 * ...` and trivially copyable when `T` is;
/// indexing is unchecked (`at` checks); `md_span` is a pointer and `Rank` extents and strides.
///
/// **Errors.** `at` throws `std::out_of_range`.
///
/// **Performance.** `benchmarks/containers/md_array_benchmark.cpp`: a 16 x 16 x 16 x 16 grid summed in order,
/// Release, Apple clang 17, Xeon W-3245, 2026-11-07: 65.3 us through `a(i, j, k, l)`, 65.0 us through the
/// offsets written by hand.
///
/// **Tests and demos.** `tests/containers/md_array_test.cpp`, `demos/containers/md_array_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<span>`, `<stdexcept>`, `<utility>`, `constexpr_for.hpp`.
///
/// **Known issues.** Row-major only; no slicing (a sub-view of fewer dimensions) yet.
///
/// **Status.** New in 0.3.0.

#ifndef snicholls_core_containers_md_array_hpp
#define snicholls_core_containers_md_array_hpp

#include <snicholls/core/constexpr_for.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace snicholls::containers {

    /// \brief A multi-dimensional array with its extents in its type, stored in place, row-major.
    /// \ingroup core
    /// \tparam T       The element type.
    /// \tparam Extents The extents, each at least 1.
    template <typename T, std::size_t... Extents>
    class md_array {
        static_assert(sizeof...(Extents) >= 1 && ((Extents >= 1) && ...), "snicholls::containers::md_array: extents must be at least 1");

    public:
        /// \brief The number of dimensions.
        static constexpr std::size_t rank = sizeof...(Extents);
        /// \brief The extents.
        static constexpr std::array<std::size_t, rank> extents{Extents...};
        /// \brief The number of elements.
        static constexpr std::size_t size = (Extents * ...);
        /// \brief The row-major strides: the last dimension's is 1.
        static constexpr std::array<std::size_t, rank> strides = [] {
            std::array<std::size_t, rank> s{};
            std::size_t stride = 1;
            for (std::size_t d = rank; d-- > 0;) {
                s[d] = stride;
                stride *= extents[d];
            }
            return s;
        }();

        std::array<T, size> elements{};   ///< The elements, row-major; public so the array is an aggregate.

        /// \brief The element at these indices, unchecked.
        /// \tparam Is Integral indices, one per dimension.
        /// \param is The indices.
        /// \return It.
        template <typename... Is>
            requires(sizeof...(Is) == rank)
        [[nodiscard]] constexpr T& operator()(Is... is) noexcept
        {
            return elements[offset(is...)];
        }

        /// \brief The element at these indices, unchecked.
        /// \tparam Is Integral indices, one per dimension.
        /// \param is The indices.
        /// \return It.
        template <typename... Is>
            requires(sizeof...(Is) == rank)
        [[nodiscard]] constexpr const T& operator()(Is... is) const noexcept
        {
            return elements[offset(is...)];
        }

        /// \brief The element at these indices, checked.
        /// \tparam Is Integral indices, one per dimension.
        /// \param is The indices.
        /// \return It.
        /// \throws std::out_of_range when an index is past its extent.
        template <typename... Is>
            requires(sizeof...(Is) == rank)
        [[nodiscard]] constexpr T& at(Is... is)
        {
            std::size_t d = 0;
            const bool inside = ((static_cast<std::size_t>(is) < extents[d++]) && ...);
            if (!inside)
                throw std::out_of_range("snicholls::containers::md_array::at");
            return elements[offset(is...)];
        }

        /// \brief The row-major offset of these indices.
        /// \tparam Is Integral indices.
        /// \param is The indices.
        /// \return The offset.
        template <typename... Is>
        [[nodiscard]] static constexpr std::size_t offset(Is... is) noexcept
        {
            return offset_of(std::index_sequence_for<Is...>{}, static_cast<std::size_t>(is)...);
        }

        /// \brief Call `f(i0, i1, ...)` for every index in row-major order, each index an `std::integral_constant`, unrolled at compile time.
        /// \tparam F The callable's type.
        /// \param f The callable.
        template <typename F>
        static constexpr void for_each_index(F&& f)
        {
            constexpr_nest<Extents...>(std::forward<F>(f));
        }

        /// \brief The elements as a span.
        /// \return It.
        [[nodiscard]] constexpr std::span<T, size> flat() noexcept { return elements; }
        /// \brief The elements as a span.
        /// \return It.
        [[nodiscard]] constexpr std::span<const T, size> flat() const noexcept { return elements; }

        /// \brief Equal when every element is.
        friend constexpr bool operator==(const md_array&, const md_array&) = default;

    private:
        template <std::size_t... Ds, typename... Is>
        static constexpr std::size_t offset_of(std::index_sequence<Ds...>, Is... is) noexcept
        {
            return ((is * strides[Ds]) + ...);
        }
    };

    /// \brief A view of contiguous storage as a `Rank`-dimensional row-major array whose extents are known at run time.
    /// \ingroup core
    /// \tparam T    The element type (const for a read-only view).
    /// \tparam Rank The number of dimensions.
    template <typename T, std::size_t Rank>
    class md_span {
        static_assert(Rank >= 1, "snicholls::containers::md_span: at least one dimension");

    public:
        /// \brief A view of `data` with these extents.
        /// \param data    At least the product of the extents elements.
        /// \param extents The extents.
        constexpr md_span(T* data, std::array<std::size_t, Rank> extents) noexcept : data_(data), extents_(extents)
        {
            std::size_t stride = 1;
            for (std::size_t d = Rank; d-- > 0;) {
                strides_[d] = stride;
                stride *= extents_[d];
            }
            size_ = stride;
        }

        /// \brief The element at these indices, unchecked.
        /// \tparam Is Integral indices, one per dimension.
        /// \param is The indices.
        /// \return It.
        template <typename... Is>
            requires(sizeof...(Is) == Rank)
        [[nodiscard]] constexpr T& operator()(Is... is) const noexcept
        {
            std::size_t offset = 0, d = 0;
            ((offset += static_cast<std::size_t>(is) * strides_[d++]), ...);
            return data_[offset];
        }

        /// \brief The extent of dimension `d`.
        /// \param d The dimension.
        /// \return It.
        [[nodiscard]] constexpr std::size_t extent(std::size_t d) const noexcept { return extents_[d]; }
        /// \brief The number of elements.
        /// \return It.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
        /// \brief The storage.
        /// \return The first element.
        [[nodiscard]] constexpr T* data() const noexcept { return data_; }

    private:
        T* data_;
        std::array<std::size_t, Rank> extents_;
        std::array<std::size_t, Rank> strides_{};
        std::size_t size_ = 0;
    };

} // namespace snicholls::containers

#endif /* snicholls_core_containers_md_array_hpp */
