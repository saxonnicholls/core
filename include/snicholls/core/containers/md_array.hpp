//
//  md_array.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `containers::md_array<T, E...>` and `md_span<T, Rank, Layout>`: multi-dimensional storage and views, zero-overhead indexing.
/// \ingroup core
///
/// **Overview.** A 3-, 4- or 5-dimensional table (scenario x tenor x strike, currency x book x day) is one
/// contiguous block indexed by a row-major offset. With the extents in the type, `md_array<T, E...>` holds the
/// block in place and its strides are constants, so `a(i, j, k)` compiles to the arithmetic a hand-written
/// `a[(i * J + j) * K + k]` does; with the extents known only at run time, `md_span<T, Rank, Layout>` is a view
/// over storage someone else owns. `for_each_index` visits every index in row-major order, unrolled at compile
/// time for an `md_array` through `constexpr_nest`.
///
/// \snippet containers/md_array_demo.cpp md_array
///
/// A view's layout says how indices map to an offset, as `std::mdspan`'s does:
///
/// | Layout | Offset of `(i0, ..., iR-1)` | Typical storage |
/// |---|---|---|
/// | `layout_right` (the default) | row-major: the last index is contiguous | C arrays, NumPy's default |
/// | `layout_left` | column-major: the first index is contiguous | Fortran, BLAS / LAPACK, Eigen's default |
/// | `layout_stride` | `sum(i_d * stride(d))`, any strides | a slice, a transpose, padded rows, a batch with gaps |
///
/// \snippet containers/md_array_demo.cpp layouts
///
/// `submdspan(s, slices...)` takes a sub-view: an index fixes a dimension (the rank drops by one), `full_extent`
/// keeps it, and a `std::pair{first, last}` keeps `[first, last)`. The result is a `layout_stride` view of the same
/// storage, so one system of a batch, a row or a column is a view, never a copy.
///
/// \snippet containers/md_array_demo.cpp submdspan
///
/// **Relationships.** `constexpr_for.hpp`'s `constexpr_nest`; C++23's `std::mdspan`, whose spellings `md_span`
/// keeps wherever it costs nothing (`extent`, `stride`, `rank`, `size`, `empty`, `data_handle`, `mapping`,
/// `extents`, `is_exhaustive`, `layout_right` / `layout_left` / `layout_stride`, `full_extent`, `submdspan`), so
/// moving to the standard type is a rename. `linalg/` takes its matrices and batches as `md_span`s.
/// Group: \ref core.
///
/// **Guarantees.** `constexpr`; `md_array` is `sizeof(T) * E0 * E1 * ...` and trivially copyable when `T` is;
/// indexing is unchecked (`md_array::at` checks); an `md_span` is a pointer plus `Rank` extents and `Rank` strides,
/// trivially copyable, and never owns. Views convert implicitly to a view of `const` elements and from
/// `layout_right` or `layout_left` to `layout_stride`, as `std::mdspan`'s do.
///
/// **Errors.** `md_array::at` throws `std::out_of_range`; nothing else throws.
///
/// **Performance.** `benchmarks/containers/md_array_benchmark.cpp`: a 16 x 16 x 16 x 16 grid summed in order,
/// Release, Apple clang 17, Xeon W-3245, 2026-11-07: 65.3 us through `a(i, j, k, l)`, 65.0 us through the
/// offsets written by hand. The same file times an `md_span` in each layout against its offsets written by hand
/// (see BENCHMARKS.md).
///
/// **Tests and demos.** `tests/containers/md_array_test.cpp`, `demos/containers/md_array_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<concepts>`, `<cstddef>`, `<span>`, `<stdexcept>`, `<tuple>`, `<type_traits>`,
/// `<utility>`, `constexpr_for.hpp`.
///
/// **Known issues.** Differences from `std::mdspan`, each chosen to stay small: extents are always run-time values
/// (a mapping is `mapping<Rank>` where the standard's is `mapping<Extents>`), `extents()` returns a
/// `std::array`, elements are read with `operator()` (C++20 has no multi-argument `operator[]`), there is no
/// accessor policy, and `submdspan` always returns a `layout_stride` view and takes no `strided_slice`.
///
/// **Status.** New in 0.5.0; layouts and `submdspan` new in 0.6.0.

#ifndef snicholls_core_containers_md_array_hpp
#define snicholls_core_containers_md_array_hpp

#include <snicholls/core/constexpr_for.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
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

    namespace detail {
        // The state every mapping shares: extents and strides, both run-time values.
        template <std::size_t Rank>
        struct strided_extents {
            std::array<std::size_t, Rank> extents{};
            std::array<std::size_t, Rank> strides{};

            [[nodiscard]] constexpr std::size_t size() const noexcept
            {
                std::size_t n = 1;
                for (std::size_t d = 0; d < Rank; ++d)
                    n *= extents[d];
                return n;
            }
            [[nodiscard]] constexpr std::size_t required_span_size() const noexcept
            {
                if (size() == 0)
                    return 0;
                std::size_t last = 0;
                for (std::size_t d = 0; d < Rank; ++d)
                    last += (extents[d] - 1) * strides[d];
                return last + 1;
            }
        };
    } // namespace detail

    /// \brief Row-major layout, `std::layout_right`: the last index is contiguous.
    /// \ingroup core
    struct layout_right {
        /// \brief The index-to-offset map of a `Rank`-dimensional row-major view.
        /// \tparam Rank The number of dimensions.
        template <std::size_t Rank>
        class mapping {
        public:
            /// \brief The layout this mapping belongs to.
            using layout_type = layout_right;

            /// \brief An empty mapping: every extent 0.
            constexpr mapping() noexcept = default;

            /// \brief The row-major mapping of these extents.
            /// \param extents The extents.
            constexpr mapping(const std::array<std::size_t, Rank>& extents) noexcept   // NOLINT: implicit, as std's
            {
                s_.extents = extents;
                std::size_t stride = 1;
                for (std::size_t d = Rank; d-- > 0;) {
                    s_.strides[d] = stride;
                    stride *= extents[d];
                }
            }

            /// \brief The offset of these indices; the last index is added without a multiplication.
            /// \tparam Is Integral indices, one per dimension.
            /// \param is The indices.
            /// \return The offset.
            template <typename... Is>
                requires(sizeof...(Is) == Rank)
            [[nodiscard]] constexpr std::size_t operator()(Is... is) const noexcept
            {
                const std::array<std::size_t, Rank> i{static_cast<std::size_t>(is)...};
                std::size_t offset = i[Rank - 1];
                for (std::size_t d = 0; d + 1 < Rank; ++d)
                    offset += i[d] * s_.strides[d];
                return offset;
            }

            /// \brief The extents.
            /// \return They.
            [[nodiscard]] constexpr const std::array<std::size_t, Rank>& extents() const noexcept { return s_.extents; }
            /// \brief The stride of dimension `d`.
            /// \param d The dimension.
            /// \return It.
            [[nodiscard]] constexpr std::size_t stride(std::size_t d) const noexcept { return s_.strides[d]; }
            /// \brief One past the largest offset.
            /// \return The product of the extents.
            [[nodiscard]] constexpr std::size_t required_span_size() const noexcept { return s_.size(); }
            /// \brief Whether no two indices share an offset: always.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_unique() noexcept { return true; }
            /// \brief Whether every offset below `required_span_size()` is used: always.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_exhaustive() noexcept { return true; }
            /// \brief Whether the offsets are a sum of index times stride: always.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_strided() noexcept { return true; }

            /// \brief Equal when the extents are.
            friend constexpr bool operator==(const mapping& a, const mapping& b) noexcept { return a.s_.extents == b.s_.extents; }

        private:
            detail::strided_extents<Rank> s_{};
        };
    };

    /// \brief Column-major layout, `std::layout_left`: the first index is contiguous.
    /// \ingroup core
    struct layout_left {
        /// \brief The index-to-offset map of a `Rank`-dimensional column-major view.
        /// \tparam Rank The number of dimensions.
        template <std::size_t Rank>
        class mapping {
        public:
            /// \brief The layout this mapping belongs to.
            using layout_type = layout_left;

            /// \brief An empty mapping: every extent 0.
            constexpr mapping() noexcept = default;

            /// \brief The column-major mapping of these extents.
            /// \param extents The extents.
            constexpr mapping(const std::array<std::size_t, Rank>& extents) noexcept   // NOLINT: implicit, as std's
            {
                s_.extents = extents;
                std::size_t stride = 1;
                for (std::size_t d = 0; d < Rank; ++d) {
                    s_.strides[d] = stride;
                    stride *= extents[d];
                }
            }

            /// \brief The offset of these indices; the first index is added without a multiplication.
            /// \tparam Is Integral indices, one per dimension.
            /// \param is The indices.
            /// \return The offset.
            template <typename... Is>
                requires(sizeof...(Is) == Rank)
            [[nodiscard]] constexpr std::size_t operator()(Is... is) const noexcept
            {
                const std::array<std::size_t, Rank> i{static_cast<std::size_t>(is)...};
                std::size_t offset = i[0];
                for (std::size_t d = 1; d < Rank; ++d)
                    offset += i[d] * s_.strides[d];
                return offset;
            }

            /// \brief The extents.
            /// \return They.
            [[nodiscard]] constexpr const std::array<std::size_t, Rank>& extents() const noexcept { return s_.extents; }
            /// \brief The stride of dimension `d`.
            /// \param d The dimension.
            /// \return It.
            [[nodiscard]] constexpr std::size_t stride(std::size_t d) const noexcept { return s_.strides[d]; }
            /// \brief One past the largest offset.
            /// \return The product of the extents.
            [[nodiscard]] constexpr std::size_t required_span_size() const noexcept { return s_.size(); }
            /// \brief Whether no two indices share an offset: always.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_unique() noexcept { return true; }
            /// \brief Whether every offset below `required_span_size()` is used: always.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_exhaustive() noexcept { return true; }
            /// \brief Whether the offsets are a sum of index times stride: always.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_strided() noexcept { return true; }

            /// \brief Equal when the extents are.
            friend constexpr bool operator==(const mapping& a, const mapping& b) noexcept { return a.s_.extents == b.s_.extents; }

        private:
            detail::strided_extents<Rank> s_{};
        };
    };

    /// \brief Any strides, `std::layout_stride`: a slice, a transpose, padded rows, a batch with gaps between systems.
    /// \ingroup core
    struct layout_stride {
        /// \brief The index-to-offset map `sum(i_d * stride(d))`.
        /// \tparam Rank The number of dimensions.
        template <std::size_t Rank>
        class mapping {
        public:
            /// \brief The layout this mapping belongs to.
            using layout_type = layout_stride;

            /// \brief An empty mapping: every extent 0.
            constexpr mapping() noexcept = default;

            /// \brief The mapping with these extents and strides.
            /// \param extents The extents.
            /// \param strides The strides, in elements.
            constexpr mapping(const std::array<std::size_t, Rank>& extents, const std::array<std::size_t, Rank>& strides) noexcept
            {
                s_.extents = extents;
                s_.strides = strides;
            }

            /// \brief The same offsets as a row-major or column-major mapping, as `std::layout_stride::mapping` converts.
            /// \tparam Other `layout_right::mapping<Rank>` or `layout_left::mapping<Rank>`.
            /// \param other The mapping.
            template <typename Other>
                requires(std::same_as<Other, layout_right::mapping<Rank>> || std::same_as<Other, layout_left::mapping<Rank>>)
            constexpr mapping(const Other& other) noexcept   // NOLINT: implicit, as std's
            {
                s_.extents = other.extents();
                for (std::size_t d = 0; d < Rank; ++d)
                    s_.strides[d] = other.stride(d);
            }

            /// \brief The offset of these indices.
            /// \tparam Is Integral indices, one per dimension.
            /// \param is The indices.
            /// \return The offset.
            template <typename... Is>
                requires(sizeof...(Is) == Rank)
            [[nodiscard]] constexpr std::size_t operator()(Is... is) const noexcept
            {
                const std::array<std::size_t, Rank> i{static_cast<std::size_t>(is)...};
                std::size_t offset = 0;
                for (std::size_t d = 0; d < Rank; ++d)
                    offset += i[d] * s_.strides[d];
                return offset;
            }

            /// \brief The extents.
            /// \return They.
            [[nodiscard]] constexpr const std::array<std::size_t, Rank>& extents() const noexcept { return s_.extents; }
            /// \brief The strides.
            /// \return They.
            [[nodiscard]] constexpr const std::array<std::size_t, Rank>& strides() const noexcept { return s_.strides; }
            /// \brief The stride of dimension `d`.
            /// \param d The dimension.
            /// \return It.
            [[nodiscard]] constexpr std::size_t stride(std::size_t d) const noexcept { return s_.strides[d]; }
            /// \brief One past the largest offset: `1 + sum((extent(d) - 1) * stride(d))`, or 0 when a extent is 0.
            /// \return It.
            [[nodiscard]] constexpr std::size_t required_span_size() const noexcept { return s_.required_span_size(); }
            /// \brief Whether no two indices share an offset. Assumed, as the standard's default does: a
            ///        mapping that repeats offsets (a zero stride) is the caller's choice.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_unique() noexcept { return true; }
            /// \brief Whether every offset below `required_span_size()` is used: true when there are no gaps.
            /// \return Whether `required_span_size()` equals the number of elements.
            [[nodiscard]] constexpr bool is_exhaustive() const noexcept { return s_.required_span_size() == s_.size(); }
            /// \brief Whether the offsets are a sum of index times stride: always.
            /// \return `true`.
            [[nodiscard]] static constexpr bool is_strided() noexcept { return true; }

            /// \brief Equal when the extents and strides are.
            friend constexpr bool operator==(const mapping& a, const mapping& b) noexcept
            {
                return a.s_.extents == b.s_.extents && a.s_.strides == b.s_.strides;
            }

        private:
            detail::strided_extents<Rank> s_{};
        };
    };

    /// \brief The tag that keeps a whole dimension in `submdspan`, `std::full_extent_t`.
    /// \ingroup core
    struct full_extent_t {
        /// \brief Construct the tag.
        explicit full_extent_t() = default;
    };
    /// \brief The `full_extent_t` value, `std::full_extent`.
    /// \ingroup core
    inline constexpr full_extent_t full_extent{};

    /// \brief A view of storage as a `Rank`-dimensional array whose extents are known at run time; `std::mdspan` in miniature.
    /// \ingroup core
    /// \tparam T      The element type (const for a read-only view).
    /// \tparam Rank   The number of dimensions.
    /// \tparam Layout `layout_right` (row-major, the default), `layout_left` (column-major) or `layout_stride`.
    template <typename T, std::size_t Rank, typename Layout = layout_right>
    class md_span {
        static_assert(Rank >= 1, "snicholls::containers::md_span: at least one dimension");

    public:
        using element_type = T;                                          ///< The element type, const for a read-only view.
        using value_type = std::remove_cv_t<T>;                          ///< The element type without const.
        using index_type = std::size_t;                                  ///< The type of an index.
        using size_type = std::size_t;                                   ///< The type of a size.
        using layout_type = Layout;                                      ///< The layout.
        using mapping_type = typename Layout::template mapping<Rank>;    ///< The layout's mapping for this rank.
        using data_handle_type = T*;                                     ///< The storage handle: a pointer.
        using reference = T&;                                            ///< The type of an element access.

        /// \brief An empty view: no storage, every extent 0.
        constexpr md_span() noexcept = default;

        /// \brief A row- or column-major view of `data` with these extents.
        /// \param data    At least the product of the extents elements.
        /// \param extents The extents.
        constexpr md_span(T* data, const std::array<std::size_t, Rank>& extents) noexcept
            requires std::constructible_from<mapping_type, const std::array<std::size_t, Rank>&>
            : data_(data), map_(extents)
        {
        }

        /// \brief A strided view of `data`.
        /// \param data    At least `required_span_size()` elements.
        /// \param extents The extents.
        /// \param strides The strides, in elements.
        constexpr md_span(T* data, const std::array<std::size_t, Rank>& extents, const std::array<std::size_t, Rank>& strides) noexcept
            requires std::same_as<Layout, layout_stride>
            : data_(data), map_(extents, strides)
        {
        }

        /// \brief A view of `data` through this mapping.
        /// \param data    At least `m.required_span_size()` elements.
        /// \param m       The mapping.
        constexpr md_span(T* data, const mapping_type& m) noexcept : data_(data), map_(m) {}

        /// \brief The same elements, as a view of const elements or in `layout_stride`, as `std::mdspan` converts.
        /// \tparam U       The other view's element type: `T`, or `T` without const.
        /// \tparam Other   The other view's layout; its mapping converts to this one's.
        /// \param other The view.
        template <typename U, typename Other>
            requires(!std::same_as<md_span<U, Rank, Other>, md_span> && std::is_convertible_v<U (*)[], T (*)[]> &&
                     std::is_convertible_v<const typename Other::template mapping<Rank>&, mapping_type>)
        constexpr md_span(const md_span<U, Rank, Other>& other) noexcept   // NOLINT: implicit, as std's
            : data_(other.data_handle()), map_(other.mapping())
        {
        }

        /// \brief The element at these indices, unchecked.
        /// \tparam Is Integral indices, one per dimension.
        /// \param is The indices.
        /// \return It.
        template <typename... Is>
            requires(sizeof...(Is) == Rank)
        [[nodiscard]] constexpr T& operator()(Is... is) const noexcept
        {
            return data_[map_(is...)];
        }

        /// \brief The number of dimensions.
        /// \return `Rank`.
        [[nodiscard]] static constexpr std::size_t rank() noexcept { return Rank; }
        /// \brief The extent of dimension `d`.
        /// \param d The dimension.
        /// \return It.
        [[nodiscard]] constexpr std::size_t extent(std::size_t d) const noexcept { return map_.extents()[d]; }
        /// \brief The extents.
        /// \return They.
        [[nodiscard]] constexpr const std::array<std::size_t, Rank>& extents() const noexcept { return map_.extents(); }
        /// \brief The stride of dimension `d`, in elements.
        /// \param d The dimension.
        /// \return It.
        [[nodiscard]] constexpr std::size_t stride(std::size_t d) const noexcept { return map_.stride(d); }
        /// \brief The number of elements.
        /// \return The product of the extents.
        [[nodiscard]] constexpr std::size_t size() const noexcept
        {
            std::size_t n = 1;
            for (std::size_t d = 0; d < Rank; ++d)
                n *= extent(d);
            return n;
        }
        /// \brief Whether the view has no elements.
        /// \return Whether an extent is 0.
        [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
        /// \brief The storage.
        /// \return The element at offset 0.
        [[nodiscard]] constexpr T* data() const noexcept { return data_; }
        /// \brief The storage, `std::mdspan`'s spelling.
        /// \return The element at offset 0.
        [[nodiscard]] constexpr T* data_handle() const noexcept { return data_; }
        /// \brief The mapping.
        /// \return It.
        [[nodiscard]] constexpr const mapping_type& mapping() const noexcept { return map_; }
        /// \brief Whether the view uses every offset below `mapping().required_span_size()`.
        /// \return It.
        [[nodiscard]] constexpr bool is_exhaustive() const noexcept { return map_.is_exhaustive(); }

    private:
        T* data_ = nullptr;
        mapping_type map_{};
    };

    namespace detail {
        // A range slice: a std::pair of any two integral types, {first, last}.
        template <typename S>
        inline constexpr bool is_index_pair = false;
        template <std::integral I, std::integral J>
        inline constexpr bool is_index_pair<std::pair<I, J>> = true;

        template <typename S>
        inline constexpr bool keeps_dimension = std::same_as<S, full_extent_t> || is_index_pair<S>;

        // A slice's first index and its extent, by overload: an index, the whole dimension, or [first, last).
        constexpr std::size_t slice_first(std::size_t i) noexcept { return i; }
        constexpr std::size_t slice_first(full_extent_t) noexcept { return 0; }
        constexpr std::size_t slice_first(const std::pair<std::size_t, std::size_t>& r) noexcept { return r.first; }
        constexpr std::size_t slice_extent(full_extent_t, std::size_t extent) noexcept { return extent; }
        constexpr std::size_t slice_extent(const std::pair<std::size_t, std::size_t>& r, std::size_t) noexcept { return r.second - r.first; }

        template <typename S>
        using slice_t = std::conditional_t<std::same_as<std::remove_cvref_t<S>, full_extent_t>, full_extent_t,
                                           std::conditional_t<is_index_pair<std::remove_cvref_t<S>>, std::pair<std::size_t, std::size_t>, std::size_t>>;
    } // namespace detail

    /// \brief A sub-view, `std::submdspan`: one slice per dimension, always a `layout_stride` view of the same storage.
    /// \ingroup core
    ///
    /// Each slice is an integral index (the dimension is dropped), `full_extent` (it is kept whole) or a
    /// `std::pair{first, last}` of any integral types (it is kept as `[first, last)`). At least one dimension
    /// must be kept. Unchecked, like indexing: every index and range must lie inside its extent.
    /// \tparam T       The element type.
    /// \tparam Rank    The view's rank.
    /// \tparam Layout  The view's layout.
    /// \tparam Slices  One slice type per dimension.
    /// \param s      The view.
    /// \param slices The slices.
    /// \return The sub-view; its rank is the number of slices that are not indices.
    template <typename T, std::size_t Rank, typename Layout, typename... Slices>
        requires(sizeof...(Slices) == Rank &&
                 ((std::integral<std::remove_cvref_t<Slices>> || detail::keeps_dimension<std::remove_cvref_t<Slices>>) && ...) &&
                 (detail::keeps_dimension<std::remove_cvref_t<Slices>> || ...))
    [[nodiscard]] constexpr auto submdspan(const md_span<T, Rank, Layout>& s, Slices... slices) noexcept
    {
        constexpr std::size_t kept = (std::size_t{detail::keeps_dimension<std::remove_cvref_t<Slices>>} + ...);
        const std::tuple<detail::slice_t<Slices>...> all{static_cast<detail::slice_t<Slices>>(slices)...};
        std::array<std::size_t, kept> extents{};
        std::array<std::size_t, kept> strides{};
        std::size_t offset = 0;
        std::size_t k = 0;
        constexpr_for<0, Rank, 1>([&](auto d) {
            using S = std::tuple_element_t<d, std::tuple<detail::slice_t<Slices>...>>;
            const auto& slice = std::get<d>(all);
            offset += detail::slice_first(slice) * s.stride(d);
            if constexpr (detail::keeps_dimension<S>) {
                extents[k] = detail::slice_extent(slice, s.extent(d));
                strides[k] = s.stride(d);
                ++k;
            }
        });
        return md_span<T, kept, layout_stride>(s.data_handle() + offset, extents, strides);
    }

} // namespace snicholls::containers

#endif /* snicholls_core_containers_md_array_hpp */
