//
//  rust_iter.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Rust's iterator and slice methods that C++20 and core lack: `windows`, `chunks`, `chunks_exact`,
///        `dedup` (`fn::rust::slice`), and `enumerate`, `step_by`, `find_map`, `max_by_key`, `min_by_key`, `max_by`,
///        `min_by`, `nth`, `last`, `try_for_each` (`fn::rust::iter`). All `constexpr` and allocation-free.
/// \ingroup core
///
/// **Overview.** Rust's `Iterator` is mostly in C++ already: `std::views` (C++20), `snicholls::zip` and
/// `functional/` cover it. This header adds what is missing, under Rust's names, the range first:
///
/// | Rust                                        | here                                    | already in core / std                       |
/// | ------------------------------------------- | --------------------------------------- | ------------------------------------------- |
/// | `s.windows(k)`, `s.chunks(k)`, `s.chunks_exact(k)` | `slice::windows(s, k)`, `chunks`, `chunks_exact`, and `windows_of<K>(s)`, `chunks_exact_of<K>(s)` with `std::span<T, K>` | C++23's `views::slide`, `views::chunk` |
/// | `v.dedup()`, `v.dedup_by_key(k)`            | `slice::dedup(a)`, `slice::dedup_by_key(a, k)` on a `std::array` | `std::ranges::unique` in place |
/// | `it.enumerate()`                            | `iter::enumerate(r)`: `(index, element)` | C++23's `views::enumerate`                  |
/// | `it.step_by(n)`                             | `iter::step_by(s, n)`                   | C++23's `views::stride`                     |
/// | `it.find_map(f)`                            | `iter::find_map(r, f)`                  |                                             |
/// | `it.max_by_key(k)`, `min_by_key`, `max_by(c)`, `min_by(c)` | `iter::max_by_key(r, k)`, ...: `std::optional`, Rust's ties | `std::ranges::max` (UB when empty, first of equals) |
/// | `it.nth(n)`, `it.last()`                    | `iter::nth(r, n)`, `iter::last(r)`: `std::optional` |                                     |
/// | `it.try_for_each(f)`                        | `iter::try_for_each(r, f)`              |                                             |
/// | `map`, `filter`, `take`, `skip`, `take_while`, `skip_while`, `rev`, `flat_map`, `chain` |      | `std::views::transform`, `filter`, `take`, `drop`, `take_while`, `drop_while`, `reverse`, `transform \| join`; `fn::concat` |
/// | `zip`, `unzip`, `fold`, `try_fold`, `scan`, `reduce` |                                | `snicholls::zip`, `fn::unzip`, `fn::foldl_of`, `fn::fold_m`, `fn::scanl`, `fn::foldl1` |
/// | `filter_map`, `position`, `all`, `any`, `sum`, `product`, `partition` |               | `fn::map_maybe`, `fn::find_index`, `fn::all`, `fn::any`, `fn::sum`, `fn::product`, `fn::partition` |
/// | `count`, `min`, `max`, `is_sorted_by_key`, `cmp`, `eq` |                              | `std::ranges::count_if`, `min`, `max`, `is_sorted` with a projection, `std::lexicographical_compare_three_way`, `std::ranges::equal` |
/// | `successors`, `repeat`, `cycle`, `from_fn`  |                                         | `fn::stream::iterate`, `repeat`, `cycle`, `unfold` |
/// | `collect::<Result<Vec<_>, _>>()`            |                                         | `fn::sequence`                              |
///
/// \snippet rust_iter_demo.cpp slices
///
/// **Relationships.** `functional/rust.hpp` (Option and cmp), `snicholls::zip`, `functional/list.hpp`,
/// `functional/monad.hpp`, `containers::inplace_vector`, `meta/compare.hpp`. Group: \ref core.
///
/// **Implementations / variants.** The slice views (`slice_view`, `step_view`) are over a contiguous range, as
/// Rust's slices are: a pointer, a length and two sizes, trivially copyable, each element a `std::span` (or a
/// reference for `step_by`) made when it is read. They are `std::ranges::view`s, so they compose with `std::views`.
///
/// **Guarantees.** `constexpr`, allocation-free, `noexcept` for the views. A view refers to the range it was made
/// from and must not outlive it; only borrowed ranges (lvalues, spans) are accepted. `windows(s, k)` has
/// `size - k + 1` windows (none when `k > size`); `chunks(s, k)` has `ceil(size / k)`, the last possibly shorter;
/// `chunks_exact(s, k)` has `size / k`, and `remainder()` is what is left. `max_by_key` and `max_by` return the
/// last of equal maxima and `min_by_key` and `min_by` the first of equal minima, as Rust does. `try_for_each`
/// stops at the first failure.
/// \pre `k > 0` and `n > 0` (checked with `assert`; Rust panics).
///
/// **Errors.** None of their own: an empty range is an empty `std::optional`; `try_for_each` returns the first
/// failure `f` gave.
///
/// **Examples.** \snippet rust_iter_demo.cpp iter
///
/// **Performance.** A window is a pointer and a length made per step: the same loads as indexing by hand.
/// `benchmarks/functional/rust_iter_benchmark.cpp` compares a moving sum over `windows` with the index loop.
///
/// **Tests and demos.** `tests/functional/rust_iter_test.cpp`, `demos/functional/rust_iter_demo.cpp`,
/// `demos/functional/rust_showcase_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<cstddef>`, `<functional>`, `<iterator>`,
/// `<optional>`, `<ranges>`, `<span>`, `<type_traits>`, `<utility>`, `<variant>`, `snicholls/core/result.hpp`,
/// `snicholls/core/zip.hpp`, `containers/inplace_vector.hpp`, `meta/compare.hpp`, `meta/concepts.hpp` and `functional/combinators.hpp`.
///
/// **Known issues and gotchas.** `enumerate` is a `snicholls::zip` of an index and the range: its elements are
/// `std::tuple<std::size_t, T&>`, read with a structured binding. A view's iterator holds a copy of the view, so
/// it stays valid when the view is moved (into a `std::views` pipeline).
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_rust_iter_hpp
#define snicholls_core_functional_rust_iter_hpp

#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/meta/compare.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/zip.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>

namespace snicholls::fn::rust {

    namespace detail {

        // A forward iterator over a view with operator[] and size(), holding a copy of the (small) view.
        template <class V>
        class index_iterator {
        public:
            using value_type = std::remove_cvref_t<decltype(std::declval<const V&>()[0])>;
            using difference_type = std::ptrdiff_t;
            using iterator_concept = std::forward_iterator_tag;

            constexpr index_iterator() noexcept = default;
            constexpr index_iterator(const V& v, std::size_t i) noexcept : v_(v), i_(i) {}

            constexpr decltype(auto) operator*() const noexcept { return v_[i_]; }
            constexpr index_iterator& operator++() noexcept
            {
                ++i_;
                return *this;
            }
            constexpr index_iterator operator++(int) noexcept
            {
                index_iterator t = *this;
                ++i_;
                return t;
            }
            friend constexpr bool operator==(const index_iterator& a, const index_iterator& b) noexcept { return a.i_ == b.i_; }

        private:
            V v_{};
            std::size_t i_ = 0;
        };

        template <class R>
        concept slice_source = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> && std::ranges::borrowed_range<R>;

        template <class R>
        using slice_elem_t = std::remove_reference_t<std::ranges::range_reference_t<R>>;

    } // namespace detail

    /// \brief Consecutive sub-slices of a contiguous range, each a `std::span<T, W>`: what `slice::windows`,
    ///        `slice::chunks` and `slice::chunks_exact` return.
    /// \ingroup core
    /// \tparam T The element type (const when the range is).
    /// \tparam W The span's extent: a compile-time width, or `std::dynamic_extent`.
    template <class T, std::size_t W = std::dynamic_extent>
    class slice_view : public std::ranges::view_interface<slice_view<T, W>> {
    public:
        /// \brief Empty.
        constexpr slice_view() noexcept = default;
        /// \brief `count` slices of `data`, slice `i` starting at `i * step`, at most `width` long, within `size`.
        /// \param data The first element.
        /// \param size The number of elements.
        /// \param width The length of a slice (the last of `chunks` may be shorter).
        /// \param step The distance between the starts of two slices.
        /// \param count The number of slices.
        constexpr slice_view(T* data, std::size_t size, std::size_t width, std::size_t step, std::size_t count) noexcept
            : data_(data), size_(size), width_(width), step_(step), count_(count)
        {
        }

        /// \brief Slice `i`. \pre `i < size()`.
        /// \param i The index.
        /// \return A span over its elements.
        [[nodiscard]] constexpr std::span<T, W> operator[](std::size_t i) const noexcept
        {
            assert(i < count_);
            const std::size_t at = i * step_;
            return std::span<T, W>(data_ + at, std::min(width_, size_ - at));
        }
        /// \brief The number of slices.
        /// \return The count.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return count_; }
        /// \brief The first slice.
        /// \return An iterator to it.
        [[nodiscard]] constexpr detail::index_iterator<slice_view> begin() const noexcept { return {*this, 0}; }
        /// \brief Past the last slice.
        /// \return The end iterator.
        [[nodiscard]] constexpr detail::index_iterator<slice_view> end() const noexcept { return {*this, count_}; }
        /// \brief The elements after the last slice (`chunks_exact`'s remainder; empty for the others).
        /// \return A span over them.
        [[nodiscard]] constexpr std::span<T> remainder() const noexcept
        {
            const std::size_t used = count_ == 0 ? 0 : std::min(size_, (count_ - 1) * step_ + width_);
            return step_ == width_ ? std::span<T>(data_ + used, size_ - used) : std::span<T>();
        }

    private:
        T* data_ = nullptr;
        std::size_t size_ = 0;
        std::size_t width_ = 0;
        std::size_t step_ = 1;
        std::size_t count_ = 0;
    };

    /// \brief Every `n`-th element of a contiguous range, from the first, by reference: what `iter::step_by` returns.
    /// \ingroup core
    /// \tparam T The element type (const when the range is).
    template <class T>
    class step_view : public std::ranges::view_interface<step_view<T>> {
    public:
        /// \brief Empty.
        constexpr step_view() noexcept = default;
        /// \brief Elements `0`, `step`, `2 * step`, ... of the `size` elements at `data`.
        /// \param data The first element.
        /// \param size The number of elements.
        /// \param step The stride, at least 1.
        constexpr step_view(T* data, std::size_t size, std::size_t step) noexcept : data_(data), size_(size), step_(step) {}

        /// \brief Element `i * step`. \pre `i < size()`.
        /// \param i The index.
        /// \return A reference to it.
        [[nodiscard]] constexpr T& operator[](std::size_t i) const noexcept
        {
            assert(i < size());
            return data_[i * step_];
        }
        /// \brief The number of elements visited.
        /// \return The count.
        [[nodiscard]] constexpr std::size_t size() const noexcept { return size_ == 0 ? 0 : (size_ - 1) / step_ + 1; }
        /// \brief The first element.
        /// \return An iterator to it.
        [[nodiscard]] constexpr detail::index_iterator<step_view> begin() const noexcept { return {*this, 0}; }
        /// \brief Past the last element.
        /// \return The end iterator.
        [[nodiscard]] constexpr detail::index_iterator<step_view> end() const noexcept { return {*this, size()}; }

    private:
        T* data_ = nullptr;
        std::size_t size_ = 0;
        std::size_t step_ = 1;
    };

    namespace detail {

        template <std::size_t W = std::dynamic_extent>
        struct windows_fn {
            template <slice_source R, class T = slice_elem_t<R>>
            constexpr slice_view<T, W> operator()(R&& s, std::size_t k) const noexcept
            {
                assert(k > 0);
                const std::size_t n = std::ranges::size(s);
                return {std::ranges::data(s), n, k, 1, n < k ? 0 : n - k + 1};
            }
        };

        template <std::size_t W = std::dynamic_extent>
        struct chunks_exact_fn {
            template <slice_source R, class T = slice_elem_t<R>>
            constexpr slice_view<T, W> operator()(R&& s, std::size_t k) const noexcept
            {
                assert(k > 0);
                return {std::ranges::data(s), std::ranges::size(s), k, k, std::ranges::size(s) / k};
            }
        };

        struct chunks_fn {
            template <slice_source R, class T = slice_elem_t<R>>
            constexpr slice_view<T> operator()(R&& s, std::size_t k) const noexcept
            {
                assert(k > 0);
                const std::size_t n = std::ranges::size(s);
                return {std::ranges::data(s), n, k, k, (n + k - 1) / k};
            }
        };

        struct dedup_by_key_fn {
            template <fn::detail::std_array A, class K, class T = typename std::remove_cvref_t<A>::value_type,
                      std::size_t N = std::tuple_size_v<std::remove_cvref_t<A>>>
            constexpr containers::inplace_vector<T, N> operator()(const A& xs, K&& key) const
            {
                containers::inplace_vector<T, N> out;
                for (std::size_t i = 0; i < N; ++i)
                    if (i == 0 || !(std::invoke(key, xs[i]) == std::invoke(key, xs[i - 1]))) out.push_back(xs[i]);
                return out;
            }
        };

        struct dedup_fn {
            template <fn::detail::std_array A>
            constexpr auto operator()(const A& xs) const
                SNICHOLLS_FN_RETURNS(dedup_by_key_fn{}(xs, std::identity{}))
        };

        struct enumerate_fn {
            template <std::ranges::viewable_range R>
                requires std::ranges::sized_range<R>
            constexpr auto operator()(R&& r) const
            {
                const auto n = static_cast<std::size_t>(std::ranges::size(r));
                return snicholls::zip(std::views::iota(std::size_t{0}, n), std::forward<R>(r));
            }
            template <std::ranges::viewable_range R>
                requires(!std::ranges::sized_range<R>)
            constexpr auto operator()(R&& r) const
            {
                return snicholls::zip(std::views::iota(std::size_t{0}), std::forward<R>(r));
            }
        };

        struct step_by_fn {
            template <slice_source R, class T = slice_elem_t<R>>
            constexpr step_view<T> operator()(R&& s, std::size_t n) const noexcept
            {
                assert(n > 0);
                return {std::ranges::data(s), std::ranges::size(s), n};
            }
        };

        template <class R>
        using range_value_t = std::remove_cvref_t<std::ranges::range_reference_t<R>>;

        struct find_map_fn {
            template <std::ranges::input_range R, class F,
                      class O = std::remove_cvref_t<std::invoke_result_t<F&, std::ranges::range_reference_t<R>>>>
                requires specialisation_of<O, std::optional>
            constexpr O operator()(R&& r, F&& f) const
            {
                for (auto&& x : r)
                    if (O o = std::invoke(f, std::forward<decltype(x)>(x)); o.has_value()) return o;
                return std::nullopt;
            }
        };

        // The best element by a three-way comparison: Last ties go to the later element (Rust's max), else the earlier.
        template <bool Last>
        struct best_by_fn {
            template <std::ranges::input_range R, class C, class V = range_value_t<R>>
            constexpr std::optional<V> operator()(R&& r, C&& cmp) const
            {
                std::optional<V> best;
                for (auto&& x : r) {
                    if (!best.has_value()) {
                        best.emplace(x);
                        continue;
                    }
                    const auto c = std::invoke(cmp, std::as_const(x), std::as_const(*best));
                    if (Last ? c >= 0 : c < 0) *best = x;
                }
                return best;
            }
        };

        template <bool Last>
        struct best_by_key_fn {
            template <std::ranges::input_range R, class K, class V = range_value_t<R>,
                      class Key = std::remove_cvref_t<std::invoke_result_t<K&, const V&>>>
            constexpr std::optional<V> operator()(R&& r, K&& key) const
            {
                std::optional<V> best;
                std::optional<Key> best_key;
                for (auto&& x : r) {
                    Key k = std::invoke(key, std::as_const(x));
                    if (best.has_value()) {
                        const auto c = snicholls::synth_three_way(k, *best_key);
                        if (!(Last ? c >= 0 : c < 0)) continue;
                    }
                    best.emplace(x);
                    best_key.emplace(std::move(k));
                }
                return best;
            }
        };

        struct nth_fn {
            template <std::ranges::input_range R, class V = range_value_t<R>>
            constexpr std::optional<V> operator()(R&& r, std::size_t n) const
            {
                for (auto&& x : r)
                    if (n-- == 0) return std::optional<V>(x);
                return std::nullopt;
            }
        };

        struct last_fn {
            template <std::ranges::input_range R, class V = range_value_t<R>>
            constexpr std::optional<V> operator()(R&& r) const
            {
                std::optional<V> out;
                for (auto&& x : r) out.emplace(x);
                return out;
            }
        };

        // What try_for_each returns when nothing failed, and whether a step succeeded, per carrier.
        template <class E>
        constexpr snicholls::result<void, E> success(std::type_identity<snicholls::result<void, E>>) noexcept
        {
            return snicholls::ok();
        }
        constexpr std::optional<std::monostate> success(std::type_identity<std::optional<std::monostate>>) noexcept
        {
            return std::monostate{};
        }
        template <class T>
        constexpr bool succeeded(const std::optional<T>& o) noexcept { return o.has_value(); }
        template <class V, class E>
        constexpr bool succeeded(const snicholls::result<V, E>& r) noexcept { return r.is_ok(); }

        struct try_for_each_fn {
            template <std::ranges::input_range R, class F,
                      class M = std::remove_cvref_t<std::invoke_result_t<F&, std::ranges::range_reference_t<R>>>>
            constexpr auto operator()(R&& r, F&& f) const -> decltype(detail::success(std::type_identity<M>{}))
            {
                for (auto&& x : r)
                    if (M m = std::invoke(f, std::forward<decltype(x)>(x)); !detail::succeeded(m)) return m;
                return detail::success(std::type_identity<M>{});
            }
        };

    } // namespace detail

    /// \brief Rust's slice methods over contiguous ranges and `std::array`.
    /// \ingroup core
    namespace slice {
        /// \brief `windows(s, k)`: every run of `k` consecutive elements, overlapping, as `std::span`s.
        inline constexpr detail::windows_fn<> windows{};
        /// \brief `chunks(s, k)`: the elements in runs of `k`, the last possibly shorter, as `std::span`s.
        inline constexpr detail::chunks_fn chunks{};
        /// \brief `chunks_exact(s, k)`: the elements in runs of exactly `k`; `remainder()` holds the rest.
        inline constexpr detail::chunks_exact_fn<> chunks_exact{};
        /// \brief `dedup(a)`: a `std::array` without consecutive repeats, in an `inplace_vector`.
        inline constexpr detail::dedup_fn dedup{};
        /// \brief `dedup_by_key(a, key)`: a `std::array` without consecutive elements of equal key.
        inline constexpr detail::dedup_by_key_fn dedup_by_key{};

        /// \brief `windows_of<K>(s)`: `windows(s, K)` with each window a `std::span<T, K>`.
        /// \tparam K The width, at least 1.
        /// \param s A contiguous range.
        /// \return The view of windows.
        template <std::size_t K, detail::slice_source R>
            requires(K > 0)
        [[nodiscard]] constexpr auto windows_of(R&& s) noexcept
        {
            return detail::windows_fn<K>{}(s, K);
        }

        /// \brief `chunks_exact_of<K>(s)`: `chunks_exact(s, K)` with each chunk a `std::span<T, K>`.
        /// \tparam K The width, at least 1.
        /// \param s A contiguous range.
        /// \return The view of chunks.
        template <std::size_t K, detail::slice_source R>
            requires(K > 0)
        [[nodiscard]] constexpr auto chunks_exact_of(R&& s) noexcept
        {
            return detail::chunks_exact_fn<K>{}(s, K);
        }
    } // namespace slice

    /// \brief Rust's iterator methods that C++20 and core lack.
    /// \ingroup core
    namespace iter {
        /// \brief `enumerate(r)`: `(index, element)` tuples, by `snicholls::zip`.
        inline constexpr detail::enumerate_fn enumerate{};
        /// \brief `step_by(s, n)`: elements `0`, `n`, `2n`, ... of a contiguous range, by reference.
        inline constexpr detail::step_by_fn step_by{};
        /// \brief `find_map(r, f)`: the first engaged `f(x)`, or empty.
        inline constexpr detail::find_map_fn find_map{};
        /// \brief `max_by(r, c)`: the greatest element by a three-way comparator, the last of equals; empty if none.
        inline constexpr detail::best_by_fn<true> max_by{};
        /// \brief `min_by(r, c)`: the least element by a three-way comparator, the first of equals; empty if none.
        inline constexpr detail::best_by_fn<false> min_by{};
        /// \brief `max_by_key(r, key)`: the element of greatest key, the last of equals; `key` once per element.
        inline constexpr detail::best_by_key_fn<true> max_by_key{};
        /// \brief `min_by_key(r, key)`: the element of least key, the first of equals; `key` once per element.
        inline constexpr detail::best_by_key_fn<false> min_by_key{};
        /// \brief `nth(r, n)`: element `n` (from 0), or empty if there are not that many.
        inline constexpr detail::nth_fn nth{};
        /// \brief `last(r)`: the last element, or empty.
        inline constexpr detail::last_fn last{};
        /// \brief `try_for_each(r, f)`: `f` of each element until one fails (`f` returns `result<void, E>` or
        ///        `std::optional<std::monostate>`); that failure, or success.
        inline constexpr detail::try_for_each_fn try_for_each{};
    } // namespace iter

} // namespace snicholls::fn::rust

#endif /* snicholls_core_functional_rust_iter_hpp */
