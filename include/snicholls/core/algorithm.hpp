//
//  algorithm.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fill`, `fill_n`, `copy`, `copy_n`, `copy_backward`, `iota`, `equal`, `reverse` and
/// `swap_ranges`: the elementary algorithms as plain loops, usable in constant evaluation from C++17.
/// \ingroup core
///
/// **Overview.** The handful of algorithms that build and compare fixed tables, as loops anyone can read in a minute:
/// fill a table, count it up, copy one into another, reverse it, swap two, compare them. Each takes iterators as the
/// standard's do and returns what the standard's returns; `fill`, `iota` and `equal` also take whole ranges (an
/// `std::array`, a C array, a span, anything with `begin` and `end`).
///
/// \snippet algorithm_demo.cpp table
///
/// Use them where `<algorithm>` is not what you want:
/// - **C++17 code that runs at compile time.** `std::fill`, `std::copy`, `std::iota`, `std::equal` and the others
///   are `constexpr` only from C++20; these are `constexpr` in C++17, so ts-moveables (C++17) builds tables with
///   them. The header is compiled as C++17 by `snicholls_core_cpp17_check`.
/// - **Device code.** A pure `constexpr` loop with no allocation and no library internals behind it is callable from
///   a CUDA kernel under `--expt-relaxed-constexpr` and synthesises for an FPGA (bounded, no recursion).
/// - **Clearing a buffer without `memset`.** GCC 13 and 15 report false `-Wstringop-overflow` on `memset` into a
///   buffer whose size they misjudge; `snicholls::fill` is a loop, which the optimiser may still turn into a
///   `memset` where that is right.
///
/// In C++20 host code the standard's algorithms are as good, and at run time `std::copy` is faster (see Performance).
///
/// **Relationships.** `constexpr_for` for loops whose index must be a constant expression; `iterate.hpp` and
/// `table<N>(f)` for a table computed from its index; `zip` for walking ranges together.
///
/// **Guarantees.**
/// - Every function is `constexpr`, and `noexcept` exactly when the iterator steps, compares and dereferences and the
///   element assignments it performs are (`std::string` elements may throw; `int` elements never do).
/// - Complexity is linear: one assignment (or compare) per element, `reverse` and `swap_ranges` three moves per swap.
/// - Nothing allocates and nothing is owned: the caller's iterators are written through, never stored.
/// - `copy` and `copy_n` write forwards, so the output may overlap the input only if it starts before it;
///   `copy_backward` writes backwards, for an output that starts after the input. As in the standard.
/// - `equal` over two ranges compares their lengths first, so ranges of different lengths are unequal (the
///   three-iterator form, as the standard's, assumes the second range is long enough).
///
/// **Errors.** None reported: a precondition broken (an output too short, `last` before `first`) is undefined
/// behaviour, as with the standard's algorithms, and a compile error in constant evaluation.
///
/// **Performance.** `benchmarks/algorithm_benchmark.cpp`, 4,096 doubles, Release `-O3`, Apple clang 17 on a Xeon
/// W-3245, 2026-11-27: `fill` 526 ns, `std::fill` 540 ns, the hand-written loop 544 ns; `copy` 804 ns, the
/// hand-written loop 801 ns, `std::copy` 563 ns. `fill` is the loop and as fast as anything. `copy` is the loop too,
/// and the standard library turns `std::copy` of trivially copyable elements into `memmove`, about 1.4 times faster:
/// for a large copy at run time in C++20 host code, use `std::copy`.
///
/// **Tests and demos.** `tests/algorithm_test.cpp`, `demos/algorithm_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<iterator>` (`std::begin`, `std::end`), `<utility>` (`std::move`,
/// `std::declval`).
///
/// **Known issues.** `reverse` and `swap_ranges` exchange elements by three moves, not by an ADL `swap`: a type
/// whose `swap` does something a move does not is exchanged by its moves. Proxy iterators (`std::vector<bool>`) are
/// not supported. The names are the standard's: call them qualified (`snicholls::fill`), as an unqualified call on
/// standard iterators would also find `std::fill` by ADL and be ambiguous.
///
/// **Status.** Stable.

#ifndef snicholls_core_algorithm_hpp
#define snicholls_core_algorithm_hpp

#include <cstddef>
#include <iterator>
#include <utility>

namespace snicholls {

namespace detail {
    // Whether walking It (step, compare, dereference) can throw: part of every function's noexcept below.
    template <class It>
    inline constexpr bool walk_noexcept_v = noexcept(++std::declval<It&>()) &&
                                            noexcept(std::declval<const It&>() != std::declval<const It&>()) &&
                                            noexcept(*std::declval<It&>());

    template <class A, class B>
    inline constexpr bool exchange_noexcept_v =
        walk_noexcept_v<A> && walk_noexcept_v<B> &&
        noexcept(*std::declval<A&>() = std::move(*std::declval<B&>())) &&
        noexcept(*std::declval<B&>() = std::move(*std::declval<A&>()));

    // Exchanges two elements by three moves: std::swap is constexpr only from C++20.
    template <class A, class B>
    constexpr void exchange_elements(A a, B b) noexcept(exchange_noexcept_v<A, B>)
    {
        auto held = std::move(*a);
        *a = std::move(*b);
        *b = std::move(held);
    }
} // namespace detail

/// \brief Assigns `value` to every element of `[first, last)`.
/// \tparam It A forward iterator whose elements are assignable from `T`.
/// \tparam T The value's type.
/// \param first The first element.
/// \param last One past the last element.
/// \param value The value assigned.
template <class It, class T>
constexpr void fill(It first, It last, const T& value) noexcept(detail::walk_noexcept_v<It> &&
                                                                noexcept(*std::declval<It&>() = value))
{
    for (; first != last; ++first)
        *first = value;
}

/// \brief Assigns `value` to every element of `range`.
/// \tparam R A range (`std::begin` and `std::end` apply) whose elements are assignable from `T`.
/// \tparam T The value's type.
/// \param range The range, written through.
/// \param value The value assigned.
template <class R, class T>
constexpr void fill(R&& range, const T& value) noexcept(noexcept(snicholls::fill(std::begin(range), std::end(range), value)))
{
    snicholls::fill(std::begin(range), std::end(range), value);
}

/// \brief Assigns `value` to the `n` elements from `first`.
/// \tparam It An output iterator whose elements are assignable from `T`.
/// \tparam T The value's type.
/// \param first The first element.
/// \param n How many elements; none if `n` is 0.
/// \param value The value assigned.
/// \return One past the last element written.
template <class It, class T>
constexpr It fill_n(It first, std::size_t n, const T& value) noexcept(detail::walk_noexcept_v<It> &&
                                                                     noexcept(*std::declval<It&>() = value))
{
    for (; n > 0; --n, ++first)
        *first = value;
    return first;
}

/// \brief Copies `[first, last)` to the elements from `out`, first to last.
/// \tparam In An input iterator.
/// \tparam Out An output iterator whose elements are assignable from `*In`.
/// \param first The first element copied.
/// \param last One past the last element copied.
/// \param out The first element written; it must not lie in `(first, last)`.
/// \return One past the last element written.
template <class In, class Out>
constexpr Out copy(In first, In last, Out out) noexcept(detail::walk_noexcept_v<In> && detail::walk_noexcept_v<Out> &&
                                                        noexcept(*std::declval<Out&>() = *std::declval<In&>()))
{
    for (; first != last; ++first, ++out)
        *out = *first;
    return out;
}

/// \brief Copies the `n` elements from `first` to the elements from `out`.
/// \tparam In An input iterator.
/// \tparam Out An output iterator whose elements are assignable from `*In`.
/// \param first The first element copied.
/// \param n How many elements.
/// \param out The first element written.
/// \return One past the last element written.
template <class In, class Out>
constexpr Out copy_n(In first, std::size_t n, Out out) noexcept(detail::walk_noexcept_v<In> &&
                                                                detail::walk_noexcept_v<Out> &&
                                                                noexcept(*std::declval<Out&>() = *std::declval<In&>()))
{
    for (; n > 0; --n, ++first, ++out)
        *out = *first;
    return out;
}

/// \brief Copies `[first, last)` to the elements ending at `out_last`, last to first, so the output may overlap the
/// input's end.
/// \tparam In A bidirectional iterator.
/// \tparam Out A bidirectional iterator whose elements are assignable from `*In`.
/// \param first The first element copied.
/// \param last One past the last element copied.
/// \param out_last One past the last element written; it must not lie in `(first, last]`.
/// \return The first element written.
template <class In, class Out>
constexpr Out copy_backward(In first, In last, Out out_last) noexcept(
    detail::walk_noexcept_v<In> && detail::walk_noexcept_v<Out> && noexcept(--std::declval<In&>()) &&
    noexcept(--std::declval<Out&>()) && noexcept(*std::declval<Out&>() = *std::declval<In&>()))
{
    while (first != last)
        *--out_last = *--last;
    return out_last;
}

/// \brief Assigns `value`, `value + 1`, `value + 2`, ... to the elements of `[first, last)`.
/// \tparam It A forward iterator whose elements are assignable from `T`.
/// \tparam T A type with prefix `++`.
/// \param first The first element.
/// \param last One past the last element.
/// \param value The first value.
template <class It, class T>
constexpr void iota(It first, It last, T value) noexcept(detail::walk_noexcept_v<It> &&
                                                         noexcept(*std::declval<It&>() = value) && noexcept(++value))
{
    for (; first != last; ++first, ++value)
        *first = value;
}

/// \brief Assigns `value`, `value + 1`, `value + 2`, ... to the elements of `range`.
/// \tparam R A range whose elements are assignable from `T`.
/// \tparam T A type with prefix `++`.
/// \param range The range, written through.
/// \param value The first value.
template <class R, class T>
constexpr void iota(R&& range, T value) noexcept(noexcept(snicholls::iota(std::begin(range), std::end(range), value)))
{
    snicholls::iota(std::begin(range), std::end(range), value);
}

/// \brief Whether `[first1, last1)` equals the elements from `first2`, element by element with `==`.
/// \tparam A An input iterator.
/// \tparam B An input iterator whose elements compare with `*A` by `==`.
/// \param first1 The first range's first element.
/// \param last1 One past the first range's last element.
/// \param first2 The second range's first element; the second range must be at least as long as the first.
/// \return `true` if every pair compares equal, and for an empty first range.
template <class A, class B>
[[nodiscard]] constexpr bool equal(A first1, A last1, B first2) noexcept(
    detail::walk_noexcept_v<A> && detail::walk_noexcept_v<B> &&
    noexcept(static_cast<bool>(*std::declval<A&>() == *std::declval<B&>())))
{
    for (; first1 != last1; ++first1, ++first2)
        if (!static_cast<bool>(*first1 == *first2))
            return false;
    return true;
}

/// \brief Whether two ranges have the same length and equal elements, compared with `==`.
/// \tparam R1 A range with `std::size`.
/// \tparam R2 A range with `std::size` whose elements compare with `R1`'s by `==`.
/// \param a The first range.
/// \param b The second range.
/// \return `true` if the lengths match and every pair compares equal.
template <class R1, class R2>
[[nodiscard]] constexpr bool equal(const R1& a, const R2& b) noexcept(
    noexcept(std::size(a)) && noexcept(std::size(b)) &&
    noexcept(snicholls::equal(std::begin(a), std::end(a), std::begin(b))))
{
    return std::size(a) == std::size(b) && snicholls::equal(std::begin(a), std::end(a), std::begin(b));
}

/// \brief Reverses the order of the elements of `[first, last)`.
/// \tparam It A bidirectional iterator over move-assignable, move-constructible elements.
/// \param first The first element.
/// \param last One past the last element.
template <class It>
constexpr void reverse(It first, It last) noexcept(detail::exchange_noexcept_v<It, It> &&
                                                   noexcept(--std::declval<It&>()) &&
                                                   noexcept(std::declval<const It&>() == std::declval<const It&>()))
{
    while (first != last && first != --last) {
        detail::exchange_elements(first, last);
        ++first;
    }
}

/// \brief Exchanges each element of `[first1, last1)` with the element at the same place from `first2`.
/// \tparam A A forward iterator.
/// \tparam B A forward iterator whose elements exchange with `*A` by moves.
/// \param first1 The first range's first element.
/// \param last1 One past the first range's last element.
/// \param first2 The second range's first element; the ranges must not overlap.
/// \return One past the last element of the second range exchanged.
template <class A, class B>
constexpr B swap_ranges(A first1, A last1, B first2) noexcept(detail::exchange_noexcept_v<A, B>)
{
    for (; first1 != last1; ++first1, ++first2)
        detail::exchange_elements(first1, first2);
    return first2;
}

} // namespace snicholls

#endif // snicholls_core_algorithm_hpp
