//
//  fold.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fn::foldl`, `foldr`, `scan`, `foldl_of`, `foldr_of`, `unfold` and `iterate_n`: folds over
///        parameter packs and sequences, and unfolds into arrays, all `constexpr`.
/// \ingroup core
///
/// **Overview.**
///
/// | here                          | what it computes                                     | Haskell             |
/// | ----------------------------- | ---------------------------------------------------- | ------------------- |
/// | `foldl(f, z, a, b, c)`        | `f(f(f(z, a), b), c)`                                | `foldl f z [a,b,c]` |
/// | `foldr(f, z, a, b, c)`        | `f(a, f(b, f(c, z)))`                                | `foldr f z [a,b,c]` |
/// | `scan(f, z, a, b, c)`         | `std::array{z, f(z, a), f(f(z, a), b), ...}`         | `scanl f z [a,b,c]` |
/// | `foldl_of(f, z, seq)`         | `foldl` over a range or a tuple                      | `foldl`             |
/// | `foldr_of(f, z, seq)`         | `foldr` over a bidirectional range or a tuple        | `foldr`             |
/// | `unfold<N>(seed, step)`       | `N` values from `step(state) -> {value, next state}` | `unfoldr`           |
/// | `iterate_n(f, n)`             | the function `x -> f(f(...f(x)))`, `n` times         | `(!! n) . iterate f` |
///
/// \snippet fold_demo.cpp fold
///
/// Over a pack the accumulator may change type at every step (`foldl` of an `int`, then a `double`);
/// `scan` and the range forms keep one accumulator type, `z`'s. A tuple is folded as the pack of its
/// elements, so its elements may differ in type too. Use these when the elements are known at compile
/// time or are a heterogeneous pack; `std::accumulate` (already `constexpr`) is the fold over an
/// iterator range when that reads better.
///
/// **Relationships.** `snicholls::nest` (`iterate_n(f, x, n)` is it), `snicholls::nest_list` and
/// `snicholls::fold_list` (`scan` over a `std::array`) in `snicholls/core/iterate.hpp`;
/// `snicholls::constexpr_for` for a loop over indices; `meta::fold_t` for types;
/// `functional/combinators.hpp`. Group: \ref core.
///
/// **Implementations / variants.** A range and a tuple are distinct overloads of `foldl_of` /
/// `foldr_of`: a `std::array`, which is both, is folded as a range (the same result).
///
/// **Guarantees.** `constexpr`, allocation-free, no recursion at run time (the pack forms recurse at
/// compile time, one instantiation per element). `noexcept` exactly when `f` and the accumulator's
/// moves are (and, for a range, iterating it). The accumulator is moved from step to step; `scan`
/// copies each step into the result. `unfold` calls `step` exactly `N` times, in order.
///
/// **Errors.** None of their own.
///
/// **Examples.** \snippet fold_demo.cpp unfold
///
/// **Performance.** The calls to `f`; not benchmarked.
///
/// **Tests and demos.** `tests/functional/fold_test.cpp`, `demos/functional/fold_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<functional>`, `<iterator>`, `<ranges>`, `<tuple>`,
/// `<type_traits>`, `<utility>`, `snicholls/core/iterate.hpp` and `snicholls/core/functional/combinators.hpp`.
///
/// **Known issues and gotchas.** `foldr` over a pack of `n` elements nests `n` calls; at a few hundred
/// elements the compiler's template depth limit is reached. Fold a `std::array` with `foldr_of` instead.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_fold_hpp
#define snicholls_core_functional_fold_hpp

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/iterate.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    namespace detail {

        template <std::size_t K>
        struct foldl_step {
            template <class F, class T, class X0, class... X>
            static constexpr auto call(F& f, T&& acc, X0&& x0, X&&... x)
                SNICHOLLS_FN_RETURNS(foldl_step<K - 1>::call(f, std::invoke(f, std::forward<T>(acc), std::forward<X0>(x0)),
                                                            std::forward<X>(x)...))
        };

        template <>
        struct foldl_step<0> {
            template <class F, class T>
            static constexpr std::remove_cvref_t<T> call(F&, T&& acc) noexcept(
                std::is_nothrow_constructible_v<std::remove_cvref_t<T>, T>)
            {
                return std::forward<T>(acc);
            }
        };

        template <std::size_t K>
        struct foldr_step {
            template <class F, class T, class X0, class... X>
            static constexpr auto call(F& f, T&& z, X0&& x0, X&&... x)
                SNICHOLLS_FN_RETURNS(std::invoke(f, std::forward<X0>(x0),
                                                 foldr_step<K - 1>::call(f, std::forward<T>(z), std::forward<X>(x)...)))
        };

        template <>
        struct foldr_step<0> {
            template <class F, class T>
            static constexpr std::remove_cvref_t<T> call(F&, T&& z) noexcept(
                std::is_nothrow_constructible_v<std::remove_cvref_t<T>, T>)
            {
                return std::forward<T>(z);
            }
        };

        struct foldl_fn {
            template <class F, class T, class... X>
            constexpr auto operator()(F&& f, T&& z, X&&... x) const
                SNICHOLLS_FN_RETURNS(foldl_step<sizeof...(X)>::call(f, std::forward<T>(z), std::forward<X>(x)...))
        };

        struct foldr_fn {
            template <class F, class T, class... X>
            constexpr auto operator()(F&& f, T&& z, X&&... x) const
                SNICHOLLS_FN_RETURNS(foldr_step<sizeof...(X)>::call(f, std::forward<T>(z), std::forward<X>(x)...))
        };

        struct scan_fn {
            template <class F, class T, class... X, class V = std::remove_cvref_t<T>>
                requires std::copy_constructible<V> && (std::is_assignable_v<V&, std::invoke_result_t<F&, const V&, X>> && ...)
            constexpr std::array<V, sizeof...(X) + 1> operator()(F&& f, T&& z, X&&... x) const noexcept(
                std::is_nothrow_constructible_v<V, T> && std::is_nothrow_copy_constructible_v<V> &&
                (std::is_nothrow_invocable_v<F&, const V&, X> && ...) &&
                (std::is_nothrow_assignable_v<V&, std::invoke_result_t<F&, const V&, X>> && ...))
            {
                V acc(std::forward<T>(z));
                // A braced list is evaluated left to right: element i + 1 is step i applied to element i.
                return std::array<V, sizeof...(X) + 1>{{acc, (acc = std::invoke(f, std::as_const(acc), std::forward<X>(x)))...}};
            }
        };

        // Iterating R, begin to end, cannot throw.
        template <class R>
        concept nothrow_iterable = requires(R&& r, std::ranges::iterator_t<R> it) {
            { std::ranges::begin(r) } noexcept;
            { std::ranges::end(r) } noexcept;
            { ++it } noexcept;
            { *it } noexcept;
            { it == std::ranges::end(r) } noexcept;
        };

        template <class F, class V, class R>
        inline constexpr bool nothrow_range_foldl_v =
            nothrow_iterable<R> && std::is_nothrow_move_constructible_v<V> &&
            std::is_nothrow_invocable_v<F&, V, std::ranges::range_reference_t<R>> &&
            std::is_nothrow_assignable_v<V&, std::invoke_result_t<F&, V, std::ranges::range_reference_t<R>>>;

        template <class F, class V, class R>
        inline constexpr bool nothrow_range_foldr_v =
            nothrow_iterable<R> && std::is_nothrow_move_constructible_v<V> &&
            requires(std::ranges::iterator_t<R> it) {
                { --it } noexcept;
            } && std::is_nothrow_invocable_v<F&, std::ranges::range_reference_t<R>, V> &&
            std::is_nothrow_assignable_v<V&, std::invoke_result_t<F&, std::ranges::range_reference_t<R>, V>>;

        struct foldl_of_fn {
            template <class F, class T, std::ranges::input_range R, class V = std::remove_cvref_t<T>>
                requires std::is_assignable_v<V&, std::invoke_result_t<F&, V, std::ranges::range_reference_t<R>>>
            constexpr V operator()(F&& f, T&& z, R&& r) const
                noexcept(std::is_nothrow_constructible_v<V, T> && nothrow_range_foldl_v<F, V, R>)
            {
                V acc(std::forward<T>(z));
                for (auto it = std::ranges::begin(r); it != std::ranges::end(r); ++it)
                    acc = std::invoke(f, std::move(acc), *it);
                return acc;
            }

            template <class F, class T, std::size_t... J, class U>
            static constexpr auto each(F& f, T&& z, U&& u, std::index_sequence<J...>)
                SNICHOLLS_FN_RETURNS(foldl_fn{}(f, std::forward<T>(z), std::get<J>(std::forward<U>(u))...))

            template <class F, class T, tuple_like U>
                requires(!std::ranges::input_range<U>)
            constexpr auto operator()(F&& f, T&& z, U&& u) const
                SNICHOLLS_FN_RETURNS(foldl_of_fn::each(f, std::forward<T>(z), std::forward<U>(u), tuple_indices<U>{}))
        };

        struct foldr_of_fn {
            template <class F, class T, std::ranges::bidirectional_range R, class V = std::remove_cvref_t<T>>
                requires std::is_assignable_v<V&, std::invoke_result_t<F&, std::ranges::range_reference_t<R>, V>>
            constexpr V operator()(F&& f, T&& z, R&& r) const
                noexcept(std::is_nothrow_constructible_v<V, T> && nothrow_range_foldr_v<F, V, R>)
            {
                V acc(std::forward<T>(z));
                const auto first = std::ranges::begin(r);
                auto it = std::ranges::next(first, std::ranges::end(r));
                while (it != first) {
                    --it;
                    acc = std::invoke(f, *it, std::move(acc));
                }
                return acc;
            }

            template <class F, class T, std::size_t... J, class U>
            static constexpr auto each(F& f, T&& z, U&& u, std::index_sequence<J...>)
                SNICHOLLS_FN_RETURNS(foldr_fn{}(f, std::forward<T>(z), std::get<J>(std::forward<U>(u))...))

            template <class F, class T, tuple_like U>
                requires(!std::ranges::input_range<U>)
            constexpr auto operator()(F&& f, T&& z, U&& u) const
                SNICHOLLS_FN_RETURNS(foldr_of_fn::each(f, std::forward<T>(z), std::forward<U>(u), tuple_indices<U>{}))
        };

        struct iterate_op {
            template <std::size_t... I, class S, class X>
            static constexpr auto call(std::index_sequence<I...>, S&& s, X&& x)
                SNICHOLLS_FN_RETURNS(snicholls::nest(detail::part<0>(std::forward<S>(s)), std::forward<X>(x),
                                                     detail::part<1>(s)))
        };

        struct iterate_n_fn {
            template <class F>
            constexpr auto operator()(F&& f, std::size_t n) const noexcept(nothrow_store_v<F>)
            {
                return detail::make<iterate_op>(std::forward<F>(f), n);
            }

            template <class F, class X>
            constexpr auto operator()(F&& f, X&& x, std::size_t n) const
                SNICHOLLS_FN_RETURNS(snicholls::nest(std::forward<F>(f), std::forward<X>(x), n))
        };

    } // namespace detail

    /// \brief Left fold over a pack: `foldl(f, z, a, b, c)` is `f(f(f(z, a), b), c)`; `foldl(f, z)` is `z`.
    /// \ingroup core
    ///
    /// The accumulator may change type at every step. The result is a value (a copy of `z` when the pack is empty).
    inline constexpr detail::foldl_fn foldl{};

    /// \brief Right fold over a pack: `foldr(f, z, a, b, c)` is `f(a, f(b, f(c, z)))`; `foldr(f, z)` is `z`.
    /// \ingroup core
    inline constexpr detail::foldr_fn foldr{};

    /// \brief Left scan over a pack: `scan(f, z, a, b)` is `std::array{z, f(z, a), f(f(z, a), b)}`.
    /// \ingroup core
    ///
    /// `sizeof...(xs) + 1` values of `z`'s decayed type; each step's result is converted back to it. For a
    /// `std::array` of inputs, `snicholls::fold_list` is the same thing.
    inline constexpr detail::scan_fn scan{};

    /// \brief `foldl` over a sequence: an input range, or a `std::tuple` / `std::pair` as the pack of its elements.
    /// \ingroup core
    ///
    /// Over a range the accumulator keeps `z`'s decayed type and is moved into each call of `f`.
    inline constexpr detail::foldl_of_fn foldl_of{};

    /// \brief `foldr` over a sequence: a bidirectional range (walked from the back), or a tuple as the pack of its elements.
    /// \ingroup core
    inline constexpr detail::foldr_of_fn foldr_of{};

    /// \brief `iterate_n(f, n)` is the function `x -> f(f(...f(x)))`, `f` applied `n` times; `iterate_n(f, x, n)` applies it now.
    /// \ingroup core
    ///
    /// The functional power `f^n`. The value form is `snicholls::nest(f, x, n)` from `snicholls/core/iterate.hpp`,
    /// which this calls: `f` maps `x`'s type to itself.
    inline constexpr detail::iterate_n_fn iterate_n{};

    /// \brief Haskell's `unfoldr` for a fixed count: `N` values, each from `step(state)`, which returns `{value, next state}`.
    /// \ingroup core
    ///
    /// `unfold<5>(std::pair{0, 1}, fib_step)` is the first five Fibonacci numbers. `step` sees the state as const and
    /// returns a pair-like (`std::pair`, `std::tuple` of two); it is called exactly `N` times, in order.
    /// \snippet fold_demo.cpp unfold
    /// \tparam N    The number of values.
    /// \tparam S    The seed's type; the state is its decayed type.
    /// \tparam Step The step's type.
    /// \param seed The first state.
    /// \param step `state -> {value, next state}`.
    /// \return The values, a `std::array` of the step's first element type, decayed.
    template <std::size_t N, class S, class Step, class State = std::remove_cvref_t<S>,
              class R = std::remove_cvref_t<std::invoke_result_t<Step&, const State&>>,
              class V = std::remove_cvref_t<std::tuple_element_t<0, R>>>
    [[nodiscard]] constexpr std::array<V, N> unfold(S&& seed, Step&& step) noexcept(
        std::is_nothrow_constructible_v<State, S> && std::is_nothrow_invocable_v<Step&, const State&> &&
        std::is_nothrow_move_assignable_v<State> && std::is_nothrow_move_constructible_v<V>)
    {
        State state(std::forward<S>(seed));
        const auto next = [&]() {
            R r = std::invoke(step, std::as_const(state));
            state = std::get<1>(std::move(r));
            return V(std::get<0>(std::move(r)));
        };
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::array<V, N>{{(static_cast<void>(I), next())...}};
        }(std::make_index_sequence<N>{});
    }

} // namespace snicholls::fn

#endif /* snicholls_core_functional_fold_hpp */
