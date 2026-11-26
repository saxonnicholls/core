//
//  curry.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fn::curry`, `uncurry`, `partial` and `partial_right`: a function of many arguments as a chain of
///        functions of fewer, all `constexpr`.
/// \ingroup core
///
/// **Overview.** A curried function takes its arguments one at a time, or several at a time, and
/// calls the original once it has them all:
///
/// \snippet curry_demo.cpp curry
///
/// | here                         | what it returns                                        |
/// | ---------------------------- | ------------------------------------------------------ |
/// | `curry(f)`                   | `f`, curried over its own arity (one signature only)   |
/// | `curry<N>(f)`                | `f`, curried over `N` arguments (a generic lambda)     |
/// | `uncurry(g)`                 | `(a, b, c) -> g(a)(b)(c)`: a hand-curried chain flattened |
/// | `partial(f, a, b)`           | `(rest...) -> f(a, b, rest...)`                        |
/// | `partial_right(f, y, z)`     | `(rest...) -> f(rest..., y, z)`                        |
///
/// A partly applied curried function is an ordinary value: store it, pass it, compose it
/// (`fn::pipe(scale(fx), add(spread))`), call it in a `static_assert`. Use `curry` when the pieces of a
/// calculation are built in one place and finished in another (a rate fixed at start of day, the
/// price per tick); use `partial` when the bound arguments are known together and the rest come
/// later; use a lambda when neither is true.
///
/// The arity of `curry(f)` is read with `snicholls::callable_traits`, so `f` needs exactly one
/// signature: a function, a function pointer, a pointer to member function (the object counts as the
/// first argument), or a class with one non-template `operator()`. A generic lambda or an overload
/// set has no arity of its own and `curry(f)` refuses it with a message; say how many with
/// `curry<N>(f)`.
///
/// **Relationships.** `functional/combinators.hpp` (the closures, `compose`, `pipe`); `functional/bind.hpp`
/// (placeholders, for arguments in any order); `snicholls::callable_traits` (the arity);
/// `snicholls::invoke_with` (arguments that exist only at run time). Group: \ref core.
///
/// **Implementations / variants.** `curry(f)` and `curry<N>(f)` return the same closure; the first
/// only computes `N`. Each partial application returns a new closure holding `f` and the arguments so
/// far; the original is unchanged.
///
/// **Guarantees.** `constexpr`. Arguments are stored by value (decayed), copied from an lvalue
/// closure and moved from an rvalue one, so a partial application never dangles; pass `std::ref(x)`
/// to bind by reference. The final call is `noexcept` exactly when `f`'s call with the stored and
/// the given arguments is; a partial application is `noexcept` when copying the parts is. A closure is
/// trivially copyable when `f` and the stored arguments are. A curried function accepts at most the
/// remaining number of arguments; more is not invocable (`std::is_invocable_v` is false).
///
/// **Errors.** `curry(f)` of a callable without one signature fails to compile with
/// "snicholls::fn::curry: a generic or overloaded callable has no arity of its own; give it: curry<N>(f)"
/// (compile-fail test `fn_curry_generic_lambda`). Calling with arguments `f` cannot take is a compile
/// error at the final call.
///
/// **Examples.** \snippet curry_demo.cpp price
///
/// **Performance.** The curried price-adjustment pipeline of
/// `benchmarks/functional/combinators_benchmark.cpp` against the hand-written lambda over the same
/// 4096 prices, fairness-checked: medians 4347 ns and 4326 ns, equal within noise (macOS x86-64 Xeon W-3245, Apple Clang 17, Release, 2026-11-26, 5 repetitions). At `-O2`
/// both compile to the same instructions (Apple Clang 17 and Clang 23, arm64 and x86-64).
///
/// **Tests and demos.** `tests/functional/curry_test.cpp`, `tests/compile_fail/fn_curry_generic_lambda.cpp`,
/// `demos/functional/curry_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<type_traits>`, `<utility>`,
/// `snicholls/core/functional/combinators.hpp` and `snicholls/core/meta/callable_traits.hpp`.
///
/// **Known issues and gotchas.**
/// - A stored argument is passed to `f` as an lvalue when the closure is an lvalue: an `f` taking
///   `T&&` can only be finished through an rvalue closure (`std::move(c)(x)`), as with `std::bind`.
/// - A parameter `T&` binds the stored copy, not the caller's object, unless the caller passed `std::ref`.
/// - Default arguments do not count: `curry(f)` waits for every declared parameter.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_curry_hpp
#define snicholls_core_functional_curry_hpp

#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/meta/callable_traits.hpp>

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace snicholls::fn {

    namespace detail {

        // Part 0 is f, parts 1... the arguments so far.
        template <std::size_t N>
        struct curry_op {
            // The last arguments: call f.
            template <std::size_t... I, class S, class... A>
                requires(sizeof...(I) + sizeof...(A) == N)
            static constexpr auto call(std::index_sequence<0, I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)),
                                                 detail::part<I>(std::forward<S>(s))..., std::forward<A>(a)...))

            // Not yet: a new closure with these arguments added.
            template <std::size_t... I, class S, class... A>
                requires(sizeof...(A) > 0 && sizeof...(I) + sizeof...(A) < N)
            static constexpr auto call(std::index_sequence<0, I...>, S&& s, A&&... a) noexcept(
                nothrow_store_v<part_t<0, S>, part_t<I, S>..., A...>)
            {
                return detail::make<curry_op>(detail::part<0>(std::forward<S>(s)),
                                              detail::part<I>(std::forward<S>(s))..., std::forward<A>(a)...);
            }
        };

        // g(a0)(a1)...(ak): one call per argument.
        template <std::size_t K>
        struct apply_each {
            template <class G, class A0, class... A>
            static constexpr auto call(G&& g, A0&& a0, A&&... a)
                SNICHOLLS_FN_RETURNS(apply_each<K - 1>::call(std::invoke(std::forward<G>(g), std::forward<A0>(a0)),
                                                             std::forward<A>(a)...))
        };

        // No arguments: nothing to call (and no recursion for a compiler that substitutes before it checks).
        template <>
        struct apply_each<0> {};

        template <>
        struct apply_each<1> {
            template <class G, class A0>
            static constexpr auto call(G&& g, A0&& a0)
                SNICHOLLS_FN_RETURNS(std::invoke(std::forward<G>(g), std::forward<A0>(a0)))
        };

        struct uncurry_op {
            template <std::size_t... I, class S, class... A>
                requires(sizeof...(A) > 0)
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(apply_each<sizeof...(A)>::call(detail::part<0>(std::forward<S>(s)),
                                                                    std::forward<A>(a)...))
        };

        struct partial_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<0, I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)),
                                                 detail::part<I>(std::forward<S>(s))..., std::forward<A>(a)...))
        };

        struct partial_right_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<0, I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)), std::forward<A>(a)...,
                                                 detail::part<I>(std::forward<S>(s))...))
        };

        // The number of arguments f takes, the object of a pointer to member function included; 0 when f has
        // no single signature (curry has already said so).
        template <class F>
        consteval std::size_t arity_of() noexcept
        {
            return 0;
        }

        template <single_signature_callable F>
        consteval std::size_t arity_of() noexcept
        {
            return callable_traits<F>::arity + (callable_traits<F>::is_member ? 1 : 0);
        }

    } // namespace detail

    /// \brief `f` curried over `N` arguments: call it with them one at a time or several at a time.
    /// \ingroup core
    ///
    /// `curry<3>(f)(a)(b, c)`, `curry<3>(f)(a, b)(c)` and `curry<3>(f)(a, b, c)` are all `f(a, b, c)`.
    /// For a generic lambda, whose arity cannot be read.
    /// \tparam N The number of arguments to collect before calling `f`.
    /// \tparam F The callable's type; stored decayed.
    /// \param f The function.
    /// \return The curried function, holding a copy of `f`.
    template <std::size_t N, class F>
    [[nodiscard]] constexpr auto curry(F&& f) noexcept(detail::nothrow_store_v<F>)
    {
        return detail::make<detail::curry_op<N>>(std::forward<F>(f));
    }

    /// \brief `f` curried over its own arity, read with `snicholls::callable_traits`.
    /// \ingroup core
    ///
    /// For a function, a function pointer, a pointer to member function (the object is the first
    /// argument) or a class with one non-template `operator()`. A generic lambda is a compile error
    /// that names `curry<N>(f)`.
    /// \tparam F The callable's type; stored decayed.
    /// \param f The function.
    /// \return The curried function, holding a copy of `f`.
    template <class F>
    [[nodiscard]] constexpr auto curry(F&& f) noexcept(detail::nothrow_store_v<F>)
    {
        static_assert(single_signature_callable<F>,
                      "snicholls::fn::curry: a generic or overloaded callable has no arity of its own; give it: "
                      "curry<N>(f)");
        return fn::curry<detail::arity_of<F>()>(std::forward<F>(f));
    }

    /// \brief `uncurry(g)(a, b, c)` is `g(a)(b)(c)`: a chain of one-argument functions called as one function.
    /// \ingroup core
    ///
    /// The inverse of `curry` for a hand-curried function (`[](int a) { return [a](int b) { ... }; }`).
    /// Takes at least one argument. For Haskell's `uncurry` over a pair, compose with `fn::apply`:
    /// `fn::apply(fn::uncurry(g))(std::pair{a, b})`.
    inline constexpr detail::unary_maker<detail::uncurry_op> uncurry{};

    /// \brief `partial(f, a, b)(rest...)` is `f(a, b, rest...)`: the leading arguments bound.
    /// \ingroup core
    ///
    /// The bound arguments are stored by value; the closure takes any number of further arguments.
    inline constexpr detail::variadic_maker<detail::partial_op> partial{};

    /// \brief `partial_right(f, y, z)(rest...)` is `f(rest..., y, z)`: the trailing arguments bound.
    /// \ingroup core
    inline constexpr detail::variadic_maker<detail::partial_right_op> partial_right{};

    /// \brief A left section, Haskell's `(x op)`: `lsection(op, x)(y)` is `op(x, y)`; `lsection(std::multiplies<>{}, 2)` is `(2 *)`.
    /// \ingroup core
    inline constexpr detail::variadic_maker<detail::partial_op> lsection{};

    /// \brief A right section, Haskell's `(op y)`: `rsection(op, y)(x)` is `op(x, y)`; `rsection(std::minus<>{}, 1)` is `(subtract 1)`.
    /// \ingroup core
    inline constexpr detail::variadic_maker<detail::partial_right_op> rsection{};

} // namespace snicholls::fn

#endif /* snicholls_core_functional_curry_hpp */
