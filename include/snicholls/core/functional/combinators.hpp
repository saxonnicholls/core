//
//  combinators.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fn` combinators: functions that take functions and return functions, all `constexpr`.
/// \ingroup core
///
/// **Overview.** The classic combinators, each a function that takes callables and returns a
/// new callable, built and called at compile time or at run time alike:
///
/// | here                     | what it returns                                      | Haskell / maths      |
/// | ------------------------ | ---------------------------------------------------- | -------------------- |
/// | `identity`               | its argument                                         | `id`                 |
/// | `constant(x)`            | a function that ignores its arguments and returns `x` | `const x` (K)        |
/// | `flip(f)`                | `f` with its first two arguments swapped             | `flip f`             |
/// | `compose(f, g, h)`       | `x -> f(g(h(x)))`, right to left                     | `f . g . h`          |
/// | `pipe(f, g, h)`          | `x -> h(g(f(x)))`, left to right                     | `h . g . f`          |
/// | `x \| then(f) \| then(g)` | `g(f(x))`: a value through a pipe                     | `x & f & g`          |
/// | `on(f, p)`               | `(a, b) -> f(p(a), p(b))`                             | ``f `on` p``         |
/// | `apply(f)`               | `tuple -> f(elements...)`                            | `uncurry` on tuples  |
/// | `juxt(f, g)`             | `x -> tuple{f(x), g(x)}`                             | `f &&& g`            |
/// | `fix(f)`                 | `x -> f(self, x)`: recursion for lambdas             | `fix` (Y)            |
/// | `tap(f)`                 | `x -> (f(x), x)`: look, pass it on                   | `traceShowId`        |
/// | `when(p, f)`, `unless(p, f)` | `x -> p(x) ? f(x) : x`, and the converse         | `bool`-guarded `f`   |
/// | `call(f, x...)`          | `f(x...)` as a value                                 | `f $ x`, OCaml `f @@ x` |
/// | `until(p, f)`            | `x -> f(f(...x))` until `p` holds                    | `until p f`          |
/// | `apply_when(b, f)`       | `x -> b ? f(x) : x`                                  | `applyWhen b f`      |
/// | `bool_(f, t, b)`         | `b ? t : f`                                          | `bool f t b`         |
/// | `rotate_args(f)`, `swap_args(f)` | `(a, b, c) -> f(b, c, a)`; `flip(f)`         | argument reordering  |
///
/// \snippet combinators_demo.cpp compose
///
/// Every entry point without an explicit template argument is itself a `constexpr` function
/// object, so it can be passed, composed and curried like any other function:
/// `fn::curry<2>(fn::compose)`, `fn::juxt(fn::identity, f)`. Use these when the function is the
/// value being built (a pricing step assembled from pieces, a projection passed to an algorithm,
/// a recursive lambda); write the plain lambda when it is used once and reads better that way.
///
/// **Relationships.** The rest of `functional/`: `curry.hpp` (currying, partial application,
/// sections), `bind.hpp` (placeholders), `typeclasses.hpp` and `instances.hpp` (Functor, Monad,
/// Monoid, ...), `state.hpp` (Reader, Writer, State), `monad.hpp` (Maybe / Either helpers), `fold.hpp`
/// and `list.hpp` (folds, Data.List), `lazy.hpp` (thunks, streams), `match.hpp` (pattern matching),
/// `recursion.hpp`, `tabulate.hpp` (compile-time memoisation), `types.hpp`, `ocaml.hpp`. `snicholls::overloaded` / `first_match` join callables into one
/// overload set; `snicholls::function_ref` refers to one without owning it; `snicholls::pipeline`
/// stages are observer chains that push values to a sink, where `then` pulls one value through.
/// `std::identity` and `std::not_fn` are the standard library's. Group: \ref core.
///
/// **Implementations / variants.** One. Every returned callable is one class template,
/// `fn::detail::closure<Op, Parts...>`: the parts stored by value, and an empty policy `Op` saying
/// what a call does. It has the four `operator()` overloads (`&`, `const&`, `&&`, `const&&`), so a
/// call through an rvalue closure moves its parts into the call (C++23's `std::forward_like`
/// without C++23), and a call through a const closure calls the parts as const.
///
/// **Guarantees.**
/// - `constexpr`: every combinator and every call, usable in `static_assert`.
/// - `noexcept` exactly when the wrapped calls and the copies made are: a closure's call is
///   `noexcept(noexcept(...))` of what it does, and building one is `noexcept` when decaying the
///   arguments into it is.
/// - SFINAE-friendly: `std::is_invocable_v` of a closure is false, not a hard error, when the
///   wrapped call is ill-formed (except `when` / `unless` / `tap`, which are constrained instead).
/// - Ownership: the parts are decayed copies (or moves of rvalues); nothing refers to its
///   arguments, so a closure can outlive them. Pass `std::ref(x)` to refer instead.
/// - Layout: a closure holds its parts and nothing else. It is trivially copyable when they are,
///   an aggregate, and an empty part (a captureless lambda, a standard function object) takes no
///   space: it is an empty base, not a member. No allocation, no virtual call, no type erasure.
/// - Results are values: `identity`, `tap` and `when` return an lvalue argument as the same lvalue
///   and an rvalue argument as a new value moved from it, never a reference to a temporary;
///   `constant` returns a copy (or, from an rvalue closure, a move) of its value.
///
/// **Errors.** None at run time. A call the parts cannot make is a compile error at the call
/// site, or `std::is_invocable_v` false.
///
/// **Performance.** A composed or curried pipeline compiles to the same code as the hand-written
/// lambda. `benchmarks/functional/combinators_benchmark.cpp`, `pipe` of three curried pieces against
/// the lambda over 4096 prices, fairness-checked (identical sums): medians 4347 ns and 4326 ns, equal
/// within noise (CV under 1%; macOS x86-64 Xeon W-3245, Apple Clang 17, Release, 2026-11-26, 5 repetitions). At `-O2` the two functions compile to identical instructions
/// (Apple Clang 17 and Clang 23, arm64; x86-64 identical but for constant-pool label names).
///
/// **Tests and demos.** `tests/functional/combinators_test.cpp`, `demos/functional/combinators_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>` (for `std::invoke`), `<tuple>`, `<type_traits>`,
/// `<utility>`, and `snicholls/core/overloaded.hpp` (for `SNICHOLLS_EMPTY_BASES`).
///
/// **Known issues and gotchas.**
/// - `fix(f)`: `f` takes the recursive function as its first parameter and must state its return
///   type (`[](auto& self, int n) -> int { ... }`): a deduced return type would need itself.
/// - A function pointer to an overload set or a function template has no single address: wrap it
///   in a lambda (`[](auto x) { return std::abs(x); }`), as for any standard algorithm.
/// - `then` is spelled `x | fn::then(f)`, not `x | f`: a bare callable on the right would capture
///   every `|` in the program. `snicholls::pipeline`'s `|` composes stage specs and is unaffected.
/// - Two parts of the same empty type (`compose(f, g, f)`) cannot share an address, so each takes
///   a byte. `[[no_unique_address]]` is not used: GCC 13 rejects reading such parts in a constant
///   expression and Clang 18 crashes on them, so empty parts are empty bases instead. A `final`
///   empty class is stored as a member and takes a byte.
///
/// **Status.** Experimental, new in 0.27.0.

#ifndef snicholls_core_functional_combinators_hpp
#define snicholls_core_functional_combinators_hpp

#include <snicholls/core/overloaded.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

/// \brief The body of a function that returns one expression, with the expression's exact `noexcept`, its exact
///        type, and substitution failure (not a hard error) when it is ill-formed.
/// \ingroup core
///
/// `template <class F> constexpr auto call(F&& f) SNICHOLLS_FN_RETURNS(std::invoke(std::forward<F>(f)))`.
#define SNICHOLLS_FN_RETURNS(...)                                                                                       \
    noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) { return __VA_ARGS__; }

namespace snicholls::fn {

    namespace detail {

        // One stored part, tagged by its owner's part list and its position, so that two parts of the same type,
        // or a part of a part, are always distinct bases. An empty class part is a base (the empty-base
        // optimisation; [[no_unique_address]] breaks GCC 13's constant evaluation of repeated empty types and
        // crashes Clang 18); anything else is a member.
        template <class... Ts>
        struct owner_tag;

        template <class T>
        inline constexpr bool empty_base_v = std::is_empty_v<T> && !std::is_final_v<T>;

        template <class Tag, std::size_t I, class T, bool = empty_base_v<T>>
        struct slot {
            T value;
        };

        template <class Tag, std::size_t I, class T>
        struct slot<Tag, I, T, true> : T {};

        template <class Is, class... Ts>
        struct slots_impl;

        template <std::size_t... Is, class... Ts>
        struct SNICHOLLS_EMPTY_BASES slots_impl<std::index_sequence<Is...>, Ts...> : slot<owner_tag<Ts...>, Is, Ts>... {};

        // The parts of a closure: an aggregate, trivially copyable when every part is.
        template <class... Ts>
        using slots = slots_impl<std::index_sequence_for<Ts...>, Ts...>;

        template <class Tag, std::size_t I, class T>
        constexpr T& slot_get(slot<Tag, I, T, false>& s) noexcept { return s.value; }
        template <class Tag, std::size_t I, class T>
        constexpr const T& slot_get(const slot<Tag, I, T, false>& s) noexcept { return s.value; }
        template <class Tag, std::size_t I, class T>
        constexpr T&& slot_get(slot<Tag, I, T, false>&& s) noexcept { return static_cast<T&&>(s.value); }
        template <class Tag, std::size_t I, class T>
        constexpr const T&& slot_get(const slot<Tag, I, T, false>&& s) noexcept { return static_cast<const T&&>(s.value); }
        template <class Tag, std::size_t I, class T>
        constexpr T& slot_get(slot<Tag, I, T, true>& s) noexcept { return s; }
        template <class Tag, std::size_t I, class T>
        constexpr const T& slot_get(const slot<Tag, I, T, true>& s) noexcept { return s; }
        template <class Tag, std::size_t I, class T>
        constexpr T&& slot_get(slot<Tag, I, T, true>&& s) noexcept { return static_cast<T&&>(s); }
        template <class Tag, std::size_t I, class T>
        constexpr const T&& slot_get(const slot<Tag, I, T, true>&& s) noexcept { return static_cast<const T&&>(s); }

        // Part I of a closure, with the closure's constness and value category (std::forward_like).
        template <std::size_t I, class S>
        constexpr auto part(S&& s) noexcept
            -> decltype(detail::slot_get<typename std::remove_cvref_t<S>::tag, I>(std::forward<S>(s).parts))
        {
            return detail::slot_get<typename std::remove_cvref_t<S>::tag, I>(std::forward<S>(s).parts);
        }

        template <std::size_t I, class S>
        using part_t = decltype(detail::part<I>(std::declval<S>()));

        // Every closure: the parts, and Op::call(indices, self, args...) as the call, through each
        // qualification of the closure (deducing `this` before C++23).
        template <class Op, class... Ts>
        struct SNICHOLLS_EMPTY_BASES closure {
            using indices = std::index_sequence_for<Ts...>;
            using tag = owner_tag<Ts...>;

            slots<Ts...> parts;

            template <class... A>
            constexpr auto operator()(A&&... a) & SNICHOLLS_FN_RETURNS(Op::call(indices{}, *this, std::forward<A>(a)...))

            template <class... A>
            constexpr auto operator()(A&&... a) const& SNICHOLLS_FN_RETURNS(Op::call(indices{}, *this,
                                                                                      std::forward<A>(a)...))

            template <class... A>
            constexpr auto operator()(A&&... a) && SNICHOLLS_FN_RETURNS(Op::call(indices{}, std::move(*this),
                                                                                  std::forward<A>(a)...))

            template <class... A>
            constexpr auto operator()(A&&... a) const&& SNICHOLLS_FN_RETURNS(Op::call(indices{}, std::move(*this),
                                                                                       std::forward<A>(a)...))
        };

        // Building a closure decays each argument into a part.
        template <class... X>
        inline constexpr bool nothrow_store_v =
            (std::is_nothrow_constructible_v<std::decay_t<X>, X> && ...) &&
            (std::is_nothrow_move_constructible_v<std::decay_t<X>> && ...);

        template <class Op, class... X>
        constexpr closure<Op, std::decay_t<X>...> make(X&&... x) noexcept(nothrow_store_v<X...>)
        {
            return {{{std::forward<X>(x)}...}};
        }

        // f(get<J>(t)...) for a std::tuple, std::pair or std::array, SFINAE-friendly (std::apply is not).
        template <class F, class T, std::size_t... J>
        constexpr auto apply_tuple(F&& f, T&& t, std::index_sequence<J...>)
            SNICHOLLS_FN_RETURNS(std::invoke(std::forward<F>(f), std::get<J>(std::forward<T>(t))...))

        template <class T>
        using tuple_indices = std::make_index_sequence<std::tuple_size<std::remove_cvref_t<T>>::value>;

        template <class T>
        concept tuple_like = requires { std::tuple_size<std::remove_cvref_t<T>>::value; };

        template <class T>
        inline constexpr bool is_std_array_v = false;
        template <class T, std::size_t N>
        inline constexpr bool is_std_array_v<std::array<T, N>> = true;

        template <class A>
        concept std_array = is_std_array_v<std::remove_cvref_t<A>>;

        // ---- the policies -------------------------------------------------------------------------------------

        struct identity_fn {
            template <class T>
            constexpr T operator()(T&& x) const noexcept(std::is_nothrow_constructible_v<T, T&&>)
            {
                return std::forward<T>(x);
            }
        };

        struct constant_op {
            template <std::size_t... I, class S, class... A, class V = std::remove_cvref_t<part_t<0, S>>>
            static constexpr V call(std::index_sequence<I...>, S&& s,
                                    A&&...) noexcept(std::is_nothrow_constructible_v<V, part_t<0, S>>)
            {
                return detail::part<0>(std::forward<S>(s));
            }
        };

        struct flip_op {
            template <std::size_t... I, class S, class A0, class A1, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A0&& a0, A1&& a1, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)), std::forward<A1>(a1),
                                                 std::forward<A0>(a0), std::forward<A>(a)...))
        };

        // part 0 after part 1: f(g(args...)).
        struct compose_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)),
                                                 std::invoke(detail::part<1>(std::forward<S>(s)),
                                                             std::forward<A>(a)...)))
        };

        struct on_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)),
                                                 std::invoke(detail::part<1>(s), std::forward<A>(a))...))
        };

        struct apply_op {
            template <std::size_t... I, class S, class T>
            static constexpr auto call(std::index_sequence<I...>, S&& s, T&& t)
                SNICHOLLS_FN_RETURNS(detail::apply_tuple(detail::part<0>(std::forward<S>(s)), std::forward<T>(t),
                                                         tuple_indices<T>{}))
        };

        struct juxt_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a) SNICHOLLS_FN_RETURNS(
                std::tuple<std::remove_cvref_t<decltype(std::invoke(detail::part<I>(std::forward<S>(s)), a...))>...>{
                    std::invoke(detail::part<I>(std::forward<S>(s)), a...)...})
        };

        // f(self, args...): self is the closure, so f recurses by calling it.
        struct fix_op {
            template <std::size_t... I, class S, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)), s, std::forward<A>(a)...))
        };

        struct tap_op {
            template <std::size_t... I, class S, class X>
                requires std::is_invocable_v<part_t<0, S>, const std::remove_reference_t<X>&>
            static constexpr X call(std::index_sequence<I...>, S&& s, X&& x) noexcept(
                std::is_nothrow_invocable_v<part_t<0, S>, const std::remove_reference_t<X>&> &&
                std::is_nothrow_constructible_v<X, X&&>)
            {
                static_cast<void>(std::invoke(detail::part<0>(std::forward<S>(s)), std::as_const(x)));
                return std::forward<X>(x);
            }
        };

        // p(x) == Apply ? f(x) : x; `when` applies on true, `unless` on false.
        template <bool Apply>
        struct guarded_op {
            template <std::size_t... I, class S, class X, class R = std::remove_cvref_t<X>>
                requires std::is_invocable_r_v<bool, part_t<0, S&>, const std::remove_reference_t<X>&> &&
                         std::is_invocable_r_v<R, part_t<1, S>, X> && std::is_constructible_v<R, X>
            static constexpr R call(std::index_sequence<I...>, S&& s, X&& x) noexcept(
                std::is_nothrow_invocable_r_v<bool, part_t<0, S&>, const std::remove_reference_t<X>&> &&
                std::is_nothrow_invocable_r_v<R, part_t<1, S>, X> && std::is_nothrow_constructible_v<R, X>)
            {
                if (static_cast<bool>(std::invoke(detail::part<0>(s), std::as_const(x))) == Apply)
                    return std::invoke(detail::part<1>(std::forward<S>(s)), std::forward<X>(x));
                return R(std::forward<X>(x));
            }
        };

        // f(b, c, ..., a): the first argument moved to the end.
        struct rotate_op {
            template <std::size_t... I, class S, class A0, class... A>
            static constexpr auto call(std::index_sequence<I...>, S&& s, A0&& a0, A&&... a)
                SNICHOLLS_FN_RETURNS(std::invoke(detail::part<0>(std::forward<S>(s)), std::forward<A>(a)...,
                                                 std::forward<A0>(a0)))
        };

        // x, f(x), f(f(x)), ... until p holds.
        struct until_op {
            template <std::size_t... I, class S, class X, class R = std::remove_cvref_t<X>>
                requires std::is_invocable_r_v<bool, part_t<0, S&>, const R&> && std::is_invocable_r_v<R, part_t<1, S&>, R>
            static constexpr R call(std::index_sequence<I...>, S&& s, X&& x) noexcept(
                std::is_nothrow_constructible_v<R, X> && std::is_nothrow_move_assignable_v<R> &&
                std::is_nothrow_invocable_r_v<bool, part_t<0, S&>, const R&> && std::is_nothrow_invocable_r_v<R, part_t<1, S&>, R>)
            {
                R v(std::forward<X>(x));
                while (!static_cast<bool>(std::invoke(detail::part<0>(s), std::as_const(v))))
                    v = std::invoke(detail::part<1>(s), std::move(v));
                return v;
            }
        };

        // b ? f(x) : x, b fixed when built.
        struct apply_when_op {
            template <std::size_t... I, class S, class X, class R = std::remove_cvref_t<X>>
                requires std::is_invocable_r_v<R, part_t<1, S>, X> && std::is_constructible_v<R, X>
            static constexpr R call(std::index_sequence<I...>, S&& s, X&& x) noexcept(
                std::is_nothrow_invocable_r_v<R, part_t<1, S>, X> && std::is_nothrow_constructible_v<R, X>)
            {
                if (detail::part<0>(s)) return std::invoke(detail::part<1>(std::forward<S>(s)), std::forward<X>(x));
                return R(std::forward<X>(x));
            }
        };

        struct call_fn {
            template <class F, class... A>
            constexpr auto operator()(F&& f, A&&... a) const
                SNICHOLLS_FN_RETURNS(std::invoke(std::forward<F>(f), std::forward<A>(a)...))
        };

        struct bool_fn {
            template <class F, class T, class R = std::common_type_t<std::decay_t<F>, std::decay_t<T>>>
            constexpr R operator()(F&& if_false, T&& if_true, bool b) const
                noexcept(std::is_nothrow_constructible_v<R, F> && std::is_nothrow_constructible_v<R, T>)
            {
                return b ? R(std::forward<T>(if_true)) : R(std::forward<F>(if_false));
            }
        };

        struct apply_when_fn {
            template <class F>
            constexpr auto operator()(bool b, F&& f) const noexcept(nothrow_store_v<F>)
            {
                return detail::make<apply_when_op>(b, std::forward<F>(f));
            }
        };

        // ---- the entry points ---------------------------------------------------------------------------------

        struct constant_fn {
            template <class T>
            constexpr auto operator()(T&& x) const noexcept(nothrow_store_v<T>)
            {
                return detail::make<constant_op>(std::forward<T>(x));
            }
        };

        struct flip_fn {
            template <class F>
            constexpr auto operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return detail::make<flip_op>(std::forward<F>(f));
            }
        };

        struct compose_fn {
            template <class F>
            constexpr std::decay_t<F> operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return std::forward<F>(f);
            }

            template <class F, class G, class... H>
            constexpr auto operator()(F&& f, G&& g, H&&... h) const noexcept(nothrow_store_v<F, G, H...>)
            {
                return detail::make<compose_op>(std::forward<F>(f), (*this)(std::forward<G>(g), std::forward<H>(h)...));
            }
        };

        struct pipe_fn {
            template <class F>
            constexpr std::decay_t<F> operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return std::forward<F>(f);
            }

            template <class F, class G, class... H>
            constexpr auto operator()(F&& f, G&& g, H&&... h) const noexcept(nothrow_store_v<F, G, H...>)
            {
                return (*this)(detail::make<compose_op>(std::forward<G>(g), std::forward<F>(f)), std::forward<H>(h)...);
            }
        };

        // The right operand of a value pipe: `x | then(f)` is f(x). The operator is found only through then_fn (by
        // argument-dependent lookup), so it never meets another library's `|`.
        template <class F>
        struct then_fn {
            F f;
        };

        template <class X, class F>
        constexpr auto operator|(X&& x, const then_fn<F>& t) SNICHOLLS_FN_RETURNS(std::invoke(t.f, std::forward<X>(x)))

        template <class X, class F>
        constexpr auto operator|(X&& x, then_fn<F>&& t)
            SNICHOLLS_FN_RETURNS(std::invoke(std::move(t.f), std::forward<X>(x)))

        struct then_maker {
            template <class F>
            constexpr then_fn<std::decay_t<F>> operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return {std::forward<F>(f)};
            }
        };

        template <class Op>
        struct binary_maker {
            template <class F, class G>
            constexpr auto operator()(F&& f, G&& g) const noexcept(nothrow_store_v<F, G>)
            {
                return detail::make<Op>(std::forward<F>(f), std::forward<G>(g));
            }
        };

        template <class Op>
        struct unary_maker {
            template <class F>
            constexpr auto operator()(F&& f) const noexcept(nothrow_store_v<F>)
            {
                return detail::make<Op>(std::forward<F>(f));
            }
        };

        template <class Op>
        struct variadic_maker {
            template <class F, class... X>
            constexpr auto operator()(F&& f, X&&... x) const noexcept(nothrow_store_v<F, X...>)
            {
                return detail::make<Op>(std::forward<F>(f), std::forward<X>(x)...);
            }
        };

        struct juxt_fn {
            template <class... F>
                requires(sizeof...(F) > 0)
            constexpr auto operator()(F&&... f) const noexcept(nothrow_store_v<F...>)
            {
                return detail::make<juxt_op>(std::forward<F>(f)...);
            }
        };

    } // namespace detail

    /// \brief `id`: returns its argument, an lvalue as the same lvalue and an rvalue as a value moved from it.
    /// \ingroup core
    ///
    /// Unlike `std::identity`, an rvalue comes back as a value, so `auto&& r = fn::identity(make())` does not
    /// dangle. `noexcept` when moving the argument is.
    inline constexpr detail::identity_fn identity{};

    /// \brief The K combinator: `constant(x)` is a function of any arguments that returns a copy of `x`.
    /// \ingroup core
    ///
    /// `constant(x)(a, b, c) == x`. The value is stored by value (decayed); a call through an rvalue
    /// closure moves it out.
    inline constexpr detail::constant_fn constant{};

    /// \brief `flip(f)(a, b, rest...)` is `f(b, a, rest...)`: the first two arguments swapped.
    /// \ingroup core
    inline constexpr detail::flip_fn flip{};

    /// \brief `compose(f, g, h)(args...)` is `f(g(h(args...)))`: right to left, as `f . g . h` in mathematics.
    /// \ingroup core
    ///
    /// The rightmost function takes all the arguments; every other takes the result of the one to its right.
    /// `compose(f)` is a copy of `f`. Each function is stored by value.
    inline constexpr detail::compose_fn compose{};

    /// \brief `pipe(f, g, h)(args...)` is `h(g(f(args...)))`: left to right, in the order the steps run.
    /// \ingroup core
    ///
    /// `pipe(f, g)` is `compose(g, f)`. The leftmost function takes all the arguments.
    inline constexpr detail::pipe_fn pipe{};

    /// \brief The right operand of a value pipe: `x | then(f) | then(g)` is `g(f(x))`.
    /// \ingroup core
    ///
    /// Applies a function to a value now, where `pipe` builds the function for later.
    /// \snippet combinators_demo.cpp then
    inline constexpr detail::then_maker then{};

    /// \brief Haskell's ``on``: `on(f, p)(a, b, ...)` is `f(p(a), p(b), ...)`, the projection applied to every argument.
    /// \ingroup core
    ///
    /// `on(std::less<>{}, &order::price)` compares orders by price. The projection is called as an lvalue,
    /// once per argument.
    inline constexpr detail::binary_maker<detail::on_op> on{};

    /// \brief `apply(f)(t)` is `f(get<0>(t), get<1>(t), ...)` for a `std::tuple`, `std::pair` or `std::array`.
    /// \ingroup core
    ///
    /// The elements are forwarded with the tuple's value category. Unlike `std::apply`, the closure is
    /// SFINAE-friendly.
    inline constexpr detail::unary_maker<detail::apply_op> apply{};

    /// \brief Clojure's `juxt`: `juxt(f, g, h)(args...)` is `std::tuple{f(args...), g(args...), h(args...)}`.
    /// \ingroup core
    ///
    /// The arguments are passed to every function as lvalues, so none is moved from twice; the results
    /// are stored by value. Every function must return a value (not `void`).
    inline constexpr detail::juxt_fn juxt{};

    /// \brief The Y combinator: `fix(f)(args...)` is `f(self, args...)`, where `self` is `fix(f)` itself.
    /// \ingroup core
    ///
    /// Recursion for lambdas without `std::function`: `fix([](auto& self, int n) -> int { return n < 2 ? 1 :
    /// n * self(n - 1); })(5) == 120`, at compile time. `f` must state its return type.
    /// \snippet combinators_demo.cpp fix
    inline constexpr detail::unary_maker<detail::fix_op> fix{};

    /// \brief `tap(f)(x)` calls `f(x)` (as const) for its effect and returns `x`.
    /// \ingroup core
    ///
    /// For logging, counting or asserting inside a pipe. `f`'s result is discarded. An lvalue comes back as
    /// the same lvalue, an rvalue as a value moved from it.
    inline constexpr detail::unary_maker<detail::tap_op> tap{};

    /// \brief `when(p, f)(x)` is `p(x) ? f(x) : x`, the result in `x`'s decayed type.
    /// \ingroup core
    ///
    /// `p` sees `x` as const; `f(x)` must convert to `x`'s type.
    inline constexpr detail::binary_maker<detail::guarded_op<true>> when{};

    /// \brief `unless(p, f)(x)` is `p(x) ? x : f(x)`, the result in `x`'s decayed type.
    /// \ingroup core
    inline constexpr detail::binary_maker<detail::guarded_op<false>> unless{};

    /// \brief Haskell's `$` and OCaml's `@@`: `call(f, args...)` is `f(args...)` (`std::invoke`), as a value.
    /// \ingroup core
    ///
    /// The reverse, Haskell's `&` and OCaml's `|>`, is `x | then(f)`.
    inline constexpr detail::call_fn call{};

    /// \brief Haskell's `until`: `until(p, f)(x)` applies `f` to `x` until `p` holds, `x` itself if it already does.
    /// \ingroup core
    inline constexpr detail::binary_maker<detail::until_op> until{};

    /// \brief Data.Function's `applyWhen`: `apply_when(b, f)(x)` is `b ? f(x) : x`, `b` fixed when it is built.
    /// \ingroup core
    inline constexpr detail::apply_when_fn apply_when{};

    /// \brief Haskell's `bool`: `bool_(if_false, if_true, b)`, the case analysis of a `bool`, false first as in Haskell.
    /// \ingroup core
    inline constexpr detail::bool_fn bool_{};

    /// \brief `rotate_args(f)(a, b, c)` is `f(b, c, a)`: the first argument moved to the end (OCaml's argument reordering).
    /// \ingroup core
    ///
    /// `flip` swaps the first two; `rotate_args` rotates them all; `bind` with placeholders reorders arbitrarily.
    inline constexpr detail::unary_maker<detail::rotate_op> rotate_args{};

    /// \brief `swap_args(f)`: `flip(f)` under the name the other languages' libraries use.
    /// \ingroup core
    inline constexpr detail::flip_fn swap_args{};

} // namespace snicholls::fn

#endif /* snicholls_core_functional_combinators_hpp */
